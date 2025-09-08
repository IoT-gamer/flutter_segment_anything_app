import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:gal/gal.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

/// A class to store data for processing in a separate isolate.
class IsolateParams {
  final List<double> lowResMasksList;
  final List<double> iouPredictionsList;
  final int originalWidth;
  final int originalHeight;
  final Uint8List originalImageBytes;
  final bool fillHoles;
  final bool removeIslands;

  IsolateParams({
    required this.lowResMasksList,
    required this.iouPredictionsList,
    required this.originalWidth,
    required this.originalHeight,
    required this.originalImageBytes,
    required this.fillHoles,
    required this.removeIslands,
  });
}

/// A class to hold the results from the isolate.
class IsolateResult {
  final Uint8List maskImageBytes;
  IsolateResult(this.maskImageBytes);
}

/// A top-level function to run the heavy image processing in an isolate.
Future<IsolateResult> _processAndCompositeInIsolate(
  IsolateParams params,
) async {
  final List<double> masksData = params.lowResMasksList;
  final List<double> iouData = params.iouPredictionsList;

  int bestMaskIdx = 0;
  if (iouData.isNotEmpty) {
    for (int i = 1; i < iouData.length; i++) {
      if (iouData[i] > iouData[bestMaskIdx]) {
        bestMaskIdx = i;
      }
    }
  }

  final int maskSize = 256 * 256;
  final int maskOffset = bestMaskIdx * maskSize;
  final maskSlice = masksData.sublist(maskOffset, maskOffset + maskSize);

  final binaryMaskImage = img.Image(
    width: 256,
    height: 256,
    format: img.Format.uint8,
    numChannels: 1,
  );
  for (int y = 0; y < 256; y++) {
    for (int x = 0; x < 256; x++) {
      final value = maskSlice[y * 256 + x] > 0 ? 255 : 0;
      binaryMaskImage.setPixel(x, y, img.ColorRgb8(value, value, value));
    }
  }

  final bmpBytes = img.encodeBmp(binaryMaskImage);

  var processedMat = cv.imdecode(bmpBytes, cv.IMREAD_GRAYSCALE);

  final kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5));

  if (params.fillHoles) {
    processedMat = cv.morphologyEx(processedMat, cv.MORPH_CLOSE, kernel);
  }

  if (params.removeIslands) {
    processedMat = cv.morphologyEx(processedMat, cv.MORPH_OPEN, kernel);
  }

  final lowResImage = img.Image(width: 256, height: 256, numChannels: 4);
  final processedBytes = processedMat.data;

  for (int y = 0; y < 256; y++) {
    for (int x = 0; x < 256; x++) {
      final pixelValue = processedBytes[y * 256 + x];
      if (pixelValue > 0) {
        lowResImage.setPixelRgba(x, y, 0, 255, 0, 150);
      }
    }
  }

  kernel.dispose();
  processedMat.dispose();

  final finalMask = img.copyResize(
    lowResImage,
    width: params.originalWidth,
    height: params.originalHeight,
    interpolation: img.Interpolation.linear,
  );
  final maskBytes = img.encodePng(finalMask);

  return IsolateResult(Uint8List.fromList(maskBytes));
}

/// A class to store a segmentation point's coordinates and label.
class SegmentationPoint {
  SegmentationPoint({required this.point, required this.label});
  final Offset point;
  final int label;
}

class PointPainter extends CustomPainter {
  PointPainter(this.points, this.originalImageSize);
  final List<SegmentationPoint> points;
  final Size originalImageSize;
  @override
  void paint(Canvas canvas, Size size) {
    final Paint positivePaint = Paint()
      ..color = Colors.green.withAlpha(200)
      ..style = PaintingStyle.fill;
    final Paint negativePaint = Paint()
      ..color = Colors.red.withAlpha(200)
      ..style = PaintingStyle.fill;
    final Paint borderPaint = Paint()
      ..color = Colors.white.withAlpha(220)
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    if (originalImageSize.isEmpty) return;

    final fittedSizes = applyBoxFit(BoxFit.contain, originalImageSize, size);
    final Size destSize = fittedSizes.destination;
    final double dx = (size.width - destSize.width) / 2;
    final double dy = (size.height - destSize.height) / 2;
    final Rect destRect = Rect.fromLTWH(
      dx,
      dy,
      destSize.width,
      destSize.height,
    );
    for (final p in points) {
      final double widgetX =
          (p.point.dx * (destRect.width / originalImageSize.width)) +
          destRect.left;
      final double widgetY =
          (p.point.dy * (destRect.height / originalImageSize.height)) +
          destRect.top;
      final paint = p.label == 1 ? positivePaint : negativePaint;
      final pointOffset = Offset(widgetX, widgetY);

      canvas.drawCircle(pointOffset, 8.0, paint);
      canvas.drawCircle(pointOffset, 8.0, borderPaint);
    }
  }

  @override
  bool shouldRepaint(covariant PointPainter oldDelegate) {
    return oldDelegate.points.length != points.length ||
        oldDelegate.originalImageSize != originalImageSize;
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EdgeTAM Segmentation',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const SegmentationPage(),
    );
  }
}

class SegmentationPage extends StatefulWidget {
  const SegmentationPage({super.key});

  @override
  State<SegmentationPage> createState() => _SegmentationPageState();
}

class _SegmentationPageState extends State<SegmentationPage> {
  static const _encoderAssetPath = 'assets/models/edgetam_encoder.onnx';
  static const _decoderAssetPath = 'assets/models/edgetam_decoder.onnx';
  static const _modelInputSize = 1024;

  OrtSession? _encoderSession;
  OrtSession? _decoderSession;
  bool _isProcessing = false;
  File? _imageFile;
  img.Image? _originalImage;
  img.Image? _maskImage;
  Uint8List? _displayImageData;
  Uint8List? _maskImageData; // Holds the bytes for the mask overlay
  final GlobalKey _imageKey = GlobalKey();
  final TransformationController _transformationController =
      TransformationController();
  final List<SegmentationPoint> _points = [];
  int _currentPointLabel = 1;

  // New state variables for post-processing options
  bool _fillHoles = false;
  bool _removeIslands = false;

  @override
  void initState() {
    super.initState();
    _initOrtSessions();
  }

  Future<void> _initOrtSessions() async {
    _encoderSession = await OnnxRuntime().createSessionFromAsset(
      _encoderAssetPath,
    );
    _decoderSession = await OnnxRuntime().createSessionFromAsset(
      _decoderAssetPath,
    );
    if (mounted) setState(() {});
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      final imageFile = File(pickedFile.path);
      final imageBytes = await imageFile.readAsBytes();
      final originalImage = img.decodeImage(imageBytes);

      setState(() {
        _imageFile = imageFile;
        _originalImage = originalImage;
        _displayImageData = imageBytes;
        _maskImage = null;
        _maskImageData = null; // Clear mask from previous image
        _points.clear();
        _transformationController.value = Matrix4.identity();
      });
    }
  }

  void _handleTapUp(TapUpDetails details) {
    if (_imageFile == null || _isProcessing || _originalImage == null) return;
    final keyContext = _imageKey.currentContext;
    if (keyContext == null) return;

    final RenderBox renderBox = keyContext.findRenderObject() as RenderBox;
    final Size widgetSize = renderBox.size;
    final Offset localPosition = renderBox.globalToLocal(
      details.globalPosition,
    );
    final fittedSizes = applyBoxFit(
      BoxFit.contain,
      Size(_originalImage!.width.toDouble(), _originalImage!.height.toDouble()),
      widgetSize,
    );
    final Size destSize = fittedSizes.destination;
    final double dx = (widgetSize.width - destSize.width) / 2;
    final double dy = (widgetSize.height - destSize.height) / 2;
    final Rect destRect = Rect.fromLTWH(
      dx,
      dy,
      destSize.width,
      destSize.height,
    );
    if (!destRect.contains(localPosition)) return;

    final double originalX =
        (localPosition.dx - destRect.left) *
        (_originalImage!.width / destRect.width);
    final double originalY =
        (localPosition.dy - destRect.top) *
        (_originalImage!.height / destRect.height);
    setState(() {
      _points.add(
        SegmentationPoint(
          point: Offset(originalX, originalY),
          label: _currentPointLabel,
        ),
      );
    });
    _runSegmentation();
  }

  void _clearPoints() {
    setState(() {
      _points.clear();
      _maskImage = null;
      _maskImageData = null; // Clear the mask overlay
    });
  }

  Future<void> _runSegmentation() async {
    if (_encoderSession == null ||
        _decoderSession == null ||
        _imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Models or image not loaded yet!')),
      );
      return;
    }
    if (_points.isEmpty) {
      setState(() {
        _maskImage = null;
        _maskImageData = null;
      });
      return;
    }

    setState(() {
      _isProcessing = true;
    });
    try {
      final imageForPreprocess = img.decodeImage(
        await _imageFile!.readAsBytes(),
      );
      if (imageForPreprocess == null) {
        throw Exception('Failed to decode image for preprocessing.');
      }
      final imageTensor = await _preprocessImage(imageForPreprocess);

      final encoderInputs = {'image': imageTensor};
      final encoderOutputs = await _encoderSession!.run(encoderInputs);
      final imageEmbed = encoderOutputs['image_embed'];
      final highResFeats0 = encoderOutputs['high_res_feats_0'];
      final highResFeats1 = encoderOutputs['high_res_feats_1'];
      if (imageEmbed == null ||
          highResFeats0 == null ||
          highResFeats1 == null) {
        throw Exception('Failed to get valid outputs from the encoder model.');
      }

      final pointData = await _preprocessPoints();
      final pointCoords = pointData['onnx_coord'];
      final pointLabels = pointData['onnx_label'];
      if (pointCoords == null || pointLabels == null) {
        throw Exception('Failed to get valid point data from preprocessing.');
      }

      final maskInput = await OrtValue.fromList(
        Float32List(1 * 1 * 256 * 256),
        [1, 1, 256, 256],
      );
      final hasMaskInput = await OrtValue.fromList(
        Float32List.fromList([0.0]),
        [1],
      );
      final Map<String, OrtValue> decoderInputs = {
        'image_embed': imageEmbed,
        'high_res_feats_0': highResFeats0,
        'high_res_feats_1': highResFeats1,
        'point_coords': pointCoords,
        'point_labels': pointLabels,
        'mask_input': maskInput,
        'has_mask_input': hasMaskInput,
      };
      final decoderOutputs = await _decoderSession!.run(decoderInputs);
      final lowResMasks = decoderOutputs['low_res_masks'];
      final iouPredictions = decoderOutputs['iou_predictions'];
      if (lowResMasks == null || iouPredictions == null) {
        throw Exception('Failed to get valid outputs from the decoder model.');
      }

      final lowResMasksList = (await lowResMasks.asFlattenedList())
          .cast<double>();
      final iouPredictionsList = (await iouPredictions.asFlattenedList())
          .cast<double>();
      final originalImageBytes = await _imageFile!.readAsBytes();

      final params = IsolateParams(
        lowResMasksList: lowResMasksList,
        iouPredictionsList: iouPredictionsList,
        originalWidth: imageForPreprocess.width,
        originalHeight: imageForPreprocess.height,
        originalImageBytes: originalImageBytes,
        fillHoles: _fillHoles,
        removeIslands: _removeIslands,
      );
      final IsolateResult result = await compute(
        _processAndCompositeInIsolate,
        params,
      );
      if (mounted) {
        setState(() {
          _maskImage = img.decodePng(result.maskImageBytes);
          _maskImageData = result.maskImageBytes;
        });
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error during segmentation: $e')));
      debugPrint('Segmentation Error: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  Future<OrtValue> _preprocessImage(img.Image image) async {
    final resizedImage = img.copyResize(
      image,
      width: _modelInputSize,
      height: _modelInputSize,
    );
    final inputData = Float32List(1 * 3 * _modelInputSize * _modelInputSize);
    final mean = [123.675, 116.28, 103.53];
    final std = [58.395, 57.12, 57.375];
    int pixelIndex = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < _modelInputSize; y++) {
        for (int x = 0; x < _modelInputSize; x++) {
          final pixel = resizedImage.getPixel(x, y);
          double value = (c == 0)
              ? pixel.r.toDouble()
              : (c == 1)
              ? pixel.g.toDouble()
              : pixel.b.toDouble();
          inputData[pixelIndex++] = (value - mean[c]) / std[c];
        }
      }
    }
    return OrtValue.fromList(inputData, [
      1,
      3,
      _modelInputSize,
      _modelInputSize,
    ]);
  }

  Future<Map<String, OrtValue>> _preprocessPoints() async {
    final List<double> pointCoordsList = [];
    final List<double> pointLabelsList = [];
    if (_originalImage == null) {
      throw Exception("Original image is null, cannot process points.");
    }
    final int originalWidth = _originalImage!.width;
    final int originalHeight = _originalImage!.height;
    for (final p in _points) {
      final double modelTapX = p.point.dx * (_modelInputSize / originalWidth);
      final double modelTapY = p.point.dy * (_modelInputSize / originalHeight);
      pointCoordsList.addAll([modelTapX, modelTapY]);
      pointLabelsList.add(p.label.toDouble());
    }

    pointCoordsList.addAll([0.0, 0.0]);
    // pointLabelsList.add(-1.0);
    pointLabelsList.add(0.0);
    final int numPoints = _points.length + 1;
    final coordValue = await OrtValue.fromList(
      Float32List.fromList(pointCoordsList),
      [1, numPoints, 2],
    );
    final labelValue = await OrtValue.fromList(
      Float32List.fromList(pointLabelsList),
      [1, numPoints],
    );
    return {'onnx_coord': coordValue, 'onnx_label': labelValue};
  }

  Future<void> _saveImage() async {
    if (_originalImage == null || _maskImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No image or mask to save!')),
      );
      return;
    }

    final hasAccess = await Gal.hasAccess();
    if (!hasAccess) {
      final status = await Gal.requestAccess();
      if (!status) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Storage permission is required to save images.'),
          ),
        );
        return;
      }
    }

    final finalImage = img.Image.from(_originalImage!);
    for (int y = 0; y < finalImage.height; y++) {
      for (int x = 0; x < finalImage.width; x++) {
        final maskPixel = _maskImage!.getPixel(x, y);
        if (maskPixel.a == 0) {
          finalImage.setPixelRgba(x, y, 0, 0, 0, 0);
        }
      }
    }

    final pngBytes = img.encodePng(finalImage);
    final tempDir = await getTemporaryDirectory();
    final tempPath =
        '${tempDir.path}/segmented_${DateTime.now().millisecondsSinceEpoch}.png';
    final tempFile = await File(tempPath).writeAsBytes(pngBytes);

    try {
      await Gal.putImage(tempFile.path);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Image saved to gallery! ✅')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Failed to save image: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final bool modelsLoaded =
        _encoderSession != null && _decoderSession != null;
    return Scaffold(
      appBar: AppBar(
        title: const Text('EdgeTAM Segmentation'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      floatingActionButton: _imageFile != null && !_isProcessing
          ? Column(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                FloatingActionButton.small(
                  onPressed: () => setState(() => _currentPointLabel = 1),
                  backgroundColor: _currentPointLabel == 1
                      ? Colors.green
                      : Colors.grey[300],
                  tooltip: 'Add Positive Point',
                  child: const Icon(Icons.add, color: Colors.white),
                ),
                const SizedBox(height: 8),
                FloatingActionButton.small(
                  onPressed: () => setState(() => _currentPointLabel = 0),
                  backgroundColor: _currentPointLabel == 0
                      ? Colors.red
                      : Colors.grey[300],
                  tooltip: 'Add Negative Point',
                  child: const Icon(Icons.remove, color: Colors.white),
                ),
                const SizedBox(height: 8),
                FloatingActionButton.small(
                  onPressed: _clearPoints,
                  backgroundColor: Colors.blueGrey,
                  tooltip: 'Clear All Points',
                  child: const Icon(Icons.delete_sweep, color: Colors.white),
                ),
              ],
            )
          : null,
      body: Center(
        child: Column(
          children: <Widget>[
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: _isProcessing
                    ? const Center(child: CircularProgressIndicator())
                    : _displayImageData == null
                    ? const Center(child: Text('Pick an image to start'))
                    : GestureDetector(
                        onTapUp: _handleTapUp,
                        child: InteractiveViewer(
                          transformationController: _transformationController,
                          minScale: 0.5,
                          maxScale: 4.0,
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              // 1. The base image
                              if (_displayImageData != null)
                                Image.memory(
                                  _displayImageData!,
                                  key: _imageKey,
                                  fit: BoxFit.contain,
                                  gaplessPlayback: true,
                                ),

                              // 2. The mask overlay
                              if (_maskImageData != null)
                                Image.memory(
                                  _maskImageData!,
                                  fit: BoxFit.contain,
                                  gaplessPlayback: true,
                                ),

                              // 3. The points drawn on top
                              if (_originalImage != null)
                                CustomPaint(
                                  painter: PointPainter(
                                    _points,
                                    Size(
                                      _originalImage!.width.toDouble(),
                                      _originalImage!.height.toDouble(),
                                    ),
                                  ),
                                ),
                            ],
                          ),
                        ),
                      ),
              ),
            ),
            if (_imageFile != null && !_isProcessing && _points.isEmpty)
              const Padding(
                padding: EdgeInsets.fromLTRB(16, 0, 16, 16),
                child: Text(
                  'Tap on an object to segment it!',
                  style: TextStyle(fontSize: 18),
                ),
              ),
            if (_imageFile != null && !_isProcessing && _points.isNotEmpty)
              Padding(
                padding: const EdgeInsets.fromLTRB(16.0, 0, 80.0, 0),
                child: Column(
                  children: [
                    SwitchListTile(
                      title: const Text('Fill Holes'),
                      value: _fillHoles,
                      onChanged: (bool value) {
                        setState(() => _fillHoles = value);
                        _runSegmentation(); // Re-run to apply the change
                      },
                      dense: true,
                    ),
                    SwitchListTile(
                      title: const Text('Remove Islands'),
                      value: _removeIslands,
                      onChanged: (bool value) {
                        setState(() => _removeIslands = value);
                        _runSegmentation(); // Re-run to apply the change
                      },
                      dense: true,
                    ),
                  ],
                ),
              ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton.icon(
                    onPressed: modelsLoaded ? _pickImage : null,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Pick Image'),
                  ),
                  ElevatedButton.icon(
                    onPressed: _maskImage != null ? _saveImage : null,
                    icon: const Icon(Icons.save_alt),
                    label: const Text('Save Image'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.teal,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _encoderSession?.close();
    _decoderSession?.close();
    _transformationController.dispose();
    super.dispose();
  }
}
