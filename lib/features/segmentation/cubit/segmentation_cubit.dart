import 'dart:io';
import 'package:bloc/bloc.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:gal/gal.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import '../models/segmentation_models.dart';
import '../services/image_processing.dart';

part 'segmentation_state.dart';

class SegmentationCubit extends Cubit<SegmentationState> {
  static const _encoderAssetPath = 'assets/models/edgetam_encoder.onnx';
  static const _decoderAssetPath = 'assets/models/edgetam_decoder.onnx';
  static const _modelInputSize = 1024;

  OrtSession? _encoderSession;
  OrtSession? _decoderSession;

  SegmentationCubit() : super(const SegmentationState()) {
    _initOrtSessions();
  }

  Future<void> _initOrtSessions() async {
    emit(state.copyWith(status: SegmentationStatus.loadingModels));
    try {
      _encoderSession = await OnnxRuntime().createSessionFromAsset(
        _encoderAssetPath,
      );
      _decoderSession = await OnnxRuntime().createSessionFromAsset(
        _decoderAssetPath,
      );
      emit(state.copyWith(status: SegmentationStatus.modelsReady));
    } catch (e) {
      emit(
        state.copyWith(
          status: SegmentationStatus.failure,
          errorMessage: 'Failed to load models: $e',
        ),
      );
    }
  }

  Future<void> pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      final imageFile = File(pickedFile.path);
      final imageBytes = await imageFile.readAsBytes();
      final originalImage = img.decodeImage(imageBytes);

      emit(
        state.copyWith(
          status: SegmentationStatus.success,
          imageFile: imageFile,
          originalImage: originalImage,
          displayImageData: imageBytes,
          maskImageData: null,
          points: [],
          clearMask: true, // Explicitly clear the mask
        ),
      );
    }
  }

  void addPoint(Offset originalPoint) {
    final newPoint = SegmentationPoint(
      point: originalPoint,
      label: state.currentPointLabel,
    );
    final updatedPoints = List<SegmentationPoint>.from(state.points)
      ..add(newPoint);
    emit(state.copyWith(points: updatedPoints));
    _runSegmentation();
  }

  void clearPoints() {
    emit(
      state.copyWith(
        points: [],
        maskImageData: null,
        clearMask: true,
        status: SegmentationStatus.success,
      ),
    );
  }

  void setPointLabel(int label) {
    emit(state.copyWith(currentPointLabel: label));
  }

  void setClassName(String name) {
    emit(state.copyWith(className: name));
  }

  void toggleFillHoles(bool value) {
    emit(state.copyWith(fillHoles: value));
    _runSegmentation();
  }

  void toggleRemoveIslands(bool value) {
    emit(state.copyWith(removeIslands: value));
    _runSegmentation();
  }

  void toggleSelectLargestArea(bool value) {
    emit(state.copyWith(selectLargestArea: value));
    _runSegmentation();
  }

  Future<void> _runSegmentation() async {
    if (_encoderSession == null ||
        _decoderSession == null ||
        state.imageFile == null) {
      return; // Or emit failure state
    }
    if (state.points.isEmpty) {
      emit(state.copyWith(clearMask: true, status: SegmentationStatus.success));
      return;
    }

    emit(state.copyWith(status: SegmentationStatus.processing));

    try {
      final imageForPreprocess = img.decodeImage(
        await state.imageFile!.readAsBytes(),
      )!;

      // Encoder run
      final imageTensor = await _preprocessImage(imageForPreprocess);
      final encoderInputs = {'image': imageTensor};
      final encoderOutputs = await _encoderSession!.run(encoderInputs);
      final imageEmbed = encoderOutputs['image_embed']!;
      final highResFeats0 = encoderOutputs['high_res_feats_0']!;
      final highResFeats1 = encoderOutputs['high_res_feats_1']!;

      // Decoder run
      final pointData = await _preprocessPoints();
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
        'point_coords': pointData['onnx_coord']!,
        'point_labels': pointData['onnx_label']!,
        'mask_input': maskInput,
        'has_mask_input': hasMaskInput,
      };

      final decoderOutputs = await _decoderSession!.run(decoderInputs);
      final lowResMasks = decoderOutputs['low_res_masks']!;
      final iouPredictions = decoderOutputs['iou_predictions']!;

      final params = IsolateParams(
        lowResMasksList: (await lowResMasks.asFlattenedList()).cast<double>(),
        iouPredictionsList: (await iouPredictions.asFlattenedList())
            .cast<double>(),
        originalWidth: imageForPreprocess.width,
        originalHeight: imageForPreprocess.height,
        originalImageBytes: await state.imageFile!.readAsBytes(),
        fillHoles: state.fillHoles,
        removeIslands: state.removeIslands,
        selectLargestArea: state.selectLargestArea,
      );

      final IsolateResult result = await compute(
        processAndCompositeInIsolate,
        params,
      );

      emit(
        state.copyWith(
          status: SegmentationStatus.success,
          maskImageData: result.maskImageBytes,
        ),
      );
    } catch (e) {
      debugPrint('Segmentation Error: $e');
      emit(
        state.copyWith(
          status: SegmentationStatus.failure,
          errorMessage: 'Error during segmentation: $e',
        ),
      );
    }
  }

  Future<void> saveImage() async {
    if (state.originalImage == null || state.maskImageData == null) {
      return;
    }
    final maskImage = img.decodePng(state.maskImageData!);
    if (maskImage == null) return;

    emit(state.copyWith(status: SegmentationStatus.saving));

    try {
      final hasAccess = await Gal.hasAccess();
      if (!hasAccess) {
        final status = await Gal.requestAccess();
        if (!status) {
          emit(
            state.copyWith(
              status: SegmentationStatus.failure,
              errorMessage: 'Storage permission is required to save images.',
            ),
          );
          return;
        }
      }

      final finalImage = img.Image(
        width: state.originalImage!.width,
        height: state.originalImage!.height,
        numChannels: 4,
      );

      for (int y = 0; y < finalImage.height; y++) {
        for (int x = 0; x < finalImage.width; x++) {
          final originalPixel = state.originalImage!.getPixel(x, y);
          final maskPixel = maskImage.getPixel(x, y);
          finalImage.setPixelRgba(
            x,
            y,
            originalPixel.r.toInt(),
            originalPixel.g.toInt(),
            originalPixel.b.toInt(),
            maskPixel.a.toInt(),
          );
        }
      }

      if (state.className.isNotEmpty) {
        finalImage.addTextData({'className': state.className});
      }

      final pngBytes = img.encodePng(finalImage);
      final tempDir = await getTemporaryDirectory();
      final tempPath =
          '${tempDir.path}/segmented_${DateTime.now().millisecondsSinceEpoch}.png';
      final tempFile = await File(tempPath).writeAsBytes(pngBytes);
      await Gal.putImage(tempFile.path);

      // Use a success message in the state if you want to show a snackbar
      emit(state.copyWith(status: SegmentationStatus.success));
    } catch (e) {
      emit(
        state.copyWith(
          status: SegmentationStatus.failure,
          errorMessage: 'Failed to save image: $e',
        ),
      );
    }
  }

  // Preprocessing functions (can be kept private within the cubit)
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
    final int originalWidth = state.originalImage!.width;
    final int originalHeight = state.originalImage!.height;
    for (final p in state.points) {
      final double modelTapX = p.point.dx * (_modelInputSize / originalWidth);
      final double modelTapY = p.point.dy * (_modelInputSize / originalHeight);
      pointCoordsList.addAll([modelTapX, modelTapY]);
      pointLabelsList.add(p.label.toDouble());
    }

    pointCoordsList.addAll([0.0, 0.0]);
    pointLabelsList.add(0.0);
    final int numPoints = state.points.length + 1;
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

  @override
  Future<void> close() {
    _encoderSession?.close();
    _decoderSession?.close();
    return super.close();
  }
}
