import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv_dart.dart' as cv;

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
Future<IsolateResult> processAndCompositeInIsolate(IsolateParams params) async {
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
