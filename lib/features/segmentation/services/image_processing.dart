import 'dart:typed_data';
import 'package:image/image.dart' as img;

class IsolateParams {
  final List<double> lowResMasksList;
  final List<double> iouPredictionsList;
  final int originalWidth;
  final int originalHeight;
  final Uint8List originalImageBytes;
  final bool fillHoles;
  final bool removeIslands;
  final bool selectLargestArea;

  IsolateParams({
    required this.lowResMasksList,
    required this.iouPredictionsList,
    required this.originalWidth,
    required this.originalHeight,
    required this.originalImageBytes,
    required this.fillHoles,
    required this.removeIslands,
    required this.selectLargestArea,
  });
}

class IsolateResult {
  final Uint8List maskImageBytes;
  IsolateResult(this.maskImageBytes);
}

Future<IsolateResult> processAndCompositeInIsolate(IsolateParams params) async {
  // =========================================================
  //                  🔧 TUNING PARAMETERS
  // =========================================================

  // Morphology kernel radius (1 = 3×3, 2 = 5×5, 3 = 7×7)
  const int morphKernel = 3;

  // Largest component: connectivity (4 or 8)
  const int connectivity = 4;

  // Threshold for mask activation
  const double activationThreshold = 0.5;

  // Optional: enable edge smoothing after all processing
  const bool enableSmoothing = true;

  // Optional: smoothing radius (1 = light blur)
  const int smoothingRadius = 1;

  // =========================================================
  //                    END TUNING SECTION
  // =========================================================

  final masksData = params.lowResMasksList;
  final iouData = params.iouPredictionsList;

  int bestMaskIdx = 0;
  if (iouData.isNotEmpty) {
    for (int i = 1; i < iouData.length; i++) {
      if (iouData[i] > iouData[bestMaskIdx]) {
        bestMaskIdx = i;
      }
    }
  }

  const int W = 256;
  const int H = 256;
  const int maskSize = W * H;

  final int maskOffset = bestMaskIdx * maskSize;
  final maskSlice = masksData.sublist(maskOffset, maskOffset + maskSize);

  // Convert to 0/255 binary mask
  final rawMask = Uint8List(maskSize);
  for (int i = 0; i < maskSize; i++) {
    rawMask[i] = maskSlice[i] > activationThreshold ? 255 : 0;
  }

  // ---------------- Morphology ----------------

  bool anyNeighbor(Uint8List src, int x, int y, int radius) {
    for (int ky = -radius; ky <= radius; ky++) {
      for (int kx = -radius; kx <= radius; kx++) {
        int nx = x + kx, ny = y + ky;
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
        if (src[ny * W + nx] > 0) return true;
      }
    }
    return false;
  }

  bool allNeighbors(Uint8List src, int x, int y, int radius) {
    for (int ky = -radius; ky <= radius; ky++) {
      for (int kx = -radius; kx <= radius; kx++) {
        int nx = x + kx, ny = y + ky;
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) return false;
        if (src[ny * W + nx] == 0) return false;
      }
    }
    return true;
  }

  Uint8List dilate(Uint8List src, int r) {
    final out = Uint8List(src.length);
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        out[y * W + x] = anyNeighbor(src, x, y, r) ? 255 : 0;
      }
    }
    return out;
  }

  Uint8List erode(Uint8List src, int r) {
    final out = Uint8List(src.length);
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        out[y * W + x] = allNeighbors(src, x, y, r) ? 255 : 0;
      }
    }
    return out;
  }

  Uint8List morphOpen(Uint8List s, int r) => dilate(erode(s, r), r);
  Uint8List morphClose(Uint8List s, int r) => erode(dilate(s, r), r);

  // ---------------- Largest Component ----------------

  Uint8List keepLargest(Uint8List src) {
    final labels = Uint32List(maskSize);
    int currentLabel = 0;
    final counts = <int, int>{};
    final q = List<int>.filled(maskSize, 0);

    final dx4 = [1, -1, 0, 0];
    final dy4 = [0, 0, 1, -1];
    final dx8 = [1, 1, 1, 0, 0, -1, -1, -1];
    final dy8 = [1, 0, -1, 1, -1, 1, 0, -1];

    final dx = connectivity == 8 ? dx8 : dx4;
    final dy = connectivity == 8 ? dy8 : dy4;

    final neighborCount = dx.length;

    for (int i = 0; i < maskSize; i++) {
      if (src[i] == 0 || labels[i] != 0) continue;

      currentLabel++;
      int head = 0, tail = 0;
      q[tail++] = i;
      labels[i] = currentLabel;
      counts[currentLabel] = 1;

      while (head < tail) {
        final p = q[head++];
        final py = p ~/ W, px = p % W;

        for (int k = 0; k < neighborCount; k++) {
          final nx = px + dx[k], ny = py + dy[k];
          if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

          final np = ny * W + nx;
          if (src[np] == 0 || labels[np] != 0) continue;

          labels[np] = currentLabel;
          counts[currentLabel] = counts[currentLabel]! + 1;
          q[tail++] = np;
        }
      }
    }

    if (counts.isEmpty) return src;

    final maxLabel = counts.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;

    final out = Uint8List(maskSize);
    for (int i = 0; i < maskSize; i++) {
      out[i] = labels[i] == maxLabel ? 255 : 0;
    }
    return out;
  }

  // ---------------- Apply Steps ----------------

  Uint8List processed = rawMask;

  if (params.selectLargestArea) {
    processed = keepLargest(processed);
  }

  if (params.fillHoles) {
    processed = morphClose(processed, morphKernel);
  }

  if (params.removeIslands) {
    processed = morphOpen(processed, morphKernel);
  }

  if (enableSmoothing) {
    processed = dilate(erode(processed, smoothingRadius), smoothingRadius);
  }

  // ---------------- Convert to RGBA and resize ----------------

  final lowResImage = img.Image(width: W, height: H, numChannels: 4);

  for (int i = 0; i < maskSize; i++) {
    final x = i % W;
    final y = i ~/ W;
    if (processed[i] > 0) {
      lowResImage.setPixelRgba(x, y, 0, 255, 0, 255);
    } else {
      lowResImage.setPixelRgba(x, y, 0, 0, 0, 0);
    }
  }

  final finalMask = img.copyResize(
    lowResImage,
    width: params.originalWidth,
    height: params.originalHeight,
    interpolation: img.Interpolation.linear,
  );

  final maskBytes = img.encodePng(finalMask);

  return IsolateResult(Uint8List.fromList(maskBytes));
}
