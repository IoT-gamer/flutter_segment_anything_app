import 'package:flutter/material.dart';
import '../../models/segmentation_models.dart';

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
