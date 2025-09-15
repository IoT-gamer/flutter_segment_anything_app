import 'package:flutter/material.dart';

/// A class to store a segmentation point's coordinates and label.
class SegmentationPoint {
  SegmentationPoint({required this.point, required this.label});
  final Offset point;
  final int label;
}
