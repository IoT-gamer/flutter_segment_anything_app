part of 'segmentation_cubit.dart';

enum SegmentationStatus {
  initial,
  loadingModels,
  modelsReady,
  processing,
  saving,
  success,
  failure,
}

class SegmentationState extends Equatable {
  const SegmentationState({
    this.status = SegmentationStatus.initial,
    this.imageFile,
    this.originalImage,
    this.displayImageData,
    this.maskImageData,
    this.points = const [],
    this.currentPointLabel = 1,
    this.fillHoles = false,
    this.removeIslands = false,
    this.selectLargestArea = false,
    this.errorMessage,
  });

  final SegmentationStatus status;
  final File? imageFile;
  final img.Image? originalImage;
  final Uint8List? displayImageData;
  final Uint8List? maskImageData;
  final List<SegmentationPoint> points;
  final int currentPointLabel;
  final bool fillHoles;
  final bool removeIslands;
  final bool selectLargestArea;
  final String? errorMessage;

  SegmentationState copyWith({
    SegmentationStatus? status,
    File? imageFile,
    img.Image? originalImage,
    Uint8List? displayImageData,
    Uint8List? maskImageData,
    List<SegmentationPoint>? points,
    int? currentPointLabel,
    bool? fillHoles,
    bool? removeIslands,
    bool? selectLargestArea,
    String? errorMessage,
    bool clearMask = false,
  }) {
    return SegmentationState(
      status: status ?? this.status,
      imageFile: imageFile ?? this.imageFile,
      originalImage: originalImage ?? this.originalImage,
      displayImageData: displayImageData ?? this.displayImageData,
      maskImageData: clearMask ? null : maskImageData ?? this.maskImageData,
      points: points ?? this.points,
      currentPointLabel: currentPointLabel ?? this.currentPointLabel,
      fillHoles: fillHoles ?? this.fillHoles,
      removeIslands: removeIslands ?? this.removeIslands,
      selectLargestArea: selectLargestArea ?? this.selectLargestArea,
      errorMessage: errorMessage ?? this.errorMessage,
    );
  }

  @override
  List<Object?> get props => [
    status,
    imageFile,
    originalImage,
    displayImageData,
    maskImageData,
    points,
    currentPointLabel,
    fillHoles,
    removeIslands,
    selectLargestArea,
    errorMessage,
  ];
}
