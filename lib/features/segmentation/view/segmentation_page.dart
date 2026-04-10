import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../cubit/segmentation_cubit.dart';
import 'widgets/point_painter.dart';

class SegmentationPage extends StatefulWidget {
  const SegmentationPage({super.key});

  @override
  State<SegmentationPage> createState() => _SegmentationPageState();
}

class _SegmentationPageState extends State<SegmentationPage> {
  final GlobalKey _imageKey = GlobalKey();
  final TransformationController _transformationController =
      TransformationController();

  Offset? _dragStart;
  Rect? _currentDragBox;

  Offset? _getOriginalCoordinates(
    Offset globalPosition,
    SegmentationState state,
  ) {
    final keyContext = _imageKey.currentContext;
    if (keyContext == null || state.originalImage == null) return null;

    final RenderBox renderBox = keyContext.findRenderObject() as RenderBox;
    final Size widgetSize = renderBox.size;
    final Offset localPosition = renderBox.globalToLocal(globalPosition);

    final fittedSizes = applyBoxFit(
      BoxFit.contain,
      Size(
        state.originalImage!.width.toDouble(),
        state.originalImage!.height.toDouble(),
      ),
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

    if (!destRect.contains(localPosition)) return null;

    final double originalX =
        (localPosition.dx - destRect.left) *
        (state.originalImage!.width / destRect.width);
    final double originalY =
        (localPosition.dy - destRect.top) *
        (state.originalImage!.height / destRect.height);

    return Offset(originalX, originalY);
  }

  void _handleTapUp(TapUpDetails details, SegmentationState state) {
    if (state.imageFile == null ||
        state.status == SegmentationStatus.processing ||
        state.originalImage == null) {
      return;
    }
    final originalPoint = _getOriginalCoordinates(
      details.globalPosition,
      state,
    );
    if (originalPoint != null) {
      context.read<SegmentationCubit>().addPoint(originalPoint);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cubit = context.read<SegmentationCubit>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Segment Anything'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      floatingActionButton: BlocBuilder<SegmentationCubit, SegmentationState>(
        builder: (context, state) {
          if (state.imageFile == null ||
              state.status == SegmentationStatus.processing) {
            return const SizedBox.shrink();
          }
          return Column(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              FloatingActionButton.small(
                onPressed: cubit.toggleBoxMode,
                backgroundColor: state.isBoxMode
                    ? Colors.orange
                    : Colors.grey[300],
                tooltip: 'Toggle Box Mode',
                child: Icon(
                  state.isBoxMode ? Icons.crop_free : Icons.touch_app,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 8),
              FloatingActionButton.small(
                onPressed: () => cubit.setPointLabel(1),
                backgroundColor: state.currentPointLabel == 1
                    ? Colors.green
                    : Colors.grey[300],
                tooltip: 'Add Positive Point',
                child: const Icon(Icons.add, color: Colors.white),
              ),
              const SizedBox(height: 8),
              FloatingActionButton.small(
                onPressed: () => cubit.setPointLabel(0),
                backgroundColor: state.currentPointLabel == 0
                    ? Colors.red
                    : Colors.grey[300],
                tooltip: 'Add Negative Point',
                child: const Icon(Icons.remove, color: Colors.white),
              ),
              const SizedBox(height: 8),
              FloatingActionButton.small(
                onPressed: cubit.clearPoints,
                backgroundColor: Colors.blueGrey,
                tooltip: 'Clear All Points',
                child: const Icon(Icons.delete_sweep, color: Colors.white),
              ),
            ],
          );
        },
      ),
      body: BlocListener<SegmentationCubit, SegmentationState>(
        listener: (context, state) {
          if (state.status == SegmentationStatus.failure) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  state.errorMessage ?? 'An unknown error occurred',
                ),
              ),
            );
          }
        },
        child: Column(
          children: <Widget>[
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: BlocBuilder<SegmentationCubit, SegmentationState>(
                  builder: (context, state) {
                    if (state.status == SegmentationStatus.processing) {
                      return const Center(child: CircularProgressIndicator());
                    }
                    if (state.displayImageData == null) {
                      return const Center(
                        child: Text('Pick an image to start'),
                      );
                    }
                    return GestureDetector(
                      // Handle Taps (Point Mode)
                      onTapUp: state.isBoxMode
                          ? null
                          : (details) => _handleTapUp(details, state),

                      // Handle Drags (Box Mode)
                      onPanStart: !state.isBoxMode
                          ? null
                          : (details) {
                              _dragStart = _getOriginalCoordinates(
                                details.globalPosition,
                                state,
                              );
                              setState(() {
                                _currentDragBox =
                                    null; // Clear previous local drawing
                              });
                            },
                      onPanUpdate: !state.isBoxMode
                          ? null
                          : (details) {
                              if (_dragStart == null) return;
                              final currentOriginal = _getOriginalCoordinates(
                                details.globalPosition,
                                state,
                              );
                              if (currentOriginal != null) {
                                // Update the UI locally for real-time 60fps drawing
                                setState(() {
                                  _currentDragBox = Rect.fromPoints(
                                    _dragStart!,
                                    currentOriginal,
                                  );
                                });
                              }
                            },
                      onPanEnd: !state.isBoxMode
                          ? null
                          : (details) {
                              if (_currentDragBox != null) {
                                // Commit the final box to the Cubit and run inference
                                cubit.updateBoundingBox(_currentDragBox);
                                cubit.submitBoundingBox();
                              }

                              setState(() {
                                _dragStart = null;
                                _currentDragBox = null;
                              });
                            },
                      child: InteractiveViewer(
                        transformationController: _transformationController,
                        minScale: 0.5,
                        maxScale: 4.0,
                        // Disable panning/zooming while drawing the box
                        panEnabled: !state.isBoxMode,
                        scaleEnabled: !state.isBoxMode,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            Image.memory(
                              state.displayImageData!,
                              key: _imageKey,
                              fit: BoxFit.contain,
                              gaplessPlayback: true,
                            ),
                            if (state.maskImageData != null)
                              Image.memory(
                                state.maskImageData!,
                                fit: BoxFit.contain,
                                gaplessPlayback: true,
                              ),
                            if (state.originalImage != null)
                              CustomPaint(
                                painter: PointPainter(
                                  state.points,
                                  Size(
                                    state.originalImage!.width.toDouble(),
                                    state.originalImage!.height.toDouble(),
                                  ),
                                  // Pass the local drawing box, or the saved state box
                                  boundingBox:
                                      _currentDragBox ?? state.boundingBox,
                                ),
                              ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),

            BlocBuilder<SegmentationCubit, SegmentationState>(
              buildWhen: (p, c) =>
                  p.imageFile != c.imageFile ||
                  p.points.length != c.points.length ||
                  p.boundingBox != c.boundingBox ||
                  p.status != c.status,
              builder: (context, state) {
                if (state.imageFile == null ||
                    state.status == SegmentationStatus.processing) {
                  return const SizedBox.shrink();
                }

                if (state.points.isEmpty && state.boundingBox == null) {
                  return const Padding(
                    padding: EdgeInsets.fromLTRB(16, 0, 16, 16),
                    child: Text(
                      'Tap or draw a box to segment an object!',
                      style: TextStyle(fontSize: 18),
                    ),
                  );
                }

                return Padding(
                  padding: const EdgeInsets.fromLTRB(16.0, 0, 80.0, 0),
                  child: Column(
                    children: [
                      SwitchListTile(
                        title: const Text('Fill Holes'),
                        value: state.fillHoles,
                        onChanged: (value) => cubit.toggleFillHoles(value),
                        dense: true,
                      ),
                      SwitchListTile(
                        title: const Text('Remove Islands'),
                        value: state.removeIslands,
                        onChanged: (value) => cubit.toggleRemoveIslands(value),
                        dense: true,
                      ),
                      SwitchListTile(
                        title: const Text('Select Largest Area'),
                        value: state.selectLargestArea,
                        onChanged: (value) =>
                            cubit.toggleSelectLargestArea(value),
                        dense: true,
                      ),
                    ],
                  ),
                );
              },
            ),

            BlocBuilder<SegmentationCubit, SegmentationState>(
              buildWhen: (p, c) => p.imageFile != c.imageFile,
              builder: (context, state) {
                if (state.imageFile == null) {
                  return const SizedBox.shrink();
                }
                return Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24.0,
                    vertical: 8.0,
                  ),
                  child: TextField(
                    decoration: const InputDecoration(
                      labelText: 'Class Name (Optional)',
                      hintText: 'e.g., "cat", "person", "car"',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    onChanged: (value) => cubit.setClassName(value),
                  ),
                );
              },
            ),

            Padding(
              padding: const EdgeInsets.all(16.0),
              child: BlocBuilder<SegmentationCubit, SegmentationState>(
                builder: (context, state) {
                  final modelsLoaded =
                      state.status != SegmentationStatus.initial &&
                      state.status != SegmentationStatus.loadingModels;
                  final isSaving = state.status == SegmentationStatus.saving;
                  return Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton.icon(
                        onPressed: modelsLoaded ? cubit.pickImage : null,
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Pick Image'),
                      ),
                      ElevatedButton.icon(
                        onPressed: (state.maskImageData != null && !isSaving)
                            ? cubit.saveImage
                            : null,
                        icon: isSaving
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              )
                            : const Icon(Icons.save_alt),
                        label: Text(isSaving ? 'Saving...' : 'Save Image'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.teal,
                          foregroundColor: Colors.white,
                        ),
                      ),
                    ],
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _transformationController.dispose();
    super.dispose();
  }
}
