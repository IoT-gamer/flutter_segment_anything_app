import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'features/segmentation/cubit/segmentation_cubit.dart';
import 'features/segmentation/view/segmentation_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => SegmentationCubit(),
      child: MaterialApp(
        title: 'Segment Anything',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
          useMaterial3: true,
        ),
        home: const SegmentationPage(),
      ),
    );
  }
}
