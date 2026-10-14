# Basketball Shot Analyzer

A comprehensive biomechanical analysis system for basketball shooting technique evaluation using computer vision and motion analysis.

## Overview

This project provides tools for analyzing basketball shooting mechanics through video processing, pose detection, and biomechanical analysis. It extracts joint movements, calculates kinematics, and provides visual feedback for performance improvement.

## Features

### Core Analysis
- **Pose Detection**: Joint detection using MediaPipe
- **Biomechanical Analysis**: Joint angle, velocity, acceleration, and jerk calculations
- **Center of Gravity (CoG) Analysis**: Movement trajectory and acceleration analysis
- **Kinetic Chain Analysis**: Coordination scoring between hip, knee, and ankle joints

### Visualization
- Peak timing analysis for angles and accelerations
- Phase portraits (velocity vs. angle)
- Kinetic chain timing visualization
- Center of gravity trajectory plots
- Smoothness analysis (jerk plots)

### Post-processing Tools
- Image cropping for screenshot analysis
- Statistical comparison between groups
- Jerk magnitude analysis with multiple mapping functions
- Reporting with CSV export

## Project Layout

```
shot_analyzer/
├── joint_analyzer.py            # kinematics, CoG analysis, plots, CSV export
├── joint_labeller.py            # joint-marked video + screenshots
└── postprocess/
    ├── image_cropper.py         # crop screenshot folders
    ├── jerk_analyzer.py         # CoG jerk scoring and X/Y jerk plots
    └── statistical_analyzer.py  # group A vs. B comparison
requirements.txt
```

Videos and generated results belong in a local `data/` folder, which is git-ignored so that recordings of participants are never committed. A suggested layout:

```
data/
├── videos/     # input clips
├── analysis/   # joint_analyzer output folders
├── labelled/   # joint_labeller output folders
└── group/      # group-level CSV and plots
```

## Installation

Requires Python 3.9–3.12 (the MediaPipe versions below have no wheels for newer Python).

```bash
pip install -r requirements.txt
```

MediaPipe 0.10.30 removed the legacy `mp.solutions` API that both analyzers use, so `requirements.txt` pins `mediapipe<0.10.30`.

Plot titles and labels are in Chinese. Matplotlib looks for SimHei, Microsoft YaHei or Arial Unicode MS; without one of those fonts the labels render as empty boxes.

## Usage

Every script takes its input on the command line; run any of them with `--help` for the full options.

### 1. Biomechanical Analysis
```bash
python -m shot_analyzer.joint_analyzer data/videos/shot.mp4 --out-dir data/analysis --no-show
```
Creates `<video>_plot_graph_<timestamp>/` containing the kinematic CSV and seven plots. Without `--out-dir` the folder is created next to the video; without `--no-show` each plot opens in a window.

```python
from shot_analyzer.joint_analyzer import SimpleJointAnalyzer

analyzer = SimpleJointAnalyzer()
analyzer.process_video("path/to/your/video.mp4")
analyzer._calculate_kinematics_and_average()
analyzer.analyze_center_of_gravity()

analyzer.plot_peak_angle_timing(save_path="angle_timing.png", show_plot=False)
analyzer.plot_kinetic_chain_timing(save_path="kinetic_chain.png", show_plot=False)
analyzer.plot_center_of_gravity_trajectory(save_path="cog_trajectory.png", show_plot=False)
analyzer.save_data_to_csv("analysis_data.csv")
```

### 2. Pose Detection and Joint Marking
```bash
python -m shot_analyzer.joint_labeller data/videos/shot.mp4 --out-dir data/labelled
```
Creates `<video>_analysis_<timestamp>/` with a marked video and a screenshot every 5 frames.

```python
from shot_analyzer.joint_labeller import JointLabeller

labeller = JointLabeller()
labeller.process_video_and_capture(
    video_path="input.mp4",
    output_video_path="marked_video.mp4",
    screenshot_dir="screenshots/",
)
```

### 3. Post-processing
```bash
# Crop screenshots (writes <folder>/cropped_images/)
python -m shot_analyzer.postprocess.image_cropper data/labelled/<run>/screenshots

# Jerk scoring and CoG jerk X/Y plots for a joint_analyzer output folder
python -m shot_analyzer.postprocess.jerk_analyzer data/analysis/<run> --jerk-ref 120 --mapping sigmoid

# Group A vs. B comparison (box/violin plots + t-tests)
python -m shot_analyzer.postprocess.statistical_analyzer data/group/shooting_training_data.csv --no-show
```

```python
from shot_analyzer.postprocess.jerk_analyzer import JerkProcessor

processor = JerkProcessor(jerk_ref=120.0, mapping_type="sigmoid")
processor.process_folder("path/to/csv/files/")
```

## File Descriptions

### Core Modules

**`shot_analyzer/joint_analyzer.py`**
- Main analysis engine for comprehensive biomechanical evaluation
- Calculates joint kinematics (angles, velocities, accelerations, jerk)
- Performs center of gravity analysis
- Generates multiple visualization plots
- Exports data to CSV format

**`shot_analyzer/joint_labeller.py`**
- Joint detection and visualization tool
- Uses MediaPipe for pose estimation
- Creates marked videos with joint annotations
- Generates screenshot sequences for detailed analysis

### Post-processing Tools

**`shot_analyzer/postprocess/jerk_analyzer.py`**
- Analyzes jerk magnitude for movement smoothness evaluation
- Maps jerk RMS to a 0–1 score (exponential, sigmoid, power, linear_threshold, tanh)
- Plots CoG jerk X and Y components
- Generates analysis reports

**`shot_analyzer/postprocess/image_cropper.py`**
- Crops screenshot images to a fixed size, keeping the bottom-left corner
- Maintains consistent image regions for comparative analysis
- Batch processes a whole folder

**`shot_analyzer/postprocess/statistical_analyzer.py`**
- Statistical comparison of CGSS and KCSS improvement between two groups
- Independent-samples t-tests
- Box plots and violin plots of group differences

## Output Files

### Analysis Results
- **CSV Data**: Complete kinematic data with timestamps
- **Plot Images**: PNG files showing different analysis aspects
- **Analysis Reports**: Text files with jerk scoring summaries

### Generated Plots
1. **Peak Angle Timing**: Shows sequence and timing of joint peak angles
2. **Peak Acceleration Timing**: Acceleration peak analysis
3. **Kinetic Chain Timing**: Normalized angular velocity coordination
4. **Phase Portraits**: Velocity vs. angle relationships
5. **Jerk Analysis**: Movement smoothness evaluation
6. **CoG Trajectory**: Center of gravity movement path
7. **CoG Acceleration**: Center of gravity acceleration magnitude

## Analysis Metrics

### Joint Kinematics
- **Angles**: Hip, knee, ankle joint angles over time
- **Velocities**: Angular velocities for movement speed analysis
- **Accelerations**: Angular accelerations for force analysis
- **Jerk**: Rate of acceleration change for smoothness evaluation

### Coordination Metrics
- **KCSS (Kinetic Chain Coordination Score)**: Measures hip-knee-ankle coordination
- **CGSS (Center of Gravity Smoothness Score)**: CoG jerk mapped to a 0–1 score
- **Peak Timing Sequence**: Order and timing of joint peak activations

### Center of Gravity Analysis
- **Trajectory**: 2D path of body's center of mass
- **Velocity Components**: X and Y velocity vectors
- **Acceleration Magnitude**: Overall acceleration intensity
- **Jerk Magnitude**: Smoothness of CoG movement

## Applications

### Training and Coaching
- Technique evaluation and improvement
- Movement pattern analysis
- Progress tracking over time
- Comparison between different shooting styles

### Research
- Biomechanical studies of basketball shooting
- Movement efficiency analysis
- Performance optimization studies

## Technical Notes

### Video Requirements
- Minimum resolution: 640x480
- Recommended: 1080p or higher
- Format: MP4, AVI, MOV
- Frame rate: 30 FPS or higher recommended

### Analysis Parameters
- **Smoothing**: Savitzky-Golay filter (window=15, order=3)
- **Joint Definitions**: Based on MediaPipe pose landmarks
- **Coordinate System**: Normalized to video dimensions
- **Time Base**: Real video timestamps (seconds)

## Troubleshooting

### Common Issues
1. **`module 'mediapipe' has no attribute 'solutions'`**: MediaPipe is 0.10.30 or newer; reinstall with `pip install -r requirements.txt`
2. **Video not loading**: Check file path and format compatibility
3. **Poor pose detection**: Ensure good lighting and clear view of subject
4. **Missing joint data**: Verify subject is fully visible in frame
5. **Memory issues**: Process shorter video segments for large files

### Performance Tips
- Use videos with clear subject visibility
- Avoid shadows and occlusions
- Maintain consistent lighting
- Position camera for side view of shooting motion

## License

This project is developed for biomechanical research and educational purposes.

## Contributing

When contributing to this project:
1. Follow the established naming conventions
2. Document new features thoroughly
3. Include example usage in docstrings
4. Test with sample video data
5. Never commit videos of participants or generated results

## Contact

For questions or support regarding the Basketball Shot Analyzer, please refer to the documentation or submit an issue.
