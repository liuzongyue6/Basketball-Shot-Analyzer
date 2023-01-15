# Basketball Shot Analyzer

A comprehensive biomechanical analysis system for basketball shooting technique evaluation using computer vision and motion analysis.

## Overview

This project provides tools for analyzing basketball shooting mechanics through video processing, pose detection, and biomechanical analysis. It extracts joint movements, calculates kinematics, and provides visual feedback for performance improvement.

## Features

### Core Analysis
- **Pose Detection**: Real-time joint detection using MediaPipe
- **Biomechanical Analysis**: Joint angle, velocity, acceleration, and jerk calculations
- **Center of Gravity (CoG) Analysis**: Movement trajectory and acceleration analysis
- **Kinetic Chain Analysis**: Coordination scoring between hip, knee, and ankle joints

### Visualization
- Joint angle progression over time
- Peak timing analysis for angles and accelerations
- Phase portraits (velocity vs. angle)
- Kinetic chain timing visualization
- Center of gravity trajectory plots
- Smoothness analysis (jerk plots)

### Post-processing Tools
- Image cropping for screenshot analysis
- Statistical comparison between different groups/techniques
- Jerk magnitude analysis with multiple mapping functions
- Comprehensive reporting with CSV export

## Installation

### Prerequisites
- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib
- Scipy
- Seaborn
- Pillow (PIL)

### Setup
```bash
pip install opencv-python mediapipe numpy pandas matplotlib scipy seaborn pillow
```

## Usage

### 1. Basic Biomechanical Analysis
```python
from biomechanical_analyzer import SimpleJointAnalyzer

analyzer = SimpleJointAnalyzer()
video_path = "path/to/your/video.mp4"

# Process video and generate analysis
analyzer.process_video(video_path)
analyzer._calculate_kinematics_and_average()
analyzer.analyze_center_of_gravity()

# Generate plots
analyzer.plot_peak_angle_timing(save_path="angle_timing.png")
analyzer.plot_kinetic_chain_timing(save_path="kinetic_chain.png")
analyzer.plot_center_of_gravity_trajectory(save_path="cog_trajectory.png")

# Save data
analyzer.save_data_to_csv("analysis_data.csv")
```

### 2. Pose Detection and Joint Marking
```python
from pose_detector import SimpleJointAnalyzer

detector = SimpleJointAnalyzer()
detector.process_video_and_capture(
    video_path="input.mp4",
    output_video_path="marked_video.mp4",
    screenshot_dir="screenshots/"
)
```

### 3. Post-processing Analysis
```python
from postprocess.jerk_analyzer import JerkProcessor

processor = JerkProcessor(jerk_ref=120.0, mapping_type="sigmoid")
processor.process_folder("path/to/csv/files/")
```

## File Descriptions

### Core Modules

**`biomechanical_analyzer.py`**
- Main analysis engine for comprehensive biomechanical evaluation
- Calculates joint kinematics (angles, velocities, accelerations, jerk)
- Performs center of gravity analysis
- Generates multiple visualization plots
- Exports data to CSV format

**`pose_detector.py`**
- Simplified joint detection and visualization tool
- Uses MediaPipe for real-time pose estimation
- Creates marked videos with joint annotations
- Generates screenshot sequences for detailed analysis

### Post-processing Tools

**`postprocess/jerk_analyzer.py`**
- Analyzes jerk magnitude for movement smoothness evaluation
- Calculates Kinetic Chain Coordination Score (KCSS)
- Supports multiple mapping functions (exponential, sigmoid, power)
- Generates comprehensive analysis reports

**`postprocess/image_cropper.py`**
- Utility for cropping screenshot images to specific dimensions
- Maintains consistent image regions for comparative analysis
- Batch processing capabilities

**`postprocess/statistical_analyzer.py`**
- Statistical comparison tools for group analysis
- T-test implementations for significance testing
- Visualization of group differences using box plots and violin plots

## Output Files

### Analysis Results
- **CSV Data**: Complete kinematic data with timestamps
- **Plot Images**: Multiple PNG files showing different analysis aspects
- **Analysis Reports**: Text files with statistical summaries

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
- **Peak Timing Sequence**: Order and timing of joint peak activations
- **Delay Analysis**: Time delays between joint activations

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
- Injury prevention research
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
1. **Video not loading**: Check file path and format compatibility
2. **Poor pose detection**: Ensure good lighting and clear view of subject
3. **Missing joint data**: Verify subject is fully visible in frame
4. **Memory issues**: Process shorter video segments for large files

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

## Contact

For questions or support regarding the Basketball Shot Analyzer, please refer to the documentation or submit an issue.
