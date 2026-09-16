import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.signal import savgol_filter

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class SimpleJointAnalyzer:
    """
    Comprehensive biomechanical analyzer for sports movements.
    - Calculates kinematics for left and right limbs and provides averages.
    - Uses Savitzky-Golay filter for data smoothing.
    - Generates insightful plots for timing, smoothness, and motor control analysis.
    - Includes Center of Gravity (CoG) analysis capabilities.
    """
    
    def __init__(self):
        # --- MediaPipe Initialization ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # --- Data Storage ---
        self.all_landmarks = []
        self.joint_angles, self.joint_velocities, self.joint_accelerations, self.joint_jerks = {}, {}, {}, {}
        self.output_data = {}
        self.cog_data = {}
        self.frame_count = 0
        self.fps = 30
        
        # --- Joint Definitions ---
        self.joints_to_calculate = {
            'ankle_right': (26, 28, 32), 'ankle_left': (25, 27, 31),
            'knee_right': (24, 26, 28), 'knee_left': (23, 25, 27),
            'hip_right': (12, 24, 26), 'hip_left': (11, 23, 25),
        }
        self.joint_averaging_pairs = {
            'ankle_avg': ('ankle_left', 'ankle_right'),
            'knee_avg': ('knee_left', 'knee_right'),
            'hip_avg': ('hip_left', 'hip_right'),
        }
        
        # --- Plot Style Configuration ---
        self.plot_config = {
            'font_size': {
                'title': 20,
                'label': 20,
                'tick': 20,
                'legend': 20,
                'text': 20
            },
            'line_width': 2,
            'marker_size': 8,
            'figure_size': (15, 8),
            'dpi': 600,
            'grid': True,  # 启用网格线
            'colors': {
                'ankle_avg': 'green',
                'knee_avg': 'blue', 
                'hip_avg': 'red'
            },
            'chinese_labels': {
                'hip_avg': '髋关节',
                'knee_avg': '膝关节',
                'ankle_avg': '踝关节'
            }
        }
    
    def _apply_plot_style(self, ax, title=None, xlabel=None, ylabel=None):
        """统一应用绘图样式配置"""
        # 设置字体大小
        if title:
            ax.set_title(title, fontsize=self.plot_config['font_size']['title'], fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.plot_config['font_size']['label'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.plot_config['font_size']['label'])
        
        # 设置刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=self.plot_config['font_size']['tick'])
        
        # 设置网格 - 只在x轴显示虚线网格
        if self.plot_config['grid']:
            ax.grid(True, axis='x', linestyle='--', alpha=0.7, linewidth=1)
        
        # 设置图例字体大小
        legend = ax.get_legend()
        if legend:
            legend.set_fontsize(self.plot_config['font_size']['legend'])
    
    def _setup_figure(self, figsize=None):
        """统一创建和配置图形"""
        if figsize is None:
            figsize = self.plot_config['figure_size']
        
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    def calculate_angle(self, p1, p2, p3):
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        v1, v2 = p1 - p2, p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    def extract_landmarks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        if results.pose_landmarks:
            h, w, _ = image.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]
            return landmarks
        return None
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError(f"Cannot open video file: {video_path}")
        
        # 获取视频的真实帧尺寸和帧率
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 打印视频信息用于调试
        print(f"Video properties:")
        print(f"  - FPS: {self.fps}")
        print(f"  - Width: {self.video_width} pixels")
        print(f"  - Height: {self.video_height} pixels")
        
        frame_angles = {joint: [] for joint in self.joints_to_calculate.keys()}
        print("Step 1/3: Processing video to extract raw angles...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            landmarks = self.extract_landmarks(frame)
            self.all_landmarks.append(landmarks)
            
            if landmarks:
                for joint_name, indices in self.joints_to_calculate.items():
                    try:
                        p1, p2, p3 = landmarks[indices[0]], landmarks[indices[1]], landmarks[indices[2]]
                        frame_angles[joint_name].append(self.calculate_angle(p1, p2, p3))
                    except (IndexError, TypeError):
                        frame_angles[joint_name].append(np.nan)
            else:
                for joint_name in self.joints_to_calculate.keys():
                    frame_angles[joint_name].append(np.nan)

            self.frame_count += 1
            if self.frame_count % 100 == 0: print(f"  ...processed {self.frame_count} frames")
        
        cap.release()
        self.joint_angles = frame_angles
        print(f"  -> Video processing finished. Total frames: {self.frame_count}")
    
    def _calculate_kinematics_and_average(self, smooth_window=15, smooth_polyorder=3):
        if not self.joint_angles: raise ValueError("Angle data not found. Run process_video first.")

        print("Step 2/3: Calculating kinematics (velocity, acceleration, jerk) and averaging...")
        time_step = 1 / self.fps
        
        # Calculate raw kinematics for every joint
        for joint_name, angles in self.joint_angles.items():
            valid_indices = ~np.isnan(angles)
            if not np.any(valid_indices):
                nan_array = np.full(len(angles), np.nan)
                self.joint_velocities[joint_name], self.joint_accelerations[joint_name], self.joint_jerks[joint_name] = nan_array, nan_array, nan_array
                continue

            xp, fp = np.where(valid_indices)[0], np.array(angles)[valid_indices]
            x = np.arange(len(angles))
            interp_angles = np.interp(x, xp, fp)

            window = min(smooth_window, len(interp_angles) - 1)
            if window % 2 == 0: window -= 1
            
            smoothed_angles = savgol_filter(interp_angles, window, smooth_polyorder) if window > smooth_polyorder else interp_angles
            
            self.joint_velocities[joint_name] = np.gradient(smoothed_angles, time_step)
            self.joint_accelerations[joint_name] = np.gradient(self.joint_velocities[joint_name], time_step)
            self.joint_jerks[joint_name] = np.gradient(self.joint_accelerations[joint_name], time_step)
        
        # Average the data for the final output
        final_data = {'angle': {}, 'velocity': {}, 'acceleration': {}, 'jerk': {}}
        source_map = {'angle': self.joint_angles, 'velocity': self.joint_velocities, 'acceleration': self.joint_accelerations, 'jerk': self.joint_jerks}

        for avg_name, (left, right) in self.joint_averaging_pairs.items():
            for key, source_dict in source_map.items():
                with np.errstate(all='ignore'): # Suppress mean of empty slice warning
                    final_data[key][avg_name] = np.nanmean(np.vstack([np.array(source_dict[left]), np.array(source_dict[right])]), axis=0)
                
        non_averaged = set(self.joints_to_calculate.keys()) - {j for p in self.joint_averaging_pairs.values() for j in p}
        for joint in non_averaged:
             for key, source_dict in source_map.items():
                final_data[key][joint] = source_dict[joint]

        self.output_data = final_data
        print("  -> Kinematics calculated and averaged successfully.")

    def analyze_center_of_gravity(self):
        print("Step 3/3: Analyzing Center of Gravity (optional)...")
        cog_x, cog_y = [], []
        mass_ratios = {'head': 0.081, 'torso': 0.497, 'r_upper_arm': 0.028, 'l_upper_arm': 0.028, 'r_lower_arm': 0.022, 'l_lower_arm': 0.022, 'r_thigh': 0.099, 'l_thigh': 0.099, 'r_shin': 0.046, 'l_shin': 0.046}
        
        # 使用真实的视频帧尺寸
        if hasattr(self, 'video_width') and hasattr(self, 'video_height'):
            frame_width = self.video_width
            frame_height = self.video_height
            print(f"  -> Using actual video dimensions: {frame_width}x{frame_height}")
        else:
            # 备用方案：从关键点估算（如果无法获取真实尺寸）
            frame_width = None
            frame_height = None
            print("  -> Warning: Using landmark estimation for frame dimensions")
        
        for landmarks in self.all_landmarks:
            if not landmarks:
                cog_x.append(np.nan); cog_y.append(np.nan)
                continue
            try:
                # 如果没有真实尺寸，才从关键点估算（备用方案）
                if frame_width is None and landmarks:
                    x_coords = [lm[0] for lm in landmarks if lm[0] > 0]
                    y_coords = [lm[1] for lm in landmarks if lm[1] > 0]
                    if x_coords and y_coords:
                        # 移除不合理的1.2倍边距，只添加5%的小边距
                        frame_width = max(x_coords) * 1.05
                        frame_height = max(y_coords) * 1.05
                        print(f"  -> Estimated dimensions from landmarks: {frame_width:.0f}x{frame_height:.0f}")
                
                segments = {'head': (landmarks[0], landmarks[0]), 'torso': (midpoint(landmarks[11], landmarks[12]), midpoint(landmarks[23], landmarks[24])), 'r_upper_arm': (landmarks[12], landmarks[14]), 'l_upper_arm': (landmarks[11], landmarks[13]), 'r_lower_arm': (landmarks[14], landmarks[16]), 'l_lower_arm': (landmarks[13], landmarks[15]), 'r_thigh': (landmarks[24], landmarks[26]), 'l_thigh': (landmarks[23], landmarks[25]), 'r_shin': (landmarks[26], landmarks[28]), 'l_shin': (landmarks[25], landmarks[27])}
                num_x, num_y, den = 0, 0, 0
                for name, points in segments.items():
                    center = midpoint(points[0], points[1]); mass = mass_ratios.get(name, 0)
                    num_x += center[0] * mass; num_y += center[1] * mass; den += mass
                cog_x.append(num_x / den if den > 0 else np.nan); cog_y.append(num_y / den if den > 0 else np.nan)
            except (IndexError, TypeError):
                cog_x.append(np.nan); cog_y.append(np.nan)
        
        # Store raw CoG data
        self.cog_data['x_raw'] = cog_x
        self.cog_data['y_raw'] = cog_y
        
        # Normalize positions using actual frame dimensions
        if frame_width and frame_height:
            cog_x_normalized = [x / frame_width if not np.isnan(x) else np.nan for x in cog_x]
            cog_y_normalized = [y / frame_height if not np.isnan(y) else np.nan for y in cog_y]
        else:
            print("Warning: Could not determine frame dimensions for normalization")
            cog_x_normalized = cog_x
            cog_y_normalized = cog_y
        
        # Store normalized positions
        self.cog_data['x'] = cog_x_normalized
        self.cog_data['y'] = cog_y_normalized
        
        # Calculate velocities
        time_step = 1 / self.fps
        self.cog_data['vx'] = np.gradient(np.nan_to_num(cog_x_normalized), time_step)
        self.cog_data['vy'] = np.gradient(np.nan_to_num(cog_y_normalized), time_step)
        
        # Calculate accelerations
        self.cog_data['ax'] = np.gradient(self.cog_data['vx'], time_step)
        self.cog_data['ay'] = np.gradient(self.cog_data['vy'], time_step)
        
        # Calculate magnitude values
        self.cog_data['speed'] = np.sqrt(np.array(self.cog_data['vx'])**2 + np.array(self.cog_data['vy'])**2)
        self.cog_data['acceleration_magnitude'] = np.sqrt(np.array(self.cog_data['ax'])**2 + np.array(self.cog_data['ay'])**2)
        
        # Calculate jerk values (新增)
        self.cog_data['jx'] = np.gradient(self.cog_data['ax'], time_step)
        self.cog_data['jy'] = np.gradient(self.cog_data['ay'], time_step)
        self.cog_data['jerk_magnitude'] = np.sqrt(np.array(self.cog_data['jx'])**2 + np.array(self.cog_data['jy'])**2)
        
        print("  -> CoG analysis finished with normalized positions, velocities, accelerations, and jerks.")

    # --- Plotting Methods ---
    def _plot_template(self, data_dict, title, ylabel, save_path, show_plot, xlim=None, ylim=None):
        if not data_dict: raise ValueError(f"No data for {title}.")
        
        # 使用真实时间轴（秒）
        time_axis = np.arange(self.frame_count) / self.fps
        fig, ax = self._setup_figure()
        
        for name, values in data_dict.items():
            color = self.plot_config['colors'].get(name, 'black')
            # 使用中文标签
            label = self.plot_config['chinese_labels'].get(name, name.replace('_', ' ').title())
            ax.plot(time_axis, values, color=color, 
                   label=label, 
                   linewidth=self.plot_config['line_width'])
        
        self._apply_plot_style(ax, title=title, xlabel='时间 (秒)', ylabel=ylabel)
        ax.legend(fontsize=self.plot_config['font_size']['legend'])
        
        # 设置轴范围
        if xlim:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=self.plot_config['dpi'])
        if show_plot: plt.show()

    def plot_joint_angles(self, save_path=None, show_plot=True):
         self._plot_template(self.output_data['angle'], '平均关节角度随时间变化', '角度 (度)', save_path, show_plot)

    def plot_joint_accelerations(self, save_path=None, show_plot=True):
        self._plot_template(self.output_data['acceleration'], '平均关节角加速度', '加速度 (度/秒^2)', save_path, show_plot)

    def plot_joint_jerks(self, save_path=None, show_plot=True):
        # 计算视频总时长
        total_time = self.frame_count / self.fps
        # Average Joint Angular Jerk (Smoothness): x 0到总时长, y -200000 - 200000
        self._plot_template(self.output_data['jerk'], '平均关节角急动度 (平滑性)', '急动度 (度/秒^3)', save_path, show_plot, xlim=(0, total_time), ylim=(-200000, 200000))

    def plot_kinetic_chain_timing(self, save_path=None, show_plot=True):
        chain_joints = ['hip_avg', 'knee_avg', 'ankle_avg']
        data_to_plot = {name: self.output_data['velocity'][name] for name in chain_joints if name in self.output_data['velocity']}
        if not data_to_plot: print("Could not find required joints for kinetic chain plot."); return
        
        # 使用真实时间轴（秒）
        time_axis = np.arange(self.frame_count) / self.fps
        total_time = self.frame_count / self.fps
        fig, ax = self._setup_figure()
        
        # 去除原来的axhline虚线
        for name, values in data_to_plot.items():
            normalized_values = values / np.nanmax(np.abs(values))
            color = self.plot_config['colors'].get(name, 'black')
            # 使用中文标签
            label = self.plot_config['chinese_labels'].get(name, name.replace('_', ' ').title())
            ax.plot(time_axis, normalized_values, color=color, 
                   label=label,
                   linewidth=self.plot_config['line_width'])
        
        self._apply_plot_style(ax, 
                             title='动力链时序 (归一化角速度)', 
                             xlabel='时间 (秒)', 
                             ylabel='归一化角速度')
        ax.legend(fontsize=self.plot_config['font_size']['legend'])
        
        # Kinetic Chain Timing: x 0到总时长, y -1.5 - 1.5
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-1.5, 1.5)
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=self.plot_config['dpi'])
        if show_plot: plt.show()

    def plot_phase_portraits(self, save_path=None, show_plot=True):
        joints_to_plot = self.output_data['angle'].keys()
        num_joints = len(joints_to_plot)
        fig, axes = plt.subplots(1, num_joints, figsize=(5 * num_joints, 5), squeeze=False)
        fig.suptitle('关节相位图 (速度 vs 角度)', 
                    fontsize=self.plot_config['font_size']['title'], fontweight='bold')
        
        for i, name in enumerate(joints_to_plot):
            ax = axes[0, i]
            angle_data = self.output_data['angle'][name]
            velocity_data = self.output_data['velocity'][name]
            color = self.plot_config['colors'].get(name, 'black')
            
            ax.plot(angle_data, velocity_data, color=color, linewidth=self.plot_config['line_width'])
            ax.plot(angle_data[0], velocity_data[0], 'go', 
                   ms=self.plot_config['marker_size'], label='开始')
            ax.plot(angle_data[-1], velocity_data[-1], 'rs', 
                   ms=self.plot_config['marker_size'], label='结束')
            
            # 使用中文标签
            title = self.plot_config['chinese_labels'].get(name, name.replace('_', ' ').title())
            self._apply_plot_style(ax, 
                                 title=title, 
                                 xlabel='角度 (度)', 
                                 ylabel='角速度 (度/秒)')
            ax.legend(fontsize=self.plot_config['font_size']['legend'])
            
            # Joint Phase Portraits: x 50 - 180, y -400 - 600
            ax.set_xlim(50, 180)
            ax.set_ylim(-400, 600)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path: plt.savefig(save_path, dpi=self.plot_config['dpi'])
        if show_plot: plt.show()
        
    # --- NEW: PEAK TIMING ANALYSIS PLOTS ---
    def _plot_peak_timing(self, data_type, title, ylabel, save_path, show_plot):
        """Generic internal function to find and plot peaks for any data type."""
        if data_type not in self.output_data: raise ValueError(f"{data_type} data not found.")
        
        data_dict = self.output_data[data_type]
        joints_to_plot = ['hip_avg', 'knee_avg', 'ankle_avg']
        
        # 使用真实时间轴（秒）
        time_axis = np.arange(self.frame_count) / self.fps
        total_time = self.frame_count / self.fps
        fig, ax = self._setup_figure()
        
        peaks_info = []
        for name in joints_to_plot:
            if name in data_dict:
                values = data_dict[name]
                if np.all(np.isnan(values)): continue
                
                peak_idx = np.nanargmax(values)
                peak_time = time_axis[peak_idx]  # 使用真实时间
                peak_value = values[peak_idx]
                color = self.plot_config['colors'].get(name, 'black')
                # 使用中文标签
                label = self.plot_config['chinese_labels'].get(name, name.replace('_', ' ').title())
                peaks_info.append({'name': name, 'label': label, 'time': peak_time, 'value': peak_value, 'color': color})
                
                # Plot the full data curve
                ax.plot(time_axis, values, color=color, alpha=0.5, 
                       label=label,
                       linewidth=self.plot_config['line_width'])

        if not peaks_info: print(f"No valid peaks found for {data_type} plot."); return

        # Sort peaks by time to establish the sequence
        peaks_info.sort(key=lambda p: p['time'])
        
        # Mark each peak on the plot (不添加文本框)
        for peak in peaks_info:
            ax.plot(peak['time'], peak['value'], 'o', color=peak['color'], 
                   markersize=self.plot_config['marker_size'], markeredgecolor='black')

        self._apply_plot_style(ax, title=title, xlabel='时间 (秒)', ylabel=ylabel)
        ax.legend(loc='lower right', fontsize=self.plot_config['font_size']['legend'])
        
        # 设置不同的y轴范围，x轴使用真实时间
        ax.set_xlim(0, 1.0)
        if data_type == 'angle':
            # Peak Angle Timing: x 0到总时长, y 50 - 180
            ax.set_ylim(50, 180)
        elif data_type == 'acceleration':
            # Peak Acceleration Timing: x 0到总时长, y -10000 - 10000
            ax.set_ylim(-10000, 10000)
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=self.plot_config['dpi'])
            print(f"Peak timing plot saved to: {os.path.basename(save_path)}")
        if show_plot: plt.show()

    def plot_peak_angle_timing(self, save_path=None, show_plot=True):
        """Public method to plot the timing of peak angles."""
        self._plot_peak_timing('angle', '峰值角度时序', '角度 (度)', save_path, show_plot)

    def plot_peak_acceleration_timing(self, save_path=None, show_plot=True):
        """Public method to plot the timing of peak accelerations."""
        self._plot_peak_timing('acceleration', '峰值加速度时序', '加速度 (度/秒^2)', save_path, show_plot)

    def plot_center_of_gravity_trajectory(self, save_path=None, show_plot=True):
        """Plot center of gravity trajectory (2D path) with normalized axes."""
        if not self.cog_data:
            raise ValueError("No center of gravity data available. Run analyze_center_of_gravity() first.")
        
        fig, ax = self._setup_figure()
        
        # Plot CoG trajectory (X vs Y path)
        ax.plot(self.cog_data['x'], self.cog_data['y'], 'g-', 
                linewidth=self.plot_config['line_width'], alpha=0.7)
        ax.scatter(self.cog_data['x'][0], self.cog_data['y'][0], color='green', 
                   s=100, label='开始', zorder=5)
        ax.scatter(self.cog_data['x'][-1], self.cog_data['y'][-1], color='red', 
                   s=100, label='结束', zorder=5)
        
        self._apply_plot_style(ax,
                             title='重心轨迹 (2D路径)',
                             xlabel='X位置 (归一化)',
                             ylabel='Y位置 (归一化)')
        ax.legend(fontsize=self.plot_config['font_size']['legend'])
        
        # 根据实际数据范围动态设置坐标轴范围
        x_data = [x for x in self.cog_data['x'] if not np.isnan(x)]
        y_data = [y for y in self.cog_data['y'] if not np.isnan(y)]
        
        if x_data and y_data:
            # 计算数据范围并添加适当的边距
            x_min, x_max = min(x_data), max(x_data)
            y_min, y_max = min(y_data), max(y_data)
            
            # 添加10%的边距以便更好地显示数据
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # 防止范围为0的情况
            if x_range == 0:
                x_range = 0.1
            if y_range == 0:
                y_range = 0.1
            
            x_margin = x_range * 0.1
            y_margin = y_range * 0.1
            
            # 设置轴范围
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            
            # 动态设置刻度间隔
            import matplotlib.ticker as ticker
            
            # 根据数据范围计算合适的刻度间隔
            x_tick_interval = x_range / 5  # 大约5个主要刻度
            y_tick_interval = y_range / 5
            
            # 将刻度间隔舍入到合适的值
            x_tick_interval = round(x_tick_interval, 2) or 0.1
            y_tick_interval = round(y_tick_interval, 2) or 0.1
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_interval))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_interval))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_tick_interval/2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_tick_interval/2))
            
            print(f"  -> CoG trajectory range: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}]")
        else:
            # 如果没有有效数据，使用默认范围
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            print("  -> Warning: No valid CoG data, using default range [0, 1]")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.plot_config['dpi'])
            print(f"  -> Center of gravity trajectory plot saved: {os.path.basename(save_path)}")
        
        if show_plot:
            plt.show()

    def plot_center_of_gravity_acceleration_magnitude(self, save_path=None, show_plot=True):
        """Plot center of gravity acceleration magnitude over real time."""
        if not self.cog_data:
            raise ValueError("No center of gravity data available. Run analyze_center_of_gravity() first.")
        
        # 使用真实时间轴（秒）
        time_axis = np.arange(self.frame_count) / self.fps
        total_time = self.frame_count / self.fps
        
        fig, ax = self._setup_figure()
        
        # Plot CoG acceleration magnitude
        # Plot CoG acceleration magnitude
        ax.plot(time_axis, self.cog_data['acceleration_magnitude'], 'orange', 
                linewidth=self.plot_config['line_width'])
        
        self._apply_plot_style(ax,
                             title='重心加速度幅度随时间变化',
                             xlabel='时间 (秒)',
                             ylabel='加速度幅度 (归一化单位/秒^2)')
        
        # Center of Gravity Acceleration Magnitude: x 0到总时长, y 0-15
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.plot_config['dpi'])
            print(f"  -> Center of gravity acceleration magnitude plot saved: {os.path.basename(save_path)}")
        
        if show_plot:
            plt.show()

    def save_data_to_csv(self, csv_path):
        if not self.joint_angles: raise ValueError("No data to save.")
        print(f"Saving comprehensive data to {os.path.basename(csv_path)}...")
        data = {'time_seconds': np.arange(self.frame_count) / self.fps}
        
        # Add joint data
        for joint_name in self.joints_to_calculate.keys():
            data[f'{joint_name}_angle'] = self.joint_angles.get(joint_name, [])
            data[f'{joint_name}_velocity'] = self.joint_velocities.get(joint_name, [])
            data[f'{joint_name}_acceleration'] = self.joint_accelerations.get(joint_name, [])
            data[f'{joint_name}_jerk'] = self.joint_jerks.get(joint_name, [])
        
        # Add averaged joint data
        for avg_name in self.joint_averaging_pairs.keys():
            data[f'{avg_name}_angle'] = self.output_data['angle'].get(avg_name, [])
            data[f'{avg_name}_velocity'] = self.output_data['velocity'].get(avg_name, [])
            data[f'{avg_name}_acceleration'] = self.output_data['acceleration'].get(avg_name, [])
            data[f'{avg_name}_jerk'] = self.output_data['jerk'].get(avg_name, [])
        
        # Add center of gravity data if available
        if self.cog_data:
            # Raw positions (in pixels)
            data['cog_x_raw'] = self.cog_data.get('x_raw', [])
            data['cog_y_raw'] = self.cog_data.get('y_raw', [])
            
            # Normalized positions
            data['cog_x_normalized'] = self.cog_data.get('x', [])
            data['cog_y_normalized'] = self.cog_data.get('y', [])
            
            # Velocities
            data['cog_velocity_x'] = self.cog_data.get('vx', [])
            data['cog_velocity_y'] = self.cog_data.get('vy', [])
            
            # Accelerations
            data['cog_acceleration_x'] = self.cog_data.get('ax', [])
            data['cog_acceleration_y'] = self.cog_data.get('ay', [])
            
            # Jerks (新增)
            data['cog_jerk_x'] = self.cog_data.get('jx', [])
            data['cog_jerk_y'] = self.cog_data.get('jy', [])
            
            # Magnitude values
            data['cog_speed'] = self.cog_data.get('speed', [])
            data['cog_acceleration_magnitude'] = self.cog_data.get('acceleration_magnitude', [])
            data['cog_jerk_magnitude'] = self.cog_data.get('jerk_magnitude', [])  
        
        df = pd.DataFrame.from_dict(data, orient='index').transpose()
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

def midpoint(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

def main():
    analyzer = SimpleJointAnalyzer()
    
    # --- IMPORTANT: CHANGE THIS PATH TO YOUR VIDEO FILE ---
    video_path = "E:/Basketball/for_CSCD_publication/lunge-shot-1-simplified.mp4" 
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at '{video_path}'"); return
    
    # --- Auto-generate output filenames ---
    input_dir  = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_dir = os.path.join(input_dir, f"{base_name}_plot_graph_{ts}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory: {results_dir}")
    
    paths = {
        'csv': os.path.join(results_dir, f"{base_name}_data_{ts}.csv"),
        'timing_angle': os.path.join(results_dir, f"{base_name}_peak_angles_{ts}.png"),
        'timing_accel': os.path.join(results_dir, f"{base_name}_peak_accel_{ts}.png"),
        'timing_kinetic': os.path.join(results_dir, f"{base_name}_kinetic_chain_{ts}.png"),
        'phase': os.path.join(results_dir, f"{base_name}_phase_portraits_{ts}.png"),
        'jerk': os.path.join(results_dir, f"{base_name}_jerk_{ts}.png"),
        'cog_trajectory': os.path.join(results_dir, f"{base_name}_cog_trajectory_{ts}.png"),  # 修改
        'cog_acceleration': os.path.join(results_dir, f"{base_name}_cog_acceleration_{ts}.png"),  # 新增
    }
    
    try:
        # --- Main Analysis Pipeline ---
        analyzer.process_video(video_path)
        analyzer._calculate_kinematics_and_average()
        analyzer.analyze_center_of_gravity()

        # --- Generate and Save All Plots ---
        print("\n--- Generating Plots ---")
        analyzer.plot_peak_angle_timing(save_path=paths['timing_angle'], show_plot=True)
        analyzer.plot_peak_acceleration_timing(save_path=paths['timing_accel'], show_plot=True)
        analyzer.plot_kinetic_chain_timing(save_path=paths['timing_kinetic'], show_plot=True)
        analyzer.plot_phase_portraits(save_path=paths['phase'], show_plot=True)
        analyzer.plot_joint_jerks(save_path=paths['jerk'], show_plot=True)
        analyzer.plot_center_of_gravity_trajectory(save_path=paths['cog_trajectory'], show_plot=True)  # 修改
        analyzer.plot_center_of_gravity_acceleration_magnitude(save_path=paths['cog_acceleration'], show_plot=True)  # 新增
        
        # --- Save All Data to a Single CSV File ---
        analyzer.save_data_to_csv(paths['csv'])
        
        print("\n--- Analysis Complete! ---")
        print(f"All output files have been saved in: {results_dir}")
        for key, path in paths.items():
            print(f"  - {key.title()} Plot: {os.path.basename(path)}")
        
    except Exception as e:
        import traceback
        print(f"\n--- An Error Occurred ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()