import argparse
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os

class JointLabeller:
    """
    简化版关节角度分析器
    功能：从视频中提取人体关节位置，并在视频和截图中进行可视化标记。
    """

    def __init__(self):
        # MediaPipe姿态检测初始化
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.frame_count = 0
        self.fps = 30 # 默认值，会被视频实际帧率覆盖

        # 关节定义 (使用MediaPipe的landmark索引)
        self.joints = {
            # 右侧关节
            'ankle_right': (26, 28, 32),
            'knee_right': (24, 26, 28),
            'hip_right': (12, 24, 26),
            'elbow_right': (12, 14, 16),
            'wrist_right': (14, 16, 18),
            'shoulder_right': (11, 12, 14),

            # 左侧关节
            'ankle_left': (25, 27, 31),
            'knee_left': (23, 25, 27),
            'hip_left': (11, 23, 25),
            'elbow_left': (11, 13, 15),
            'wrist_left': (13, 15, 17),
            'shoulder_left': (12, 11, 13),
        }

    def extract_landmarks(self, image):
        """
        从图像中提取关节点位置
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if results.pose_landmarks:
            landmarks = []
            h, w, _ = image.shape

            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))

            return landmarks, results

        return None, None

    def draw_joint_markers(self, image, landmarks, joint_name, points_indices):
        """
        在图像上绘制关节标记
        """
        colors = {
            'ankle_right': (0, 255, 0), 'knee_right': (255, 0, 0),
            'hip_right': (0, 0, 255), 'elbow_right': (255, 255, 0),
            'wrist_right': (0, 255, 255), 'shoulder_right': (255, 0, 255),
            'ankle_left': (0, 128, 0), 'knee_left': (128, 0, 0),
            'hip_left': (0, 0, 128), 'elbow_left': (128, 128, 0),
            'wrist_left': (0, 128, 128), 'shoulder_left': (128, 0, 128),
        }
        color = colors.get(joint_name, (255, 255, 255))

        joint_point = landmarks[points_indices[1]]
        cv2.circle(image, joint_point, 8, color, -1)

        point1 = landmarks[points_indices[0]]
        point2 = landmarks[points_indices[1]]
        point3 = landmarks[points_indices[2]]

        cv2.line(image, point1, point2, color, 2)
        cv2.line(image, point2, point3, color, 2)

        cv2.putText(image, joint_name.replace('_', ' ').title(),
                    (joint_point[0] + 10, joint_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image

    def process_video_and_capture(self, video_path, output_video_path=None, screenshot_dir=None):
        """
        处理视频文件，提取关节点并生成带标记的视频和截图
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_video_path:
            # 确保视频输出路径的目录存在
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (width, height))

        if screenshot_dir:
            os.makedirs(screenshot_dir, exist_ok=True)

        print("开始处理视频并生成带标记视频/截图...")
        self.frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, _ = self.extract_landmarks(frame)
            if landmarks:
                for joint_name, points_indices in self.joints.items():
                    try:
                        if all(idx < len(landmarks) for idx in points_indices):
                            frame = self.draw_joint_markers(frame, landmarks, joint_name, points_indices)
                    except Exception as e:
                        print(f"绘制{joint_name}标记时出错: {e}")

            if out:
                out.write(frame)

            if screenshot_dir and (self.frame_count % 5 == 0):
                screenshot_path = os.path.join(screenshot_dir, f"frame_{self.frame_count:05d}.png")
                cv2.imwrite(screenshot_path, frame)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                print(f"已处理 {self.frame_count} 帧")

        cap.release()
        if out:
            out.release()

        print(f"\n视频处理完成，共处理 {self.frame_count} 帧")

def main():
    """
    为视频生成带关节标记的视频和每 5 帧一张的截图
    """
    parser = argparse.ArgumentParser(description="Draw MediaPipe joint markers on a video and save screenshots.")
    parser.add_argument("video", help="input video file")
    parser.add_argument("--out-dir", help="where to create the <video>_analysis_<timestamp> folder (default: the video's folder)")
    args = parser.parse_args()

    # 1. 视频路径
    video_path = args.video

    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在于 '{video_path}'")
        return

    analyzer = JointLabeller()

    # 2. 创建输出主文件夹（默认在视频所在目录下）
    output_root = args.out_dir or os.path.dirname(os.path.abspath(video_path))
    # 获取不带扩展名的视频文件名
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 新的文件夹名
    new_folder_name = f"{video_base_name}_analysis_{timestamp}"
    
    output_dir = os.path.join(output_root, new_folder_name)

    # 创建这个主文件夹
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有输出内容将保存在文件夹: '{output_dir}'")

    # 3. 定义所有输出文件的路径，使其位于新创建的文件夹内
    output_video_path = os.path.join(output_dir, f"{video_base_name}_marked.mp4")
    screenshot_folder_path = os.path.join(output_dir, "screenshots")

    try:
        # 4. 处理视频，并将所有输出指向指定路径
        analyzer.process_video_and_capture(
            video_path,
            output_video_path=output_video_path,
            screenshot_dir=screenshot_folder_path
        )

        print("分析全部完成！")
        print(f"输出文件:")
        print(f"  标记视频: {output_video_path}")
        print(f"  截图目录: {screenshot_folder_path}")

    except Exception as e:
        print(f"处理过程中出现严重错误: {e}")

if __name__ == "__main__":
    main()
