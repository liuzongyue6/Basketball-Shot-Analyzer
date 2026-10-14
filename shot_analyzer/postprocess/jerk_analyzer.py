import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

class JerkProcessor:
    """
    处理CSV文件中的cog_jerk_magnitude数据
    计算Jerk幅度的均方根（RMS），并使用指数衰减函数映射到0-1区间
    """
    
    def __init__(self, jerk_ref=120.0, mapping_type="sigmoid"):
        """
        初始化处理器
        
        Args:
            jerk_ref (float): 参考Jerk值，用于指数衰减函数的归一化，默认为120.0
            mapping_type (str): 映射函数类型
        """
        self.jerk_ref = jerk_ref
        self.mapping_type = mapping_type
        # 设置matplotlib统一字体支持中文
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_csv_files(self, folder_path):
        """
        加载文件夹中的所有CSV文件
        
        Args:
            folder_path (str): 包含CSV文件的文件夹路径
            
        Returns:
            list: CSV文件路径列表
        """
        csv_files = []
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(folder_path, file))
        return csv_files
    
    def calculate_jerk_rms(self, jerk_data):
        """
        计算Jerk幅度的均方根（RMS）
        
        Args:
            jerk_data (numpy.array): Jerk幅度数据
            
        Returns:
            float: Jerk RMS值
        """
        return np.sqrt(np.mean(jerk_data ** 2))
    
    def exponential_decay_mapping(self, jerk_rms, jerk_ref=None, mapping_type="exponential"):
        """
        使用不同的函数将Jerk RMS映射到0-1区间
        
        Args:
            jerk_rms (float): Jerk RMS值
            jerk_ref (float): 参考Jerk值，如果None则使用实例化时的值
            mapping_type (str): 映射函数类型
                - "exponential": 1 - exp(-jerk_rms / jerk_ref)
                - "sigmoid": 1 / (1 + exp(-(jerk_rms - jerk_ref) / (jerk_ref * 0.2)))
                - "power": (jerk_rms / jerk_ref) ^ 2 (截断到1)
                - "linear_threshold": 线性映射，但有阈值
                - "tanh": tanh函数映射
            
        Returns:
            float: 映射到0-1区间的值
        """
        if jerk_ref is None:
            jerk_ref = self.jerk_ref
        
        if mapping_type == "exponential":
            # 原始指数衰减函数
            mapped_value = 1 - np.exp(-jerk_rms / jerk_ref)
            
        elif mapping_type == "sigmoid":
            # Sigmoid函数，在jerk_ref附近有更陡的斜率
            # 调整参数使得在jerk_ref处约为0.5，斜率可调
            steepness = 0.2  # 越小越陡
            mapped_value = 1 / (1 + np.exp(-(jerk_rms - jerk_ref) / (jerk_ref * steepness)))
            
        elif mapping_type == "power":
            # 幂函数映射，对高值更敏感
            power = 2.0  # 可以调整幂次
            mapped_value = (jerk_rms / jerk_ref) ** power
            mapped_value = min(mapped_value, 1.0)  # 截断到1
            
        elif mapping_type == "linear_threshold":
            # 线性映射但有阈值
            threshold = jerk_ref * 0.5  # 阈值
            if jerk_rms <= threshold:
                mapped_value = 0.0
            else:
                # 从阈值到2*jerk_ref线性映射到0-1
                max_val = jerk_ref * 2.0
                mapped_value = (jerk_rms - threshold) / (max_val - threshold)
                mapped_value = min(mapped_value, 1.0)
                
        elif mapping_type == "tanh":
            # tanh函数，S型曲线但有上下界
            # tanh((jerk_rms - jerk_ref) / (jerk_ref * 0.3)) 然后缩放到0-1
            steepness = 0.3
            mapped_value = np.tanh((jerk_rms - jerk_ref * 0.5) / (jerk_ref * steepness))
            mapped_value = (mapped_value + 1) / 2  # 从[-1,1]映射到[0,1]
            
        else:
            # 默认使用指数函数
            mapped_value = 1 - np.exp(-jerk_rms / jerk_ref)
        
        return mapped_value
    
    def calculate_kinetic_chain_score(self, df, k=1.0):
        """
        计算动力学链协调评分 (KCSS)
        
        Args:
            df (pandas.DataFrame): 包含关节加速度数据的DataFrame
            k (float): 高斯函数的锐度参数，默认为1.0
            
        Returns:
            dict: 包含延迟分析和KCSS评分的字典
        """
        try:
            # 检查必需的列是否存在
            required_cols = ['time_seconds', 'hip_avg_acceleration', 'knee_avg_acceleration', 'ankle_avg_acceleration']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {
                    'error': f'缺少必需的列: {missing_cols}'
                }
            
            # 获取时间和加速度数据
            time = df['time_seconds'].values
            hip_accel = df['hip_avg_acceleration'].values
            knee_accel = df['knee_avg_acceleration'].values
            ankle_accel = df['ankle_avg_acceleration'].values
            
            # 移除NaN值的索引
            valid_indices = ~(np.isnan(hip_accel) | np.isnan(knee_accel) | np.isnan(ankle_accel) | np.isnan(time))
            
            if np.sum(valid_indices) == 0:
                return {
                    'error': '没有有效的加速度数据'
                }
            
            time_valid = time[valid_indices]
            hip_accel_valid = hip_accel[valid_indices]
            knee_accel_valid = knee_accel[valid_indices]
            ankle_accel_valid = ankle_accel[valid_indices]
            
            # 1. 找到各关节加速度的最大值及对应时间
            hip_max_idx = np.argmax(np.abs(hip_accel_valid))
            knee_max_idx = np.argmax(np.abs(knee_accel_valid))
            ankle_max_idx = np.argmax(np.abs(ankle_accel_valid))
            
            hip_max_time = time_valid[hip_max_idx]
            knee_max_time = time_valid[knee_max_idx]
            ankle_max_time = time_valid[ankle_max_idx]
            
            hip_max_value = hip_accel_valid[hip_max_idx]
            knee_max_value = knee_accel_valid[knee_max_idx]
            ankle_max_value = ankle_accel_valid[ankle_max_idx]
            
            # 2. 计算延迟
            delay1 = knee_max_time - hip_max_time  # 髋关节到膝关节的延迟
            delay2 = ankle_max_time - knee_max_time  # 膝关节到踝关节的延迟
            
            # 计算总延迟
            d_total = delay1 + delay2
            
            # 3. 计算延迟比率
            if d_total != 0:
                r_delay = delay1 / d_total
            else:
                r_delay = 0.5  # 如果总延迟为0，设为中性值
            
            # 4. 使用高斯函数计算KCSS评分
            # KCSS = e^(-k(R_delay - 0.5)²)
            kcss = np.exp(-k * (r_delay - 0.5) ** 2)
            
            return {
                'hip_max_time': hip_max_time,
                'knee_max_time': knee_max_time,
                'ankle_max_time': ankle_max_time,
                'hip_max_value': hip_max_value,
                'knee_max_value': knee_max_value,
                'ankle_max_value': ankle_max_value,
                'delay1': delay1,
                'delay2': delay2,
                'd_total': d_total,
                'r_delay': r_delay,
                'kcss': kcss,
                'k_parameter': k
            }
            
        except Exception as e:
            return {
                'error': f'计算动力学链评分时出错: {str(e)}'
            }

    def process_csv_file(self, csv_file_path):
        """
        处理单个CSV文件
        
        Args:
            csv_file_path (str): CSV文件路径
            
        Returns:
            dict: 处理结果字典
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            
            # 检查是否包含cog_jerk_magnitude列
            if 'cog_jerk_magnitude' not in df.columns:
                return {
                    'file': csv_file_path,
                    'error': 'CSV文件中未找到cog_jerk_magnitude列'
                }
            
            # 获取cog_jerk_magnitude数据
            jerk_data = df['cog_jerk_magnitude'].values
            
            # 移除NaN值
            jerk_data = jerk_data[~np.isnan(jerk_data)]
            
            if len(jerk_data) == 0:
                return {
                    'file': csv_file_path,
                    'error': 'cog_jerk_magnitude列中没有有效数据'
                }
            
            # 计算Jerk RMS
            jerk_rms = self.calculate_jerk_rms(jerk_data)
            
            # 使用指定的映射函数映射到0-1区间
            mapped_value = self.exponential_decay_mapping(jerk_rms, mapping_type=self.mapping_type)
            
            # 计算动力学链协调评分
            kinetic_chain_result = self.calculate_kinetic_chain_score(df)
            
            result = {
                'file': csv_file_path,
                'filename': os.path.basename(csv_file_path),
                'jerk_rms': jerk_rms,
                'mapped_value': mapped_value,
                'jerk_ref': self.jerk_ref,
                'mapping_type': self.mapping_type,
                'data_points': len(jerk_data),
                'jerk_mean': np.mean(jerk_data),
                'jerk_std': np.std(jerk_data),
                'jerk_min': np.min(jerk_data),
                'jerk_max': np.max(jerk_data),
                'kinetic_chain': kinetic_chain_result
            }
            
            return result
            
        except Exception as e:
            return {
                'file': csv_file_path,
                'error': f'处理文件时出错: {str(e)}'
            }
    
    def process_folder(self, folder_path, output_file=None):
        """
        处理文件夹中的所有CSV文件
        
        Args:
            folder_path (str): 包含CSV文件的文件夹路径
            output_file (str): 输出文件路径，如果None则自动生成在输入文件夹中
            
        Returns:
            str: 输出文件路径
        """
        # 加载CSV文件
        csv_files = self.load_csv_files(folder_path)
        
        if not csv_files:
            print(f"在文件夹 {folder_path} 中未找到CSV文件")
            return None
        
        # 处理每个CSV文件
        results = []
        for csv_file in csv_files:
            print(f"正在处理: {os.path.basename(csv_file)}")
            result = self.process_csv_file(csv_file)
            results.append(result)
        
        # 生成输出文件名，保存在输入文件夹中
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"jerk_analysis_results_{timestamp}.txt"
            output_file = os.path.join(folder_path, output_filename)
        
        # 写入结果到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Jerk Analysis and Kinetic Chain Coordination Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理文件夹: {folder_path}\n")
            f.write(f"参考Jerk值 (jerk_ref): {self.jerk_ref}\n")
            f.write(f"映射函数类型: {self.mapping_type}\n")
            f.write(f"映射函数: 根据类型而定\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"文件 {i}: {result['filename']}\n")
                f.write("-" * 50 + "\n")
                
                if 'error' in result:
                    f.write(f"错误: {result['error']}\n")
                else:
                    # Jerk分析结果
                    f.write(f"=== Jerk分析结果 ===\n")
                    f.write(f"数据点数量: {result['data_points']}\n")
                    f.write(f"Jerk统计信息:\n")
                    f.write(f"  - 均值: {result['jerk_mean']:.6f}\n")
                    f.write(f"  - 标准差: {result['jerk_std']:.6f}\n")
                    f.write(f"  - 最小值: {result['jerk_min']:.6f}\n")
                    f.write(f"  - 最大值: {result['jerk_max']:.6f}\n")
                    f.write(f"Jerk RMS: {result['jerk_rms']:.6f}\n")
                    f.write(f"映射值 (0-1区间): {result['mapped_value']:.6f}\n\n")
                    
                    # 动力学链协调分析结果
                    kinetic = result['kinetic_chain']
                    if 'error' in kinetic:
                        f.write(f"=== 动力学链分析结果 ===\n")
                        f.write(f"错误: {kinetic['error']}\n")
                    else:
                        f.write(f"=== 动力学链协调分析结果 ===\n")
                        f.write(f"关节峰值加速度时间:\n")
                        f.write(f"  - 髋关节峰值时间: {kinetic['hip_max_time']:.6f}s\n")
                        f.write(f"  - 膝关节峰值时间: {kinetic['knee_max_time']:.6f}s\n")
                        f.write(f"  - 踝关节峰值时间: {kinetic['ankle_max_time']:.6f}s\n")
                        f.write(f"关节峰值加速度值:\n")
                        f.write(f"  - 髋关节峰值: {kinetic['hip_max_value']:.6f}\n")
                        f.write(f"  - 膝关节峰值: {kinetic['knee_max_value']:.6f}\n")
                        f.write(f"  - 踝关节峰值: {kinetic['ankle_max_value']:.6f}\n")
                        f.write(f"延迟分析:\n")
                        f.write(f"  - Delay1 (髋→膝): {kinetic['delay1']:.6f}s\n")
                        f.write(f"  - Delay2 (膝→踝): {kinetic['delay2']:.6f}s\n")
                        f.write(f"  - 总延迟 D_total: {kinetic['d_total']:.6f}s\n")
                        f.write(f"  - 延迟比率 R_delay: {kinetic['r_delay']:.6f}\n")
                        f.write(f"动力学链协调评分 (KCSS): {kinetic['kcss']:.6f}\n")
                        f.write(f"  - 使用参数 k = {kinetic['k_parameter']}\n")
                        f.write(f"  - 公式: KCSS = exp(-k*(R_delay-0.5)²)\n")
                
                f.write("\n")
            
            # 添加总结
            successful_results = [r for r in results if 'error' not in r]
            if successful_results:
                f.write("=" * 80 + "\n")
                f.write("总结\n")
                f.write("=" * 80 + "\n")
                f.write(f"成功处理的文件数量: {len(successful_results)}\n")
                f.write(f"失败的文件数量: {len(results) - len(successful_results)}\n")
                
                if len(successful_results) > 1:
                    all_jerk_rms = [r['jerk_rms'] for r in successful_results]
                    all_mapped = [r['mapped_value'] for r in successful_results]
                    all_kcss = [r['kinetic_chain']['kcss'] for r in successful_results if 'error' not in r['kinetic_chain']]
                    
                    f.write(f"\n所有文件的Jerk RMS统计:\n")
                    f.write(f"  - 均值: {np.mean(all_jerk_rms):.6f}\n")
                    f.write(f"  - 标准差: {np.std(all_jerk_rms):.6f}\n")
                    f.write(f"  - 最小值: {np.min(all_jerk_rms):.6f}\n")
                    f.write(f"  - 最大值: {np.max(all_jerk_rms):.6f}\n")
                    
                    f.write(f"\n所有文件的映射值统计:\n")
                    f.write(f"  - 均值: {np.mean(all_mapped):.6f}\n")
                    f.write(f"  - 标准差: {np.std(all_mapped):.6f}\n")
                    f.write(f"  - 最小值: {np.min(all_mapped):.6f}\n")
                    f.write(f"  - 最大值: {np.max(all_mapped):.6f}\n")
                    
                    if all_kcss:
                        f.write(f"\n所有文件的KCSS评分统计:\n")
                        f.write(f"  - 均值: {np.mean(all_kcss):.6f}\n")
                        f.write(f"  - 标准差: {np.std(all_kcss):.6f}\n")
                        f.write(f"  - 最小值: {np.min(all_kcss):.6f}\n")
                        f.write(f"  - 最大值: {np.max(all_kcss):.6f}\n")
        
        print(f"分析结果已保存到: {output_file}")
        return output_file

    def plot_cog_jerk_xy(self, csv_file_path, output_dir=None, show_plot=False):
        """
        绘制cog_jerk_x和cog_jerk_y在同一张图上
        
        Args:
            csv_file_path (str): CSV文件路径
            output_dir (str): 输出图片的目录，如果None则保存在CSV文件同目录
            show_plot (bool): 是否显示图片，默认False
            
        Returns:
            str: 保存的图片路径，如果出错则返回None
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            
            # 检查必需的列是否存在
            required_cols = ['cog_jerk_x', 'cog_jerk_y']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"错误: CSV文件中未找到列: {missing_cols}")
                return None
            
            # 获取数据
            jerk_x = df['cog_jerk_x'].values
            jerk_y = df['cog_jerk_y'].values
            
            # 检查是否有时间列
            if 'time_seconds' in df.columns:
                time_data = df['time_seconds'].values
                x_label = '时间 (秒)'
            elif 'frame' in df.columns:
                time_data = df['frame'].values
                x_label = '帧数'
            else:
                # 如果没有时间列，使用索引
                time_data = np.arange(len(jerk_x))
                x_label = '数据点索引'
            
            # 创建图形，增大图形尺寸
            plt.figure(figsize=(14, 10))
            
            # 绘制cog_jerk_x和cog_jerk_y，增加线条粗细
            plt.plot(time_data, jerk_x, label='重心急动度 X', color='blue', linewidth=3, alpha=0.8)
            plt.plot(time_data, jerk_y, label='重心急动度 Y', color='red', linewidth=3, alpha=0.8)
            
            # 设置图形属性，统一字体大小
            plt.xlabel(x_label, fontweight='bold')
            plt.ylabel('急动度 (度/秒^3)', fontweight='bold')
            plt.title('重心急动度 (X和Y分量)', fontweight='bold', pad=20)
            plt.legend(loc='upper right')
            plt.grid(True, axis='x', linestyle='--', alpha=0.7, linewidth=1)
            
            # 自动调整布局
            plt.tight_layout()
            
            # 确定输出目录和文件名
            if output_dir is None:
                output_dir = os.path.dirname(csv_file_path)
            
            # 创建输出目录（如果不存在）
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_name}_cog_jerk_xy_{timestamp}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存图片，增加分辨率
            plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
            print(f"图片已保存到: {output_path}")
            
            # 是否显示图片
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"绘制图片时出错: {str(e)}")
            return None

    def plot_folder_cog_jerk_xy(self, folder_path, output_dir=None, show_plot=False):
        """
        为文件夹中的所有CSV文件绘制cog_jerk_x和cog_jerk_y图
        
        Args:
            folder_path (str): 包含CSV文件的文件夹路径
            output_dir (str): 输出图片的目录，如果None则在各CSV文件同目录
            show_plot (bool): 是否显示图片，默认False
            
        Returns:
            list: 成功保存的图片路径列表
        """
        # 加载CSV文件
        csv_files = self.load_csv_files(folder_path)
        
        if not csv_files:
            print(f"在文件夹 {folder_path} 中未找到CSV文件")
            return []
        
        successful_plots = []
        
        # 为每个CSV文件绘制图片
        for csv_file in csv_files:
            print(f"正在为文件绘制图片: {os.path.basename(csv_file)}")
            
            # 如果指定了输出目录，使用该目录；否则使用CSV文件所在目录
            plot_output_dir = output_dir if output_dir else os.path.dirname(csv_file)
            
            result_path = self.plot_cog_jerk_xy(csv_file, plot_output_dir, show_plot)
            
            if result_path:
                successful_plots.append(result_path)
        
        print(f"\n成功绘制了 {len(successful_plots)} 张图片")
        return successful_plots


def main():
    """
    主函数：处理指定文件夹中的CSV文件
    """
    parser = argparse.ArgumentParser(description="Score CoG jerk and plot CoG jerk X/Y for every CSV in a joint_analyzer output folder.")
    parser.add_argument("folder", help="a *_plot_graph_* folder; results are written into it")
    parser.add_argument("--jerk-ref", type=float, default=120.0, help="reference jerk RMS (default: 120.0)")
    parser.add_argument("--mapping", default="sigmoid",
                        choices=["exponential", "sigmoid", "power", "linear_threshold", "tanh"],
                        help="jerk-to-score mapping (default: sigmoid)")
    args = parser.parse_args()
    folder_path = args.folder

    # 创建处理器实例
    processor = JerkProcessor(jerk_ref=args.jerk_ref, mapping_type=args.mapping)
    
    # 处理文件夹中的CSV文件
    output_file = processor.process_folder(folder_path)
    
    if output_file:
        print(f"\n处理完成！结果保存在: {output_file}")
    else:
        print("\n处理失败或未找到CSV文件")
    
    # 新增：绘制所有CSV文件的cog_jerk_x和cog_jerk_y图
    print("\n开始绘制COG Jerk XY图...")
    plot_results = processor.plot_folder_cog_jerk_xy(folder_path, show_plot=False)
    
    if plot_results:
        print(f"成功绘制了 {len(plot_results)} 张COG Jerk XY图")
    else:
        print("未能绘制任何图片")


if __name__ == "__main__":
    main()
