import argparse
import os
from PIL import Image

def crop_images_in_folder(input_folder, crop_width=752, crop_height=842):
    """
    Crops all PNG images in a specified folder to a given size, keeping the bottom-left corner fixed.

    Args:
        input_folder (str): The path to the folder containing the images.
        crop_width (int): The width of the cropped image.
        crop_height (int): The height of the cropped image.
    """
    # 在输入文件夹下创建一个新的文件夹来保存裁剪后的图片
    output_folder = os.path.join(input_folder, "cropped_images")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 获取文件夹中所有文件的列表
    files = os.listdir(input_folder)
    
    # 筛选出PNG文件
    png_files = [f for f in files if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"在文件夹 {input_folder} 中没有找到PNG图片。")
        return

    print(f"开始处理 {len(png_files)} 个PNG图片...")

    for filename in png_files:
        try:
            # 构建完整的文件路径
            file_path = os.path.join(input_folder, filename)
            
            # 打开图片
            with Image.open(file_path) as img:
                original_width, original_height = img.size
                
                # 定义裁剪区域 (left, upper, right, lower)
                # PIL库的坐标原点是左上角
                left = 0
                right = left + crop_width
                # 保持左下角不变，所以bottom是图片原始高度
                bottom = original_height
                # upper是原始高度减去裁剪高度
                upper = bottom - crop_height

                # 检查裁剪尺寸是否超出原图范围
                if crop_width > original_width or crop_height > original_height:
                    print(f"警告: 文件 '{filename}' ({original_width}x{original_height}) 的尺寸小于指定的裁剪尺寸 ({crop_width}x{crop_height})，已跳过。")
                    continue
                
                # 裁剪图片
                cropped_img = img.crop((left, upper, right, bottom))
                
                # 构建输出文件路径
                output_path = os.path.join(output_folder, f"cropped_{filename}")
                
                # 保存裁剪后的图片
                cropped_img.save(output_path)
                print(f"已裁剪 '{filename}' 并保存到 '{output_path}'")

        except Exception as e:
            print(f"处理文件 '{filename}' 时发生错误: {e}")

    print("\n所有图片处理完成！")
    print(f"裁剪后的图片保存在: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop every PNG in a folder, keeping the bottom-left corner fixed.")
    parser.add_argument("folder", help="folder of PNG screenshots; output goes to <folder>/cropped_images")
    parser.add_argument("--width", type=int, default=752, help="crop width in px (default: 752)")
    parser.add_argument("--height", type=int, default=842, help="crop height in px (default: 842)")
    args = parser.parse_args()

    # 检查文件夹是否存在
    if not os.path.isdir(args.folder):
        print(f"错误: 文件夹路径不存在 -> {args.folder}")
    else:
        # 调用函数开始裁剪
        crop_images_in_folder(args.folder, crop_width=args.width, crop_height=args.height)