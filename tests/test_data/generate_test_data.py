import cv2
import os
from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs

def downsample_image(image_path, size):
    # 读取图像
    image = cv2.imread(image_path)
    # 使用指定的大小调整图像
    resized = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return resized

def prepare_keys_test(folder_path):
    """Prepare image path list and keys for test dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def downsample_lr(input_dir, output_dir, downsample_ratio):
    # 遍历输入目录中的所有图像
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 获取图像的完整路径
            image_path = os.path.join(input_dir, filename)
            # 读取图像
            image = cv2.imread(image_path)
            # 计算新的分辨率（原始分辨率的downsample_ratio分之一）
            new_size = (image.shape[0] // downsample_ratio, image.shape[1] // downsample_ratio)
            # 打印输入分辨率和图像名称
            print(f"{filename} - Input resolution: {image.shape[0]}x{image.shape[1]}")
            # 下采样到新的分辨率
            resized_image = downsample_image(image_path, new_size)
            # 打印输出分辨率
            print(f"{filename} - Output resolution: {resized_image.shape[0]}x{resized_image.shape[1]}")
            # 新建目录
            os.makedirs(output_dir, exist_ok=True)
            # 在指定目录下保存下采样后的图像
            cv2.imwrite(os.path.join(output_dir, filename), resized_image)

def downsample_gt_2size(input_dir, output_dir, size1, size2):
    # 遍历输入目录中的所有图像
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 获取图像的完整路径
            image_path = os.path.join(input_dir, filename)
            # 读取图像
            image = cv2.imread(image_path)
            if filename == 'people.png':
                # 打印输入分辨率和图像名称
                print(f"{filename} - Input resolution: {image.shape[0]}x{image.shape[1]}")
                # 下采样到指定的分辨率
                resized_image = downsample_image(image_path, size1)
            else:
                # 打印输入分辨率和图像名称
                print(f"{filename} - Input resolution: {image.shape[0]}x{image.shape[1]}")
                # 下采样到指定的分辨率
                resized_image = downsample_image(image_path, size2)
            # 打印输出分辨率
            print(f"{filename} - Output resolution: {resized_image.shape[0]}x{resized_image.shape[1]}")
            # 新建目录
            os.makedirs(output_dir, exist_ok=True)
            # 在指定目录下保存下采样后的图像
            cv2.imwrite(os.path.join(output_dir, filename), resized_image)

def downsample_gt(input_dir, output_dir, size):
    # 遍历输入目录中的所有图像
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 获取图像的完整路径
            image_path = os.path.join(input_dir, filename)
            # 读取图像
            image = cv2.imread(image_path)
            # 打印输入分辨率和图像名称
            print(f"{filename} - Input resolution: {image.shape[0]}x{image.shape[1]}")
            # 下采样到指定的分辨率
            resized_image = downsample_image(image_path, size)
            # 打印输出分辨率
            print(f"{filename} - Output resolution: {resized_image.shape[0]}x{resized_image.shape[1]}")
            # 新建目录
            os.makedirs(output_dir, exist_ok=True)
            # 在指定目录下保存下采样后的图像
            cv2.imwrite(os.path.join(output_dir, filename), resized_image)

def generate_lmdb(folder_path, lmdb_path):
    img_path_list, keys = prepare_keys_test(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


if __name__ == "__main__":
    downsample_gt_2size('/workspace/Code/BasicSR/tests/data/1080P', '/workspace/Code/BasicSR/tests/data/gt', (480, 492),
                        (360, 240))
    downsample_lr('/workspace/Code/BasicSR/tests/data/gt', '/workspace/Code/BasicSR/tests/data/lq', 4)
    generate_lmdb('/workspace/Code/BasicSR/tests/data/gt', '/workspace/Code/BasicSR/tests/data/gt.lmdb')
    generate_lmdb('/workspace/Code/BasicSR/tests/data/lq', '/workspace/Code/BasicSR/tests/data/lq.lmdb')
