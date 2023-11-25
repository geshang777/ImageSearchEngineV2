from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import os
import pickle
from scipy.spatial import cKDTree


if __name__ == '__main__':
    fe = FeatureExtractor()
    # 设置文件夹路径和文本文件路径
    folder_path = './static/imagenet-mini/train'
    txt_file_path = 'jpg_files_paths.txt'
    cls= "n04542943"

    des = []
    inds = []
    # 打开文本文件并读取所有行
    with open(txt_file_path, 'r') as f:
        for line in f:
            # 移除每行末尾的换行符
            relative_image_path = line.strip()

            # 构建完整的图片文件路径
            full_image_path = os.path.join(folder_path, relative_image_path)


            if relative_image_path.split("/")[0]!=cls:
                des = np.vstack(des)
                kd_tree = cKDTree(des)

                with open(os.path.join("./static/imagenetkdt",cls + '_kd_tree.pkl', 'wb')) as f:
                    pickle.dump((kd_tree, inds), f)
                    print(f"{cls + '_kd_tree.pkl'} is done")
                    des = []
                    inds.clear()
                cls = relative_image_path.split("/")[0]
            print(full_image_path)
            if Image.open(full_image_path).mode == 'L':
                continue
            feature = fe.extract(image=Image.open(full_image_path))
            des.append(feature)
            inds.append(relative_image_path)
            if des == []:
                continue








    # # 定义源目录和目标目录
    # source_dir = './static/images'
    # target_dir = './static/feature'
    #
    # # 确保目标目录存在
    # os.makedirs(target_dir, exist_ok=True)
    #
    # # 遍历源目录中的所有文件夹和文件
    # for subdir, dirs, files in os.walk(source_dir):
    #     des = []
    #     inds = []
    #     for file in files:
    #         # 构建完整的文件路径
    #         file_path = os.path.join(subdir, file)
    #
    #         # 检查是否为图片文件（这里假设图片文件以.jpg, .jpeg, .png, .gif等结尾）
    #         if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
    #             # 构建目标文件夹路径，它应该镜像源文件夹的结构
    #             destination_folder = os.path.join(target_dir, os.path.relpath(subdir, source_dir))
    #             os.makedirs(destination_folder, exist_ok=True)  # 确保目标文件夹存在
    #             feature = fe.extract(image=Image.open(file_path))
    #             des.append(feature)
    #             relative_path = os.path.relpath(subdir, './static/feature/')
    #             feature_path = Path("./static/images") / relative_path / (
    #                     file[:-4] + ".jpg")
    #             relative_path = os.path.relpath(subdir, source_dir)
    #             inds.append(feature)
    #             print(feature_path)
    #     if des == []:
    #         continue
    #     des = np.vstack(des)
    #     kd_tree = cKDTree(des)
    #     with open(os.path.join(subdir, 'kd_tree.pkl'), 'wb') as f:
    #         pickle.dump((kd_tree, inds), f)
    #             # np.save(feature_path, feature)
