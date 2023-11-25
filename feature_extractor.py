
import torch
import torchvision
import torchvision.transforms as transforms

# 定义一个类，用于封装resnet50模型和图像特征提取的方法
class FeatureExtractor:

    # 初始化类，加载预训练的resnet50模型
    def __init__(self):
        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        # 设置模型为评估模式，不进行梯度更新
        self.model.eval()
        # 定义一个变换，用于将图像转换为适合模型输入的张量
        self.transform = transforms.Compose([
            transforms.Resize(256), # 调整图像大小为256x256
            transforms.CenterCrop(224), # 从中心裁剪出224x224的区域
            transforms.ToTensor(), # 将图像转换为张量，范围为[0, 1]
            transforms.Normalize( # 标准化张量，使用ImageNet的均值和标准差
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    # 定义一个方法，用于提取图像的特征
    def extract(self, image):
        # 将图像变换为适合模型输入的张量
        tensor = self.transform(image)
        # 在张量上增加一个维度，表示批量大小为1
        tensor = tensor.unsqueeze(0)
        # 使用模型提取图像的特征，得到一个2048维的向量
        feature = self.model(tensor)
        # 将特征向量转换为一维的张量
        feature = feature.view(-1)
        # 返回特征向量
        return feature.detach().numpy()
