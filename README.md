# Metal quality classification

利用显微镜成像的金相图片，评估定级结果（晶粒度）的分类器。类别从6.5开始到13.0，每0.5为一个级别，共为13类。
使用PyTorch作为framework，使用resnet152进行分类。
### 清理数据
有较多重复图片，基于md5去重

### 预处理
Crop:     将图片从(1376,1104)使用center crop变为(1104,1104)，后resize为(224,224)。

RGB normalize：标准imagenet normalization

黑白图片normalize：mean = 0.5，std = 0.5
### 数据增强
```python
albumentations.Compose([
              albumentations.HorizontalFlip(p=0.5),
              albumentations.VerticalFlip(p=0.5),
              albumentations.Transpose(p=0.2),
              albumentations.ElasticTransform(alpha=2000,sigma=100,alpha_affine=1,p=0.2),
              albumentations.MotionBlur(blur_limit=5,p=0.1),
              albumentations.Rotate(limit=(-180,180),p = 0.1, border_mode=cv2.BORDER_WRAP)])
```
### 损失函数
使用基于Focal loss的损失函数。
相似晶粒度之间经常有相同特征，故而使用FocalLoss(l) + w * (FocalLoss(l+0.5) + FocalLoss(l-0.5))作为损失函数，l为正确标签，w为超参数。

### 模型融合
比赛禁止使用超过两个模型的ensemble。
本repo使用黑白图片训练的图片与resnet152训练的图片进行融合，output经过softmax归一后相加，权重相同(都是0.5)。

### TTA
无


### 例图
类别 8.0
![8.0](https://github.com/Wooooprime/Metal-quality-classification/blob/main/Sample%20data/100-101-8.0-500x.jpg)
类别 10.0
![10.0](https://github.com/Wooooprime/Metal-quality-classification/blob/main/Sample%20data/100-100-10.0-500x.jpg)

