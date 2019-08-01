# CIFAR_pytorch

Train model use
```
python train.py --model lenet --dataset CIFAR10
```

## Requestments

Python 3.6
PyTorch 1.0.1

## CIFAR 10 Results

### 1. Noramal LeNet

| LeNet | CIFAR 10 |
| :----:| :------: |
| LeNet_max_pooling | 74.79% |
| LeNet_avg_pooling | 73.72% |

### 2. Noraml VGG and ResNet

| Baseline | CIFAR 10 |
| :----:| :------: |
| VGG16 | 91.21% |
| VGG16_bn | 92.43% |
| ResNet18 | 85.85% |
| ResNet34 | 86.15% |
| ResNet50 | 81.82% |

### 3. Use PreTraining 

| PreTraining | CIFAR 10 |
| :----:| :------: |
| VGG16 | 93.28% |
| VGG16_bn | 93.93% |
| ResNet18 | 80.84% |
| ResNet34 | 85.91% |
| ResNet50 | 90.17% |

### 4. Set conv1 stride equals 1 + Remove first max pooling + Pretaining model

| Remove Stride | CIFAR 10 |
| :----:| :------: |
| ResNet18 | 94.17% |
| ResNet34 | 94.33% |
| ResNet50 | 97.07% |

### 5. Use ResNext
| Remove Stride | CIFAR 10 |
| :----:| :------: |
| ResNeXt50_32x4d | 97.37% |
| ResNeXt101_32x8d | 96.72% | -- 60 epoch

### 6. decay learning rate
| Remove Stride | CIFAR 10 |
| :----:| :------: |
| ResNeXt50_32x4d | 97.48% |

### 5. Use Efficient to train model

| Efficientnet | CIFAR 10 |
| :----:| :------: |
| efficientnet-b0 | 93.96 |
| efficientnet-b1 | 94.03 |
| efficientnet-b2 | 94.32 |
| efficientnet-b3 |  |
| efficientnet-b4 |  |
| efficientnet-b5 |  |


| Remove Stride | CIFAR 10 |
| 保留意见 |  |
| ResNet18 | 92.97% |
| ResNet34 | 93.43% |
| ResNet50 | 92.74% |