# lent
# python train.py --model lenet

# train 稿后面去
python train.py --model resnext50_32x4d --pretrain True
# python train.py --model resnext101_32x8d --pretrain True

# normal
# python train.py --model vgg16
# python train.py --model vgg16_bn
# python train.py --model resnet18
# python train.py --model resnet34
# python train.py --model resnet50

# resnet 两个地方再测试以下，数据有点不美丽
# pretrain
# python train.py --model vgg16 --pretrain True
# python train.py --model vgg16_bn --pretrain True
# python train.py --model resnet18 --pretrain True
# python train.py --model resnet34 --pretrain True
# python train.py --model resnet50 --pretrain True

# reomve stride
# python train.py --model resnetv2_18 --pretrain True
# python train.py --model resnetv2_34 --pretrain True
# python train.py --model resnetv2_50 --pretrain True

# python train.py --model efficientnet-b0 --pretrain True
# python train.py --model efficientnet-b1 --pretrain True
# python train.py --model efficientnet-b2 --pretrain True
# python train.py --model efficientnet-b3 --pretrain True
# python train.py --model efficientnet-b4 --pretrain True
# python train.py --model efficientnet-b5 --pretrain True
