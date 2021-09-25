# ResNet18 + RandomCrop256 when training
#python main.py --crop_train 256 --model resnet --block basicblock --layers [2,2,2,2]

# WaDIQaM-FR + No crop for dataset
# python main.py --multi_crop_size 32 --model wadiqam --weighted_average True

# DISTS + Resize256 for dataset
# python main.py --resize 256 --model dists --imagenet_pretrain ./pth/vgg16-397923af.pth --lr 0.01 --decay 20-40-60 --gamma 0.5

# ViT + 32x32 patches
python main.py --resize 256 --model vit --vit_patch_size 32 --ts_blk 6 --att_head 16 --mlp_dim 2048 --decay 20-40-60 --gamma 0.5
