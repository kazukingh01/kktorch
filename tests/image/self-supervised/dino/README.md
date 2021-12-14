# DINO
https://arxiv.org/abs/2104.14294

## How to use
- ViT
```
python train_vit.py
python predict_vit.py --weight weight.pth
```
- EfficientNet
```
python train_eff.py
python predict_eff.py --weight weight.pth
```

# TensorBoard
https://tensorboard.dev/experiment/tMNQqFP7RnO5tPQbCbI19Q/

# Weights
Weight is trained with [PASCALvoc2012D](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
- [ViT](https://drive.google.com/file/d/1D5EENJtb9uf4mm2P3JOIQNPQNDWp-X3P/view?usp=sharing)
- [EfficientNet](https://drive.google.com/file/d/1_MAHKyJsdT4RouqHuO_OtLdfZQJjP-tC/view?usp=sharing)

# Accuracy
- ViT<br>
5823 2105 acc: 0.3614975098746351
- EfficientNet<br>
5823 2655 acc: 0.4559505409582689

# KNN
The leftmost image is the test image, and the rightmost image is the train with k=10 neighborhood.
## ViT
![](./knn_vit.png)

## EfficientNet
![](./knn_eff.png)

# ViT attention matrix
![](./imgattnmap_vit.png)
