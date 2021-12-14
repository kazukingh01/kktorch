# SwAV
https://arxiv.org/abs/2006.09882

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
https://tensorboard.dev/experiment/5VtgkgYUROWcp9bRbAiRlw/

# Weights
Weight is trained with [PASCALvoc2012D](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
- [ViT](https://drive.google.com/file/d/18cnvLklH3fPuv74LFbnygZB4lpMR_TYT/view?usp=sharing)
- [EfficientNet](https://drive.google.com/file/d/1SvRMqmWoiikuGCuJ92pRVv90-UzEwl3g/view?usp=sharing)

# Accuracy
- ViT<br>
5823 2259 acc: 0.38794435857805254
- EfficientNet<br>
5823 2068 acc: 0.35514339687446334

# KNN
## ViT
![](./knn_vit.png)

## EfficientNet
![](./knn_eff.png)

# ViT attention matrix
![](./imgattnmap_vit.png)
