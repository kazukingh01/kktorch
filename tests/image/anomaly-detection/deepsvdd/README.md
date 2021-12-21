# Deep SVDD
http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf

## How to use
- training MNIST All data
```
python train_mnist_all.py
```
- training MNIST except anomaly data<br>
![](./explain1.png) 
```
python train_mnist_ano.py
```

# Rsult

## MNIST All data
Scatter plots and radius R plots for training with the latent variable dimension set to 2.<br>
![](./mnist_train_z.png)

distance > R * 1.25 samples is here.<br>
![](./mnist_train_anomaly.png)

## MNIST except anomaly data
The latent variable dimension set to 16 and Visualization with t-SNE.
![](./mnist_tsne_0_anomaly.png)
![](./mnist_tsne_1_anomaly.png)
![](./mnist_tsne_2_anomaly.png)
![](./mnist_tsne_3_anomaly.png)
![](./mnist_tsne_4_anomaly.png)
![](./mnist_tsne_5_anomaly.png)
![](./mnist_tsne_6_anomaly.png)
![](./mnist_tsne_7_anomaly.png)
![](./mnist_tsne_8_anomaly.png)
![](./mnist_tsne_9_anomaly.png)
