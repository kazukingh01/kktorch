# Anomary detection with AutoEncoder

## Usage
- training
```
python train.py --type ssim ( or mse, msenorm )
```

-inference
```
python train.py --type ssim ( or mse, msenorm ) --weight ./XXXXXX/model_YYY.pth
```

# TensorBoard
https://tensorboard.dev/experiment/Hg9fvXbRQDCj9NSrPO0QAQ/

# Reconstruction Image
Left: GT, Center: Prediction, Right: Difference( GT - Prediction )

- SSIM

![](./rec_diff_ssim_test.png)

- MSE without Normalization

![](./rec_diff_mse_test.png)

- MSE with Normalization

![](./rec_diff_msenorm_test.png)

# Gray scale pixel value Histgrum
- SSIM

![](./hist_ssim.png)

- MSE without Normalization

![](./hist_mse.png)

- MSE with Normalization

![](./hist_msenorm.png)

# Result
Determine that an image with a pixel value of 65 or more for 5 counts or more is anomaly ( label = 1 ).

- SSIM

| Pred / GT | 0 | 1 |
|---|---|---|
|0|23|86|
|1|0|23|

- MSE without Normalization

| Pred / GT | 0 | 1 |
|---|---|---|
|0|22|84|
|1|1|25|

- MSE with Normalization

| Pred / GT | 0 | 1 |
|---|---|---|
| 0 | 23 | 84 |
| 1 | 0  | 25 |
