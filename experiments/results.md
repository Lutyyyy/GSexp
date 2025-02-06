# GSDepth exp
## Motivation
### Original GSmain sparse_view=4

```bash
Command: --sh_degree 2 --source_path ../../GSexp/data/mip360/kitchen --model_path output/sparse_gs/kitchen --images images --resolution 4 --white_background True --data_device cuda --eval False --max_num_splats 3000000 --iterations 30000 --position_lr_init 0.00016 --position_lr_final 1.6e-06 --position_lr_delay_mult 0.01 --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005 --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --lambda_silhouette 0.01 --densification_interval 100 --opacity_reset_interval 1000 --remove_outliers_interval 500 --densify_from_iter 500 --densify_until_iter 18000 --densify_grad_threshold 0.0002 --start_sample_pseudo 400000 --end_sample_pseudo 1000000 --sample_pseudo_interval 10 --random_background True --convert_SHs_python False --compute_cov3D_python False --debug False --ip 127.0.0.1 --port 6009 --debug_from -1 --detect_anomaly False --test_iterations [7000] --save_iterations [7000, 15000, 30000] --quiet False --checkpoint_iterations [] --start_checkpoint None --sparse_view_num 4 --use_mask True --init_pcd_name visual_hull_4 --transform_the_world False --mono_depth_weight 0.0005
```

| Dataset | Iter| L1 | PSNR |
| -------- | -------- | ------- | ------- |
| kitchen | 7000 | 0.018511730059981347 | 22.225626754760743  |
| kitchen | 10000| 0.0183987042467509 | 22.280342156546457 |
| kitchen | 30000| 0.02061660901776382 | 21.41695382254464 |

修改的codebase没有问题:
| Dataset | Iter| L1 | PSNR |
| -------- | -------- | ------- | ------- |
| kitchen | 07000 | 0.018309636786580086 | 22.35202909197126 |
| kitchen | 10000 | 0.020226402075162957 | 21.53639864240374 |
| kitchen | 30000 | 0.020669686182269026 | 21.466976819719587 |

- sparse_view 场景下非常容易过拟合导致性能下降

### Adjust depth_loss_weight sparse_view=4 

#### exp1
```bash
Command: --sh_degree 2 --source_path ../../GSexp/data/mip360/kitchen --model_path output/sparse_gs/kitchen --images images --resolution 4 --white_background True --data_device cuda --eval False --max_num_splats 3000000 --iterations 10000 --position_lr_init 0.00016 --position_lr_final 1.6e-06 --position_lr_delay_mult 0.01 --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005 --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --lambda_silhouette 0.01 --densification_interval 100 --opacity_reset_interval 1000 --remove_outliers_interval 500 --densify_from_iter 500 --densify_until_iter 6000 --densify_grad_threshold 0.0002 --start_sample_pseudo 400000 --end_sample_pseudo 1000000 --sample_pseudo_interval 10 --random_background True --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01 --convert_SHs_python False --compute_cov3D_python False --debug False --ip 127.0.0.1 --port 6009 --debug_from -1 --detect_anomaly False --test_iterations [7000] --save_iterations [7000, 15000, 10000] --quiet False --checkpoint_iterations [] --start_checkpoint None --sparse_view_num 4 --use_mask True --init_pcd_name visual_hull_4 --transform_the_world False --mono_depth_weight 0.0005
```

| Dataset | Iter| L1 | PSNR |
| -------- | -------- | ------- | ------- |
| kitchen | 7000 | 0.1502888383609908 | 11.048929868425642 |
| kitchen | 10000| 0.15082136584179742 | 10.905458940778459|
| kitchen | 30000| 0.1382204960499491 | 10.713459287370954 |
| kitchen | 50000| 0.12878245668751853 | 10.919726003919328 |
| kitchen | 70000| 0.12689002411706107 | 10.913635117667061 |
| kitchen | 90000| 0.12640090818916047 |10.906011063711983 |

#### exp2
- Adjust depth lr to a smaller interval

```bash
Command: --sh_degree 2 --source_path ../../GSexp/data/mip360/kitchen --model_path output/sparse_gs/kitchen --images images --resolution 4 --white_background True --data_device cuda --eval False --max_num_splats 3000000 --iterations 10000 --position_lr_init 0.00016 --position_lr_final 1.6e-06 --position_lr_delay_mult 0.01 --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005 --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --lambda_silhouette 0.01 --densification_interval 100 --opacity_reset_interval 1000 --remove_outliers_interval 500 --densify_from_iter 500 --densify_until_iter 6000 --densify_grad_threshold 0.0002 --start_sample_pseudo 400000 --end_sample_pseudo 1000000 --sample_pseudo_interval 10 --random_background True --depth_l1_weight_init 0.01 --depth_l1_weight_final 0.0001 --convert_SHs_python False --compute_cov3D_python False --debug False --ip 127.0.0.1 --port 6009 --debug_from -1 --detect_anomaly False --test_iterations [7000] --save_iterations [7000, 15000, 10000] --quiet False --checkpoint_iterations [] --start_checkpoint None --sparse_view_num 4 --use_mask True --init_pcd_name visual_hull_4 --transform_the_world False --mono_depth_weight 0.0005
```

| Dataset | Iter| L1 | PSNR |
| -------- | -------- | ------- | ------- |
| kitchen | 7000 | 0.02118007420961346 | 21.35457954406738|
| kitchen | 10000| 0.021811586352331297 | 21.1206787109375 |
| kitchen | 30000| 0.022657590199794086 | 20.880568858555385 |
| kitchen | 50000|0.022733117480363163 | 20.84707314627511 |

- 在后期depth的loss基本不更新了 导致没法学习到很好的几何结构 所以最后PSNR和L1结果都不够好


```bash
Command: --sh_degree 2 --source_path ../../GSexp/data/mip360/kitchen --model_path output/sparse_gs/kitchen --images images --resolution 4 --white_background True --data_device cuda --eval False --max_num_splats 3000000 --iterations 10000 --position_lr_init 0.00016 --position_lr_final 1.6e-06 --position_lr_delay_mult 0.01 --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005 --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --lambda_silhouette 0.01 --densification_interval 100 --opacity_reset_interval 1000 --remove_outliers_interval 500 --densify_from_iter 500 --densify_until_iter 6000 --densify_grad_threshold 0.0002 --start_sample_pseudo 400000 --end_sample_pseudo 1000000 --sample_pseudo_interval 10 --random_background True --depth_l1_weight_init 0.01 --depth_l1_weight_final 0.0005 --convert_SHs_python False --compute_cov3D_python False --debug False --ip 127.0.0.1 --port 6009 --debug_from -1 --detect_anomaly False --test_iterations [7000] --save_iterations [7000, 15000, 10000] --quiet False --checkpoint_iterations [] --start_checkpoint None --sparse_view_num 4 --use_mask True --init_pcd_name visual_hull_4 --transform_the_world False --mono_depth_weight 0.0005
```

| Dataset | Iter| L1 | PSNR |
| -------- | -------- | ------- | ------- |
| kitchen | 7000 | 0.021398553172392504 | 21.25406870160784 |
| kitchen | 10000| 0.021834435925952026 | 21.07646255493164 |
| kitchen | 30000| 0.02239498317773853 | 20.889677592686244 |


- 使其收敛到原论文的depth weight. 依然没有很多变化 是否是因为前期depth weight太高使得优化陷入局部最小值后面出不来了
## Method
### Method1
- CVPR 2020 best paper 的去噪思想来修正 depth shift and scale 和 depth regularization 结合
### Method2
- Depth completetion Method
### Method3
- Scale invariant method to fix depth supervision signal and combine with opacity augmentation
### Method4
- 前期可以进行depth prior，后期可以用上自监督的方式进行训练。二者结合效果更好。因为pretrained的depth prior前期可以加速收敛，到后期反而会因为细节的累积而将optimization引导至local minima。进而导致随着input views的增加效果却并不明显。自监督用MVS计算depth的方式在low texture区域效果很差，此时如果使用depth prior反而会很好。
### Method5
- MVS-based的方法
    - Google那篇经典论文
    - disparity约束
    - 各种warp
        - warp过程中可以分forward和backward mapping，然后根据不同的插值方法，对高斯球直接进行监督，而不是proj.到图片再进行颜色监督[MVPGS](https://arxiv.org/pdf/2409.14316)。投影过程中/反投影过程中，可以用KNN颜色监督3D点的那一块区域，而不是2D颜色图片。
    - 各种MVSTransformer
    - 
## Reference
### Depth/Normal
- [SparseNeRF](https://arxiv.org/pdf/2303.16196)
- [SolidGS](https://arxiv.org/pdf/2412.15400)
    - LLFF(NONE)
    - DTU
        - 3: 21.32
    - T&T(NONE)
### Mesh
- [SparseNeuS](https://arxiv.org/pdf/2206.05737)
- 
### Graphics
- [Binocular-Guided 3D Gaussian Splatting with View Consistency for Sparse View Synthesis](https://arxiv.org/pdf/2410.18822)
    - LLFF(1/8) **ALL SOTA**
        - 3: 21.44, 0.751, 0.168 
        - 6: 24.87, 0.845, 0.106 
        - 9: 26.17, 0.877, 0.090
    - DTU(1/4)
        - 3: 20.71, 0.862, 0.111 
        - 6: 24.31, 0.917, 0.073 
        - 9: 26.70, 0.947, 0.052
    - NeRF Blender Synthetic dataset (Blender) (1/2)
        - 8: 24.71, 0.872, 0.101
    - 实验细节详细
- [GeCoNeRF](https://arxiv.org/pdf/2301.10941)
- [GeoAug](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770326.pdf)
- [SCGaussian](https://arxiv.org/pdf/2411.03637)
    - LLFF(1/8)
        - 3: 20.77, 0.705, 0.218 (*高斯球几何位置约束*)
    - LLFF(1/4)
        - 3: 20.09, 0.679, 0.252
    - DTU
        - 3: 20.56, 0.864, 0.122
    - NeRF Blender Synthetic dataset (Blender) (1/2)  **SOTA**
        - 8: 25.618, 0.894, 0.086
    - IBRNet
        - 3: 21.59, 0.731, 0.233
    - T&T(960 ×540)
        - 3: 22.17, 0.752, 0.257
        - 4: 25.00
        - 5: 26.00
        - 6: 26.95, 0.869, 0.149
- [FewViewGS](https://arxiv.org/pdf/2411.02229)
    - MVS(ROMA匹配)+color约束;10000 iterations(2000+7500+500)
    - DTU(1/4)
        - 3: 19.74, 0.861, 0.127
        - 6: 24.33, 0.920, 0.069
        - 9: 27.31, 0.953, 0.041
    - LLFF(1/8)
        - 3: 20.54, 0.693, 0.214
        - 6: 24.35, 0.826, 0.126
        - 9: 25.90, 0.868, 0.095
    - Blender
        - 8: 25.550, 0.886, 0.092
- [FatesGS](https://arxiv.org/pdf/2501.04628)
    - Smooth&Ranking loss提depth精确度效果明显。特别是ranking+multiscal feature matching的提升更明显。单独使用smooth+ranking效果会下降
    - DTU(1/4)
        - 3: 21.80, 0.904, 0.077
- [MVPGS](https://arxiv.org/pdf/2409.14316)
    - **MVSFormer初始化**+**Foward warping**+Depth监督
    - LLFF(1/8)
        - 2: 18.53, 0.607, 0.280
        - 3: 20.54, 0.727, 0.194
        - 4: 21.28, 0.750, 0.180 
        - 5: 22.18, 0.773, 0.164
    - LLFF(1/4)
        - 3: 19.91, 0.696, 0.229
    - DTU(1/4)
        - DTU数据集一般都只warp前景物体，用mask把背景遮掉
        - 2: 17.55, 0.823, 0.147
        - 3: 20.65, 0.877, 0.099
        - 4: 21.57, 0.887, 0.091
        - 5: 22.87, 0.899, 0.080
    - DTU(1/2)
        - 3: 20.24, 0.858, 0.124
    - NVS-RGBD
        - 【ZED2】3: 26.62, 0.841, 0.185
        - 【Kinect】3: 27.04, 0.887, 0.151
    - T&T
        - 3: 25.57, 0.846, 0.139
