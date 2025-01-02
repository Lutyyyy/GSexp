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
## Reference
### Depth
- [SparseNeRF](https://arxiv.org/pdf/2303.16196)
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
