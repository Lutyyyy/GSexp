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
07000: 0.018309636786580086 | 22.35202909197126
10000: 0.020226402075162957 | 21.53639864240374
30000: 0.020669686182269026 | 21.466976819719587

- sparse_view 场景下非常容易过拟合导致性能下降

### Adjust depth_loss_weight sparse_view=4 

#### exp1
```bash
Command: --sh_degree 2 --source_path ../../GSexp/data/mip360/kitchen --model_path output/sparse_gs/kitchen --images images --resolution 4 --white_background True --data_device cuda --eval False --max_num_splats 3000000 --iterations 10000 --position_lr_init 0.00016 --position_lr_final 1.6e-06 --position_lr_delay_mult 0.01 --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005 --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --lambda_silhouette 0.01 --densification_interval 100 --opacity_reset_interval 1000 --remove_outliers_interval 500 --densify_from_iter 500 --densify_until_iter 6000 --densify_grad_threshold 0.0002 --start_sample_pseudo 400000 --end_sample_pseudo 1000000 --sample_pseudo_interval 10 --random_background True --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01 --convert_SHs_python False --compute_cov3D_python False --debug False --ip 127.0.0.1 --port 6009 --debug_from -1 --detect_anomaly False --test_iterations [7000] --save_iterations [7000, 15000, 10000] --quiet False --checkpoint_iterations [] --start_checkpoint None --sparse_view_num 4 --use_mask True --init_pcd_name visual_hull_4 --transform_the_world False --mono_depth_weight 0.0005
```

| Dataset | Iter| L1 | PSNR |
| -------- | -------- | ------- | ------- |
| kitchen | 7000 | 0.06686059768710817 | 15.75955502646309 |
| kitchen | 10000| 0.047505941987037656| 17.50093571799142 |

#### exp2
- Adjust depth lr to a smaller interval

```bash
Command: --sh_degree 2 --source_path ../../GSexp/data/mip360/kitchen --model_path output/sparse_gs/kitchen --images images --resolution 4 --white_background True --data_device cuda --eval False --max_num_splats 3000000 --iterations 10000 --position_lr_init 0.00016 --position_lr_final 1.6e-06 --position_lr_delay_mult 0.01 --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005 --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --lambda_silhouette 0.01 --densification_interval 100 --opacity_reset_interval 1000 --remove_outliers_interval 500 --densify_from_iter 500 --densify_until_iter 6000 --densify_grad_threshold 0.0002 --start_sample_pseudo 400000 --end_sample_pseudo 1000000 --sample_pseudo_interval 10 --random_background True --depth_l1_weight_init 0.01 --depth_l1_weight_final 0.0001 --convert_SHs_python False --compute_cov3D_python False --debug False --ip 127.0.0.1 --port 6009 --debug_from -1 --detect_anomaly False --test_iterations [7000] --save_iterations [7000, 15000, 10000] --quiet False --checkpoint_iterations [] --start_checkpoint None --sparse_view_num 4 --use_mask True --init_pcd_name visual_hull_4 --transform_the_world False --mono_depth_weight 0.0005
```

| Dataset | Iter| L1 | PSNR |
| -------- | -------- | ------- | ------- |
| kitchen | 7000 | 0.0194312270996826 | 21.903804070608956 |
| kitchen | 10000|  0.01950149397764887 | 21.873552867344447 |

- 在后期depth的loss基本不更新了 导致没法学习到很好的几何结构 所以最后PSNR和L1效果都不够好


```bash
Command: --sh_degree 2 --source_path ../../GSexp/data/mip360/kitchen --model_path output/sparse_gs/kitchen --images images --resolution 4 --white_background True --data_device cuda --eval False --max_num_splats 3000000 --iterations 10000 --position_lr_init 0.00016 --position_lr_final 1.6e-06 --position_lr_delay_mult 0.01 --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.05 --scaling_lr 0.005 --rotation_lr 0.001 --percent_dense 0.01 --lambda_dssim 0.2 --lambda_silhouette 0.01 --densification_interval 100 --opacity_reset_interval 1000 --remove_outliers_interval 500 --densify_from_iter 500 --densify_until_iter 6000 --densify_grad_threshold 0.0002 --start_sample_pseudo 400000 --end_sample_pseudo 1000000 --sample_pseudo_interval 10 --random_background True --depth_l1_weight_init 0.01 --depth_l1_weight_final 0.0005 --convert_SHs_python False --compute_cov3D_python False --debug False --ip 127.0.0.1 --port 6009 --debug_from -1 --detect_anomaly False --test_iterations [7000] --save_iterations [7000, 15000, 10000] --quiet False --checkpoint_iterations [] --start_checkpoint None --sparse_view_num 4 --use_mask True --init_pcd_name visual_hull_4 --transform_the_world False --mono_depth_weight 0.0005
```

| Dataset | Iter| L1 | PSNR |
| -------- | -------- | ------- | ------- |
| kitchen | 7000 | 0.01980680961694036 | 21.808598164149693 |
| kitchen | 10000| 0.019869494837309633 | 21.786050306047713 |

- 使其收敛到原论文的depth weight. 依然没有很多变化 是否是因为前期depth weight太高使得优化陷入局部最小值后面出不来了
