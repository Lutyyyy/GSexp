### Original gaussian-splatting
| Dataset | Iter| L1 | PSNR |
| -------- | -------- | ------- | ------- |
| truck | 7000 | 0.03509791493415833 | 24.375564193725587|
| truck | 30000 | 0.02342766933143139 | 27.41238670349121 |

### My gaussian-splatting
Command: --sh_degree 3 --source_path ../datas/tandt/truck --model_path output/tandt/truck --images images --depths  --resolution -1 --white_background False --train_test_exp False --data_device cuda --eval False --iterations 30000 --position_lr_init 0.00016 --position_lr_final 1.6e-06 --position_lr_delay_mult 0.01 --position_lr_max_steps 30000 --feature_lr 0.0025 --opacity_lr 0.025 --scaling_lr 0.005 --rotation_lr 0.001 --exposure_lr_init 0.01 --exposure_lr_final 0.001 --exposure_lr_delay_steps 0 --exposure_lr_delay_mult 0.0 --percent_dense 0.01 --lambda_dssim 0.2 --densification_interval 100 --opacity_reset_interval 3000 --densify_from_iter 500 --densify_until_iter 15000 --densify_grad_threshold 0.0002 --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01 --random_background False --optimizer_type default --convert_SHs_python False --compute_cov3D_python False --debug False --antialiasing False --ip 127.0.0.1 --port 6009 --debug_from -1 --detect_anomaly False --test_iterations [7000, 30000] --save_iterations [7000, 30000, 30000] --quiet False --disable_viewer False --checkpoint_iterations [] --start_checkpoint None
| Dataset | Iter| L1 | PSNR |
| -------- | -------- | ------- | ------- |
| truck | 7000 |0.0353591300547123 | 24.345150756835938 |
| truck | 30000 | 0.023872000351548198 | 27.406385040283205 |