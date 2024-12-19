#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Modified according to GSexp/main

import os
import csv
import torch
from random import randint
from typing import Optional
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
import json

from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.util_print import STR_DEBUG, STR_STAGE
from utils.util_logger import create_logger
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(args, dataset, opt, pipe, testing_iterations: list, saving_iterations: list, checkpoint_iterations, checkpoint, debug_from):

    # GSObj 没有SparseAdam
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer, logger = prepare_output_and_logger(dataset)
    logger.log_command(args)
    gaussians = GaussianModel(dataset.sh_degree, logger, opt.optimizer_type)   #NOTE 该初始化函数仅仅是创建一堆空的GS属性张量 没有具体的赋值
    #NOTE Scene的初始化函数中已经包括是否选择恢复点云数据来创建高斯场景 但不包括一些累积的梯度、优化器状态、denom、spatial_lr_scale等
    scene = Scene(dataset, gaussians) # extra_opts仅在Scene读稀疏数据集用上的
    gaussians.training_setup(opt)
    if checkpoint:  #NOTE 具体 GS 的属性初始化在 Scene 初始化的时候实现了 这一步主要是为了恢复梯度以及信息 也就是上面 Scene 没包括的信息 应该会自动覆盖原先初始化好的 GS 属性
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)  # restore函数包括训练状态全部恢复出来  对应的保存函数是gaussians.capture()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # 设置背景颜色 根据数据集是否有白色背景来选择背景的颜色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转化为 Tensor，并移到 GPU 上

    # 创建两个 CUDA 事件，用于测量迭代时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)  # exponential decay learning rate

    viewpoint_stack = scene.getTrainCameras().copy()  # 返回一串训练相机
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    results = {}

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:  # 检查 GUI 是否连接，如果连接则接收 GUI 发送的消息
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record() # type: ignore
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera 随机选择一个训练相机 也就是说每次forward-backward都只有一张图片参与
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        # Render 一定步数之后开始渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        input_dict, layer_input_dict = render_pkg["input"], render_pkg["layer_input"]
        # visibility_filter = radii > 0

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = None
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        training_psnr = psnr(image, gt_image).mean().double()

        if tb_writer is not None:
            tb_writer.add_scalar('training_loss/l1_loss', Ll1, iteration)
            tb_writer.add_scalar('training_loss/ssim_loss', ssim_value, iteration)
            tb_writer.add_scalar('training_metrics/psnr', training_psnr, iteration)
            tb_writer.add_scalar('training_loss/depth_loss', Ll1depth_pure, iteration)

        layer_grads = {}
        def save_grad(name):
            def hook(grad):
                layer_grads[name] = grad
            return hook
        for k, v in layer_input_dict.items():
            # print(STR_ERROR, k, v.grad_fn)  # add hook for recording the middle nodes' gradient
            v.register_hook(save_grad(k))

        loss.backward()  #NOTE 在执行这一步之前 所有的求梯度的操作比如.grad操作都是不可access的 因为这一步backward就是回传所有梯度

        input_grads = {}
        for k, v in input_dict.items():
            input_grads[k] = v.grad

        param_report(input_dict, layer_input_dict, tb_writer, iteration)
        grads_report(input_grads, layer_grads, tb_writer, iteration)
        # if not (viewspace_point_tensor.grad is means_2d.grad): print(STR_ERROR, viewspace_point_tensor.grad == means_2d.grad, viewspace_point_tensor.grad is means_2d.grad)
        # print(STR_WARNING, means_2d.shape, means_2d.grad.shape, means_3d.shape, means_3d.grad.shape)
        iter_end.record()

        with torch.no_grad():  # 在每个训练轮次结束之后 不再需要梯度信息 开始执行不需要梯度信息的操作
            # 记录损失的指数移动平均值，并定期更新进度条
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            num_gauss = len(gaussians._xyz)
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "EMA_Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "number_of_GS": f"{num_gauss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # 记录densify之前的状态 包括高斯球的数量和损失值 以及测试时候才记录的损失值和高斯球不透明度
            results.update({
                f'{iteration}': training_report(tb_writer, iteration, Ll1, loss, l1_loss, Ll1depth_pure, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE), dataset.train_test_exp)
            })

            if (iteration in saving_iterations):  # 达到保存次数则保存场景
                print(STR_DEBUG, "[ITER {}] Saving Gaussians in {}".format(iteration, os.path.join(scene.model_path, "point_cloud/iteration_{}/point_cloud.ply".format(iteration))))
                scene.save(iteration)  # scene的保存仅仅是保存下来ply文件 没有高斯球的训练状态和梯度信息

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # 将每个像素位置上的最大半径记录在 max_radii2D 中。这是为了密集化时进行修剪（pruning）操作时的参考
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 将与密集化相关的统计信息添加到 gaussians 模型中，包括视图空间点和可见性过滤器
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                # 每隔一定迭代次数或在白色背景数据集上的指定迭代次数时重置不透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    # print(STR_ERROR, f'Resetting Opacity in {iteration}')
                    logger.debug(f'Resetting Opacity in {iteration}')

                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 保存检查点
            if (iteration in checkpoint_iterations):
                print(STR_STAGE, "[ITER {}] Saving Checkpoint in {}".format(iteration, scene.model_path + "/ckpt" + str(iteration) + ".pth"))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt" + str(iteration) + ".pth")  # capture函数保存了训练状态 对应的恢复函数是restore

    # saving json result after training
    with open(args.model_path + '/training_results.json', 'w') as fp:
        print(STR_DEBUG, f"saving results in {args.model_path + '/training_results.json'}")
        json.dump(results, fp, indent=True)

    # saving csv result after training
    fieldnames = set()
    for data in results.values():
        fieldnames.update(data.keys())
    fieldnames = ["epoch"] + list(fieldnames)

    # write results in CSV file
    with open(os.path.join(args.model_path, "training_results.csv"), mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头

        for epoch, data in results.items():
            # 将 epoch 作为字典中的一项，以便写入 CSV 文件
            row = {"epoch": epoch, **data}
            writer.writerow(row)

    logger.warning(f"Training finished. Results saved in {args.model_path + '/training_results.json/csv'}")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
            args.model_path = os.path.join("./output/", unique_str)
        else:
            unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    # Create Logger
    logger = create_logger(logpath=args.model_path)
    return tb_writer, logger

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, depth_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    result = {
        'train_loss_patches/l1_loss': Ll1.item(),  # 和cal_loss函数中画图的数值一样 因为就是直接传进去的
        'train_loss_patches/depth_loss': depth_loss, # 和cal_loss函数中画图的数值一样 因为就是直接传进去的
        'train_loss_patches/total_loss': loss.item(),
        'iter_time': elapsed,
        'total_points': scene.gaussians.get_xyz.shape[0],
    }
    if tb_writer:
        for k, v in result.items():
            tb_writer.add_scalar(k, v, iteration)
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        # tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)
        # tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)  # 训练过程中一样记录高斯球数量

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, depth_test_loss = 0.0, 0.0, 0.0  # 计算所有35个test相机和隔5个train相机的总metric
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # 测试模型抗遮挡能力 所以需要遮盖一部分来计算损失
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    # draw the image 从train_camera和test_camera随机挑渲染图和GT图出来展示
                    if tb_writer and (idx < 5):
                        #? 只返回前五个相机的图片?
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:  # 随便挑一个轮次展示GT即可 防止重复展示GT
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    depth_test_loss += depth_loss.mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                depth_test_loss /= len(config['cameras'])
                # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                result.update({
                    'Evaluation_Camera': config['name'],
                    'L1_test': l1_test.item(),
                    'PSNR_test': psnr_test.item(),
                    'Depth_test_loss': depth_test_loss.item(),
                })
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '_camera/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '_camera/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '_camera/loss_viewpoint - depth', depth_test_loss, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return result


def grads_report(input_grads: dict, layer_grads: dict, tb_writer: Optional[SummaryWriter], iteration=0):
    # print(STR_VERBOSE, iteration)
    # for k, v in input_dict.items():
        # if v.grad is not None: tb_writer.add_scalar(f'training_grads_{k}', v.grad.mean().double(), iteration)
        # if v.grad is not None: print(k, v.grad.shape)

    for k, v in input_grads.items():
        tb_writer.add_scalar(f'training_input_grads/{k}_grads', v.mean().double(), iteration)
   
    for k, v in layer_grads.items():
        tb_writer.add_scalar(f'training_layer_grads/{k}_grads', v.mean().double(), iteration)


def param_report(input_dict: dict, layer_input_dict: dict, tb_writer: Optional[SummaryWriter], iteration=0):
    for k, v in input_dict.items():
        tb_writer.add_scalar(f'training_params/{k}', v.mean().double(), iteration)

    for k, v in layer_input_dict.items():
        tb_writer.add_scalar(f'training_layer_params/{k}', v.mean().double(), iteration)
   


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 用于保存训练状态 包括高斯球的梯度信息
    parser.add_argument("--start_checkpoint", type=str, default = None)  # 用于恢复高斯球的训练状态
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)  # 初始化一个GUI
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # Pytorch检测梯度异常
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
