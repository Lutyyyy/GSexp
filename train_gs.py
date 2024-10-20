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

# Coarse 3DGS training code

import os
import sys
import uuid
import json
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.functional.regression import pearson_corrcoef
from torch.utils.tensorboard.writer import SummaryWriter
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, monodisp
from utils.util_print import STR_DEBUG, STR_VERBOSE, STR_STAGE, STR_WARNING, STR_ERROR


TENSORBOARD_FOUND = True


def training(args, dataset, opt, pipe, testing_iterations: list, saving_iterations: list, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)  #NOTE 该初始化函数仅仅是创建一堆空的GS属性张量 没有具体的赋值
    #NOTE Scene的初始化函数中已经包括是否选择恢复点云数据来创建高斯场景 但不包括一些累积的梯度、优化器状态、denom、spatial_lr_scale等
    scene = Scene(dataset, gaussians, extra_opts=args)
    gaussians.training_setup(opt)
    if checkpoint:  #NOTE 具体 GS 的属性初始化在 Scene 初始化的时候实现了 这一步主要是为了恢复梯度以及信息 也就是上面 Scene 没包括的信息 应该会自动覆盖原先初始化好的 GS 属性
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)  # restore函数包括训练状态全部恢复出来  对应的保存函数是gaussians.capture()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # 设置背景颜色 根据数据集是否有白色背景来选择背景的颜色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转化为 Tensor，并移到 GPU 上

    # 创建两个 CUDA 事件，用于测量迭代时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack, augview_stack = None, None  # TODO: how to use augview
    ema_loss_for_log = 0.0
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
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
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
            gaussians.oneupSHdegree()  # 每1000次迭代增加球谐函数系数

        # Pick a random Camera 随机选择一个训练相机 也就是说每次forward-backward都只有一张图片参与
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render 一定步数之后开始渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        input_dict, layer_input_dict = render_pkg["input"], render_pkg["layer_input"]
        # visibility_filter = radii > 0

        # Loss
        loss, Ll1 , depth_loss = cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, tb_writer=tb_writer, iteration=iteration)  # 返回初始的l1_loss和总的loss

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
            num_gauss = len(gaussians._xyz)
            if iteration % 10 == 0:
                progress_bar.set_postfix({'Loss': f"{ema_loss_for_log:.{7}f}",  'number_of_GS': f"{num_gauss}"})  # 设定后缀
                progress_bar.update(10)
            if iteration == opt.iterations:  # default 10000
                progress_bar.close()

            # Log and save
            results.update({
                f'{iteration}': training_report(tb_writer, iteration, Ll1, loss, l1_loss, depth_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            })

            if (iteration in saving_iterations):  # 达到保存次数则保存场景
                print(STR_DEBUG, "[ITER {}] Saving Gaussians in {}".format(iteration, os.path.join(scene.model_path, "point_cloud/iteration_{}/point_cloud.ply".format(iteration))))
                scene.save(iteration)  # scene的保存仅仅是保存下来ply文件 没有高斯球的训练状态和梯度信息

            # Densification  在一定的迭代次数内进行密集化处理 本文为了节省内存调低成60% 也就是60%后不再densitify操作
            if iteration < opt.densify_until_iter and num_gauss < opt.max_num_splats:  # TODO ablation study
                # 在达到迭代次数和高斯数量之前都执行
                # Keep track of max radii in image-space for pruning
                # 将每个像素位置上的最大半径记录在 max_radii2D 中。这是为了密集化时进行修剪（pruning）操作时的参考
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 将与密集化相关的统计信息添加到 gaussians 模型中，包括视图空间点和可见性过滤器
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:  # 在指定的迭代次数之后，每隔一定的迭代间隔进行以下密集化操作
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None  # 设置密集化的阈值
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)  # 执行密集化和修剪操作
                
                # 每隔一定迭代次数或在白色背景数据集上的指定迭代次数时重置不透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print(STR_ERROR, f'Resetting Opacity in {iteration}')
                    gaussians.reset_opacity()

                # 每隔一定迭代次数移除不正确的点
                #TODO: ablation study
                if iteration % opt.remove_outliers_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.remove_outliers(opt, iteration, linear=True)

            # Optimizer step 结束迭代前都要梯度清零（实际上是最后一次迭代无需清零更新梯度 因为已经结束训练了）
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 保存检查点
            if (iteration in checkpoint_iterations):
                print(STR_STAGE, "[ITER {}] Saving Checkpoint in {}".format(iteration, scene.model_path + "/ckpt" + str(iteration) + ".pth"))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt" + str(iteration) + ".pth")  # capture函数保存了训练状态 对应的恢复函数是restore

    # saving json result after training
    with open(args.model_path + '/train_gs.json', 'w') as fp:
        print(STR_DEBUG, f"saving results in {args.model_path + '/train_gs.json'}")
        json.dump(results, fp, indent=True)


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
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, depth_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    result = {
        'train_loss_patches/l1_loss': Ll1.item(),  # 和cal_loss函数中画图的数值一样 因为就是直接传进去的
        'train_loss_patches/depth_loss': depth_loss.item(), # 和cal_loss函数中画图的数值一样 因为就是直接传进去的
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
    #NOTE Before: if iteration in testing_iterations:
    if iteration % 100 == 0:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()}, 
            # {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
            #NOTE 注释掉可以只测试test相机的效果 {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, depth_test_loss = 0.0, 0.0, 0.0  # 计算所有35个test相机和隔5个train相机的总metric
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)  # range in [0.0, 1.3]
                    image = render_pkg['render']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = viewpoint.original_image.to("cuda")  # already range in [0.0, 1.0]
                    gt_image = torch.clamp(gt_image, 0.0, 1.0)

                    #
                    #NOTE for depth loss:
                    #? depth loss一定需要mask吗
                    gt_mask = torch.where(viewpoint.mask > 0.5, True, False)
                    render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
                    mask = torch.logical_and(gt_mask, render_mask)
                    if mask.sum() < 10:
                        depth_loss = 0.0
                    else:
                        disp_mono = 1 / viewpoint.mono_depth[mask].clamp(1e-6) # shape: [N]
                        disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                        depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]
                    #

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
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)  # 只在测试过程中记录高斯球的不透明度
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
   

def cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, silhouette_loss_type="bce", mono_loss_type="mid", tb_writer: Optional[SummaryWriter]=None, iteration=0):
    """
    Calculate the loss of the image, contains l1 loss and ssim loss.
    l1 loss: Ll1 = l1_loss(image, gt_image)
    ssim loss: Lssim = 1 - ssim(image, gt_image)
    Optional: [silhouette loss, monodepth loss]
    """
    gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()
    if opt.random_background:
        gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()
    Ll1 = l1_loss(image, gt_image)
    Lssim = (1.0 - ssim(image, gt_image))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

    training_psnr = psnr(image, gt_image).mean().double()

    if tb_writer is not None:
        tb_writer.add_scalar('training_loss/l1_loss', Ll1, iteration)
        tb_writer.add_scalar('training_loss/ssim_loss', Lssim, iteration)
        tb_writer.add_scalar('training_metrics/psnr', training_psnr, iteration)

    if hasattr(args, "use_mask") and args.use_mask:
        if silhouette_loss_type == "bce":
            silhouette_loss = F.binary_cross_entropy(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        elif silhouette_loss_type == "mse":
            silhouette_loss = F.mse_loss(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        else:
            raise NotImplementedError
        loss = loss + opt.lambda_silhouette * silhouette_loss
        if tb_writer is not None:
            tb_writer.add_scalar('training_loss/silhouette_loss', silhouette_loss, iteration)

    if hasattr(viewpoint_cam, "mono_depth") and viewpoint_cam.mono_depth is not None:
        if mono_loss_type == "mid":
            # we apply masked monocular loss
            gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
            render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
            mask = torch.logical_and(gt_mask, render_mask)
            if mask.sum() < 10:
                depth_loss = torch.tensor(0.0, device=loss.device)
            else:
                disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]
        elif mono_loss_type == "pearson":
            disp_mono = 1 / viewpoint_cam.mono_depth[viewpoint_cam.mask > 0.5].clamp(1e-6) # shape: [N]  clamp防止除0错误
            disp_render = 1 / render_pkg["rendered_depth"][viewpoint_cam.mask > 0.5].clamp(1e-6) # shape: [N]
            depth_loss = (1 - pearson_corrcoef(disp_render, -disp_mono)).mean()
        else:
            raise NotImplementedError

        loss = loss + args.mono_depth_weight * depth_loss
        if tb_writer is not None:
            tb_writer.add_scalar('training_loss/depth_loss', depth_loss, iteration)

    return loss, Ll1, depth_loss

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])  # 用于测试高斯球的效果 进行training_report
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000])  # 用于保存高斯球 也就是当前场景下高斯点的信息 不包括梯度信息
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 用于保存训练状态 包括高斯球的梯度信息
    parser.add_argument("--start_checkpoint", type=str, default = None)  # 用于恢复高斯球的训练状态
    ### some exp args
    parser.add_argument("--sparse_view_num", type=int, default=-1, 
                        help="Use sparse view or dense view, if sparse_view_num > 0, use sparse view, \
                        else use dense view. In sparse setting, sparse views will be used as training data, \
                        others will be used as testing data.")  # 4
    parser.add_argument("--use_mask", default=True, help="Use masked image, by default True")
    parser.add_argument("--init_pcd_name", default='origin', type=str, 
                        help="the init pcd name. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument("--transform_the_world", action="store_true", help="Transform the world to the origin")
    parser.add_argument('--mono_depth_weight', type=float, default=0.0005, help="The rate of monodepth loss")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)  # 10000
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)  # 初始化一个GUI
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # Pytorch检测梯度异常
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
