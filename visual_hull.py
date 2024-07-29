import argparse
import math
import os
import copy
from argparse import Namespace
from typing import List, NamedTuple

import torch
import camtools as ct
import numpy as np
import open3d as o3d
from torch.nn import functional as F
from torchvision import transforms
from tqdm import trange

from scene.cameras import Camera
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


class SparseSceneInfo(NamedTuple):
    Ks: list
    Ts: list
    images: list
    masks: list
    

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def points2homopoints(points):
    assert points.shape[-1] == 3
    bottom = torch.ones_like(points[...,0:1])
    return torch.cat([points, bottom], dim=-1)

def batch_projection(Ks, Ts, points):
    '''
    Ks: B, 3, 3
    Ts: B, 4, 4
    points: B, N, 3
    return: [u, v]: , depth
    '''
    pre_fix = points.shape[:-1] # [100, 100]
    points = points.reshape(-1, 3) # [M, 3]

    Ts = torch.stack(Ts, dim=0) # [N, 4, 4]
    Ks = torch.stack(Ks, dim=0).to(Ts.device) # [N, 3, 3]
    camera_num = Ks.shape[0]
    homopts = points2homopoints(points) # [M, 4]
    # world to camera # [N, M, 4] @ [N, 4, 4] = [N, M, 4]
    homopts_cam = torch.bmm(homopts.unsqueeze(0).repeat_interleave(Ts.shape[0], dim=0), Ts.transpose(1,2)) 
    # camera to image space  # [N, M, 4] @ [N, 4, 3] = [N, M, 3]
    homopts_img = torch.bmm(homopts_cam[...,:3], Ks.transpose(1,2))
    # normalize
    homopts_img = homopts_img / (homopts_img[...,2:] + 1e-6)
    # reshape back
    homopts_img = homopts_img.reshape(camera_num, *pre_fix, 3)
    homopts_cam = homopts_cam.reshape(camera_num, *pre_fix, 4)
    return homopts_img[...,0:2], homopts_cam[...,2]

def query_from_list_with_list(listA: list, listB: list):
    '''
    listA: [1, 2, 3]
    listB: [3, 2, 1]
    return: [2, 1, 0]
    '''
    return [listB[i] for i in listA]

def simple_resize_image(img, size):
    return transforms.Resize(size, antialias=True)(img)  # type: ignore

def get_visual_hull(voxel_num, bbox, scene_info, cam_center):
    u"""
    N: 需要采样的深度射线

    先构建3D点
        构建 2D mesh
        均匀采样出深度
        组合 mesh 和深度的坐标值构成 3D 点
        移动 3D 点使点集以相机为中心
    投影 3D 点到 2D 平面进而得到 2D mesh
        batch_projection
        去掉 2D 平面上投影出来不合法的点
    按照 2D mesh 对原图像采样 **相当于得到原图像到3D空间点的映射**
    平均起来得到 3D 空间中带颜色的初始化点
    """
    pcs = []
    color = []
    Ks = scene_info.Ks
    Ts = scene_info.Ts
    images = scene_info.images
    masks = scene_info.masks

    [xs, ys, zs], [xe, ye, ze] = bbox[0], bbox[1]  # [-4., -4., -4.], [4., 4., 4.]

    # please note that in vasedeck, the images are not same size, for simplify, just resize them
    new_images = []
    new_masks = []
    img_size = images[0].shape[1:]  # [1039, 1558]
    for image, mask in zip(images, masks):
        new_images.append(simple_resize_image(image, img_size))
        new_masks.append(simple_resize_image(mask, img_size))

    images = torch.stack(new_images) # N C H W
    masks = torch.stack(new_masks) # N 1 H W

    for h_id in trange(voxel_num):
        i, j = torch.meshgrid(torch.linspace(xs, xe, voxel_num).cuda(),
                              torch.linspace(ys, ye, voxel_num).cuda())  # i,j shape both are: (200, 200)
        i, j = i.t(), j.t()
        pts = torch.stack([i, j, torch.ones_like(i).cuda()], -1)  # (200, 200, 3)
        pts[..., 2] = h_id / voxel_num * (ze - zs) + zs # 100, 100, 3  均匀地从深度值中采样出来！

        """refine coord:
        _ _ _ y
        |\
        | \
        |  \
        x    z
        """

        # shift the pts to be centered at the camera center
        pts[...,0] += cam_center[0]  # note the order, [x, y, z], width, height, depth
        pts[...,1] += cam_center[1]
        pts[...,2] += cam_center[2]

        # now we have the pts, we need to project them to the image plane
        # batched projection
        uv, z = batch_projection(Ks, Ts, pts) # [N, 100, 100, 2], [N, 100, 100]  把3D点投影到2D平面上
        valid_z_mask = z > 0  # 要求投影后深度值为正
        valid_x_y_mask = (uv[...,0] > 0) & (uv[...,0] < cam_info.image_width) & (uv[...,1] > 0) & (uv[...,1] < cam_info.image_height)
        valid_pt_mask = valid_z_mask & valid_x_y_mask  # 挑出合法的3D点

        # simple resize the uv to [-1, 1]
        # 归一化到NDC空间便于下面采样
        uv[...,0] = uv[...,0] / cam_info.image_width * 2 - 1
        uv[...,1] = uv[...,1] / cam_info.image_height * 2 - 1

        # 采样得到3D空间的图片
        # now we have the uv, we use grid_sample to sample the image to get the color
        result = F.grid_sample(images.float(), uv, padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1) # N, 100, 100, 3
        # sample mask
        result_mask = F.grid_sample(masks.float(), uv, padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1) # N, 100, 100, 1

        #? valid_pt_mask全为0时 为什么得出的最后的valid_pt_mask不是全0?
        valid_pt_mask = result_mask.squeeze() > 0 & valid_pt_mask  # 颜色大于0的mask点

        pcs.append(valid_pt_mask.float().sum(0) >= (images.shape[0] - 1)) # (200, 200)  拿到当前的四张图片共有的合法的点的位置坐标 要求至少3张图片都出现的点才是合法的
        color.append(result.mean(0)) # [200, 200, 3] 平均采样深度上所有图像的color值???
    
    pcs = torch.stack(pcs, -1)  # (200, 200, 200)  大小为200*200的共200张图
    color = torch.stack(color, -1)  # (200, 200, 3, 200)

    r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]  # 都是(200, 200, 200)
    idx = torch.where(pcs > 0)

    color = torch.stack((r[idx] * 255, g[idx] * 255, b[idx] * 255), -1)  # (200, 200, 3, 200)

    idx = torch.stack([idx[1], idx[0], idx[2]], -1) # note the order is hwz -> xyz  (22537, 3)
    # turn the idx to the point position used in batch_projection
    idx = idx.float() / voxel_num  # 恢复到正常的全局坐标系下 原来为了采样是在网格坐标系下的索引
    idx[...,0] = idx[...,0] * (xe - xs) + xs + cam_center[0]
    idx[...,1] = idx[...,1] * (ye - ys) + ys + cam_center[1]
    idx[...,2] = idx[...,2] * (ze - zs) + zs + cam_center[2]

    print("visual hull is Okay, with {} points".format(idx.shape[0]))
    # we get the point cloud, use open3d to visualize it
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(idx.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy() / 255)

    # get bbox
    bbox = pcd.get_axis_aligned_bounding_box()
    return pcd, bbox


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='generate k views covering object')
    parser.add_argument('--data_dir', type=str, default='sparse_nerf_datasets/sparse_omni3d_undistorted/backpack_016', help='data directory, we only support colmap type data, kitchen, garden')
    parser.add_argument("--cube_size", type=float, default=4.0, help="size of the cube in meters")
    parser.add_argument("--voxel_num", type=int, default=200, help="size of a voxel in meters")
    parser.add_argument('--sparse_id', type=int, default=-1, help='sparse id')  # 4
    parser.add_argument('--reso', type=int, default=1, help='the resolution of image, 1 for omni3d, 4 or 8 for mip360')  # 2
    parser.add_argument('--not_vis', action='store_true', help='whether vis the visual hull, is enable, not vis')
    # visual hull 的中心点
    parser.add_argument("--cube_size_shift_x", type=float, default=0.0, help="shift sizex of the cube in meters")
    parser.add_argument("--cube_size_shift_y", type=float, default=0.0, help="shift sizey of the cube in meters")
    parser.add_argument("--cube_size_shift_z", type=float, default=0.0, help="shift sizez of the cube in meters")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extra_opts = Namespace()
    extra_opts.sparse_view_num = -1
    extra_opts.resolution = args.reso
    extra_opts.use_mask = True
    extra_opts.data_device = 'cuda'
    extra_opts.init_pcd_name = 'origin'
    extra_opts.white_background = False


    # load the camera parameters
    # we assume that the camera parameters are stored in the data_dir
    # 加载场景信息和相机信息
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.data_dir, 'images', eval=False, extra_opts=extra_opts)  # 如果不是eval，则全部拿来，否则每8个作为一个test样本，剩下7个样本才是train
    camlist: List[Camera] = cameraList_from_camInfos(scene_info.train_cameras, 1.0, extra_opts)  # 279

    # if sparse id is not zero, we only use given frames to construct the visual hull
    if args.sparse_id >= 0:
        selected_id = np.loadtxt(os.path.join(args.data_dir, f"sparse_{str(args.sparse_id)}.txt"), dtype=np.int32)
        print("the sparse id is {}, with {} frames".format(args.sparse_id, len(selected_id)))  # 4 with 4 frames
        assert args.sparse_id == len(selected_id)
    else:
        selected_id = np.arange(len(camlist))

    # get all camera locations to recenter the scene
    cam_locations = []
    cam_rotations = []
    cam_T = []
    Ts = []
    Ks = []
    images = []
    masks = []
    for cam_info in camlist:
        cam_locations.append(cam_info.camera_center)
        cam_rotations.append(cam_info.R)
        cam_T.append(cam_info.T)
        Ts.append(cam_info.world_view_transform.T)
        fx = fov2focal(cam_info.FoVx, cam_info.image_width)
        fy = fov2focal(cam_info.FoVy, cam_info.image_height)
        Ks.append(torch.tensor([[fx, 0, cam_info.image_width/2], [0, fy, cam_info.image_height/2], [0, 0, 1]]))
        images.append(cam_info.original_image)
        masks.append(cam_info.mask)

    # in this time, we already have the camera parameters
    # first, we get the cemera locations center
    #? 为什么稀疏视角要用全视角的相机中心？如果构建不了怎么办？
    cam_center = torch.stack(cam_locations).mean(0)  # 拿到所有相机中心 是一个三元素的张量
    print('the camera center is:', cam_center)  # tensor([-0.0710, -0.0025,  0.0700], device='cuda:0')
 
    # 挑出 refimg 的信息 长度为4
    Ks = query_from_list_with_list(selected_id, Ks)
    Ts = query_from_list_with_list(selected_id, Ts)
    images = query_from_list_with_list(selected_id, images)
    masks = query_from_list_with_list(selected_id, masks)

    scene_info = SparseSceneInfo(Ks, Ts, images, masks)
    Ks_clone = copy.deepcopy(Ks)

    ############ get visual hull ############ 

    # 构建一个大小为2bx*2bx*2bx的正方体
    bx = args.cube_size
    init_bbox = [[args.cube_size_shift_x-bx, args.cube_size_shift_y-bx, args.cube_size_shift_z-bx], 
                 [args.cube_size_shift_x+bx, args.cube_size_shift_y+bx, args.cube_size_shift_z+bx]]
    # we run the get_visual_hull twice, first to get the bound, second to get the visual hull
    pcd, bbox = get_visual_hull(args.voxel_num, init_bbox, scene_info, cam_center)  # 25537 points
    
    # 扩充bbx
    # since we get the bound, we use this bound to better recon
    # we use the center of the bound as the center of the scene
    # please note that the bbox may need bigger, since the camera may not cover the whole scene
    bbox_min = bbox.get_min_bound()
    bbox_max = bbox.get_max_bound()
    # Calculate the center point of the original bounding box
    center = (bbox_min + bbox_max) / 2
    # Calculate the extents of the original bounding box
    extents = bbox_max - bbox_min
    # Calculate the scale factor to increase the size by 20% (1.2 times)
    scale_factor = 2
    # Calculate the scaled extents
    scaled_extents = extents * scale_factor
    # Calculate the new minimum and maximum points of the enlarged bounding box
    enlarged_bbox_min = center - scaled_extents / 2
    enlarged_bbox_max = center + scaled_extents / 2

    # 重新采样64个深度值，然后得到新的 visual hull
    # 第一次生成visual hull的时候点全部都移动到以相机为中心了，不需要再移动了
    pcd, bbox_new = get_visual_hull(64, [enlarged_bbox_min, enlarged_bbox_max], scene_info, [0,0,0])  # 8616 points
    # save the pointcloud
    if args.sparse_id >= 0:
        o3d.io.write_point_cloud(os.path.join(args.data_dir, f"visual_hull_{str(args.sparse_id)}.ply"), pcd)
    else:
        o3d.io.write_point_cloud(os.path.join(args.data_dir, "visual_hull_full.ply"), pcd)

    if not args.not_vis:
        #NOTE Visualize point3D.ply
        Ts = np.array([i.cpu().numpy() for i in Ts])
        Ks = np.array(Ks_clone)
        cameras = ct.camera.create_camera_frames(Ks, Ts, highlight_color_map={0: [1, 0, 0], -1: [0, 1, 0]})
        # build LineSet to represent the coordinate
        world_coord = o3d.geometry.LineSet()
        world_coord.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [2, 0, 0], 
                                                                [0, 0, 0], [0, 2, 0], 
                                                                [0, 0, 0], [0, 0, 2]]))
        world_coord.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 3], [0, 5]]))
        # X->red, Y->green, Z->blue
        world_coord.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        
        pcdo = o3d.io.read_point_cloud(os.path.join(args.data_dir, "sparse/0/points3D.ply"))

        # init viewer
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(cameras)
        viewer.add_geometry(pcd)
        viewer.add_geometry(world_coord)
    
        opt = viewer.get_render_option()
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()
