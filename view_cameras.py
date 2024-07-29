# from https://github.com/GaussianObject/GaussianObject/issues/25
import argparse
import math
import os
from argparse import Namespace

import camtools as ct
import cv2
import numpy as np
import open3d as o3d
import torch

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def pointidx2camidx(pointidx, cam_asset=5):
    camidx = []
    for id in pointidx:
        camidx.append(id//cam_asset)
    return camidx

def distance(a, b):
    return np.linalg.norm(a - b)

def error_same(a, b, epsilon=1e-6):
    dist = distance(a ,b)
    return dist < epsilon

def from_loc_find_idx(point_loc, all_loc):
    indices_list = []
    for pt in point_loc:
        for id, loc in enumerate(all_loc):
            if error_same(pt, loc):
                indices_list.append(id)
    assert len(indices_list) == len(point_loc)
    return indices_list

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='generate k views covering object')
    parser.add_argument('--num_views', type=int, default=4, help='number of views, K')
    parser.add_argument('--data_dir', type=str, default='data/mip360/kitchen', help='data directory, we only support colmap type data')

    args = parser.parse_args()

    data_dir = args.data_dir
    num_views = args.num_views
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extra_opts = Namespace()
    extra_opts.sparse_view_num = -1
    extra_opts.resolution = 1
    extra_opts.use_mask = False
    extra_opts.data_device = 'cuda'
    extra_opts.init_pcd_name = 'origin'
    extra_opts.white_background = False

    # load the camera parameters
    # we assume that the camera parameters are stored in the data_dir
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.data_dir, None, None, extra_opts=extra_opts) 
    camlist = cameraList_from_camInfos(scene_info.train_cameras, 1.0, extra_opts)

    # get all camera locations to recenter the scene
    cam_locations = []
    cam_rotations = []
    cam_T = []
    Ts = []
    Ks = []

    for cam_info in camlist:
        cam_locations.append(cam_info.camera_center)
        cam_rotations.append(cam_info.R)
        cam_T.append(cam_info.T)
        Ts.append(cam_info.world_view_transform.T)
        fx = fov2focal(cam_info.FoVx, cam_info.image_width)
        fy = fov2focal(cam_info.FoVy, cam_info.image_height)
        Ks.append(np.array([[fx, 0, cam_info.image_width/2], [0, fy, cam_info.image_height/2], [0, 0, 1]]))

    # just turn to numpy array
    cam_locations = np.array([i.cpu().numpy() for i in cam_locations])
    cam_rotations = np.array(cam_rotations)
    cam_T = np.array(cam_T)
    Ts = np.array([i.cpu().numpy() for i in Ts])
    Ks = np.array(Ks)

    # load pointcloud
    pcd = o3d.io.read_point_cloud(os.path.join(data_dir, "sparse/0/points3D.ply"))

    # vis
    # NOTE AttributeError: module 'camtools.camera' has no attribute 'create_camera_ray_frames'. Did you mean: 'create_camera_frames'?
    cameras = ct.camera.create_camera_frames(Ks, Ts, highlight_color_map={0: [1, 0, 0], -1: [0, 1, 0]})

    # init viewer
    numPoints = num_views
    PickedPointsNum = 0
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window()
    vis.add_geometry(cameras)
    while PickedPointsNum < numPoints:
        PickedPointsNum = len(vis.get_picked_points())  # 用户选择顶点
        vis.poll_events()
    vis.destroy_window()

    # initialize PickedPointSet object and store
    point_loc = []
    pickedPoints = vis.get_picked_points()[::-1]
    pickedPointIndexes = np.zeros(len(pickedPoints), dtype=int)
    for i, pickedPoint in enumerate(pickedPoints):
        pickedPointIndexes[i] = pickedPoint.index
        point_loc.append(pickedPoint.coord)

    # print(pickedPointIndexes)

    # one camera has 5 points, thus, the idx can be convert to cam idx.
    camidx = sorted(pointidx2camidx(pickedPointIndexes))
    print(f"camera idx is: {camidx}")

    # double check
    camidx2 = sorted(from_loc_find_idx(point_loc, cam_locations))
    print(f"id2 is {camidx2}")

    assert camidx == camidx2

    # build LineSet to represent the coordinate
    world_coord = o3d.geometry.LineSet()
    world_coord.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [2, 0, 0], 
                                                              [0, 0, 0], [0, 2, 0], 
                                                              [0, 0, 0], [0, 0, 2]]))
    world_coord.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3], [4, 5]]))
    # X->red, Y->green, Z->blue
    world_coord.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    selcameras = ct.camera.create_camera_ray_frames(Ks[np.asarray(camidx)], Ts[np.asarray(camidx)])

    o3d.visualization.draw_geometries([selcameras, pcd, world_coord])

    # save the idxs
    np.savetxt(os.path.join(data_dir, f"sparse_{str(num_views)}.txt"), np.array(camidx), fmt='%d')

    # to check the images selected, we also save the images from idx.
    os.makedirs(os.path.join(data_dir, 'visulization'), exist_ok=True)
    imgs = []
    for idx in camidx:
        img = (camlist[idx].original_image.permute(1,2,0).cpu().numpy()*255.).astype(np.uint8)
        imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    try:
        cv2.imwrite(os.path.join(data_dir, 'visulization', str(num_views) + ".png"), np.hstack(imgs))
    except:
        os.makedirs(os.path.join(data_dir, 'visulization', str(num_views)))
        for i, img in enumerate(imgs):
            cv2.imwrite(os.path.join(data_dir, 'visulization', str(num_views), str(i) + ".png"), img)
