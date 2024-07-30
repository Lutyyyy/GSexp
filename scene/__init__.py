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

import os
import random
import json
import time

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.util_print import STR_STAGE, STR_VERBOSE, STR_DEBUG


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0], extra_opts=None, load_ply=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path # type: ignore
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration and load_ply is None:  # 如果有已加载模型的迭代次数并且没有指定哪个ply文件 那么直接尝试恢复已加载模型的训练步数
            if load_iteration == -1:  # 如果没有提供 load_iteration，则将点云数据和相机信息保存到文件中
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:  # 否则保存好已加载模型的训练步数
                self.loaded_iter = load_iteration
            print(STR_DEBUG + f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}
        self.render_cameras = {}

        # 根据场景类型（Colmap 或 Blender）加载所有的场景信息
        if os.path.exists(os.path.join(args.source_path, "sparse")): # type: ignore
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, extra_opts=extra_opts) # type: ignore
        elif os.path.exists(os.path.join(args.source_path, "transforms_alignz_train.json")): # type: ignore
            print("Found transforms_alignz_train.json file, assuming OpenIllumination data set!")
            scene_info = sceneLoadTypeCallbacks["OpenIllumination"](args.source_path, args.white_background, args.eval, extra_opts=extra_opts) # type: ignore
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")): # type: ignore
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, extra_opts=extra_opts) # type: ignore
        elif os.path.exists(os.path.join(args.source_path, "hydrant", "frame_annotations.jgz")): # type: ignore
            scene_info = sceneLoadTypeCallbacks["CO3D"](args.source_path, extra_opts=extra_opts) # type: ignore
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter and load_ply is None:  # 如果没有预训练模型的迭代初始值并且也没有指定ply文件 则保存相机信息
            # NOTE :this dump use the file name, we dump the SceneInfo.pcd as the input.ply
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.render_cameras:
                camlist.extend(scene_info.render_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 设置相机范围
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 加载训练、测试、环形相机
        for resolution_scale in resolution_scales:
            init_time = time.time()
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, mode="train")
            init_time2 = time.time()
            print(STR_STAGE, "Loading training cameras with {}s for {} cameras".format(init_time2 - init_time, len(self.train_cameras[resolution_scale])))
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            init_time3 = time.time()
            print(STR_STAGE, "Loading test cameras with {}s for {} cameras".format(time.time() - init_time2, len(self.test_cameras[resolution_scale])))
            self.render_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.render_cameras, resolution_scale, args)
            print(STR_STAGE, "Loading render cameras with {}s for {} cameras".format(time.time() - init_time3, len(self.render_cameras[resolution_scale])))

        # 加载或创建高斯模型
        if self.loaded_iter:
            # 如果已加载模型，则调用 load_ply 方法尝试从默认的ply文件中加载点云数据
            # 相当于初始化点云信息 但没有存下梯度信息
            load_name = "point_cloud.ply"
            loaded_path = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), load_name)
            print(STR_DEBUG, f"In scene: loaded_iter from {loaded_path}")
            self.gaussians.load_ply(loaded_path)
        elif load_ply:  # 如果指定了ply文件路径 则直接加载
            print(STR_DEBUG, f"In scene: load_ply from {load_ply}")
            self.gaussians.load_ply(load_ply)
            # in this case, we need it to be trainable, so we need to make sure the spatial_lr_scale is not 0
            self.gaussians.spatial_lr_scale = self.cameras_extent
        else:
            # 否则，调用 create_from_pcd 方法根据场景信息中的点云数据创建高斯模型
            # 也就是从 init_pcd_name 点云中创建高斯模型
            print(STR_DEBUG, f"In scene/__init__.py: create_from_pcd {scene_info.ply_path}")
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.save_ply(os.path.join(self.model_path, "input.ply"))

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getAllCameras(self, scale=1.0):
        return self.train_cameras[scale] + self.test_cameras[scale]

    def getRenderCameras(self, scale=1.0):
        return self.render_cameras[scale]
