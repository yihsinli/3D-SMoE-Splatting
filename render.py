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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from scene.cameras import Camera
import imageio
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,offset):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "round_renders")
    #gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    #makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        #gt = view.original_image[0:3, :, :]

        #if args.train_test_exp:
        #    rendering = rendering[..., rendering.shape[-1] // 2:]
        #    gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}.png'.format(idx+offset)))
        #torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in tqdm(sorted(os.listdir(image_folder))[:2000], desc="Rendering progress"):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree,"")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        pose_path = os.path.join(dataset.model_path,'pose','ours_{}'.format(iteration))
        thres = dataset.model_path.split('/')[1]
        inter_pose = np.load(os.path.join(pose_path.replace('GS','SMoE').replace(thres,'th.005'), 'pose_interpolated.npy'))# 'pose_interpolated.npy'
        current_views = scene.getTrainCameras()

        for offset in range(0,2000,200):
            inter_cameras = [Camera(colmap_id=i, R=inter_pose[i][:3,:3], T=inter_pose[i][:3,3], 
                  FoVx=current_views[0].FoVx, FoVy=current_views[0].FoVy, 
                  image=current_views[0].original_image, gt_alpha_mask=current_views[0].original_image,
                  image_name='{:03d}.jpg'.format(i), uid=i, data_device=dataset.data_device) for i in range(offset, min([len(inter_pose),offset+200]))]
            #poses = np.ones(len(scene.getTrainCameras()),3,4)
            #for i,view in enumerate(scene.getTrainCameras()):
            #    pose
            #    print(view.R, view.T)
            #if not skip_train:
            print('{} / 2000'.format(offset))
            render_set(dataset.model_path, "infer", scene.loaded_iter, inter_cameras, gaussians, pipeline, background, offset)

        image_path = os.path.join(dataset.model_path, "infer", "ours_{}".format(iteration), "renders")
        thres = dataset.model_path.split('/')[1].replace('th','0')
        scene_name = dataset.model_path.split('/')[3].split('-')[0]
        if thres == '0.005':
            video_path = os.path.join(dataset.model_path, "infer", "ours_{}".format(iteration),'{}_l2.mp4'.format(scene_name,thres))
        else:
            video_path = os.path.join(dataset.model_path, "infer", "ours_{}".format(iteration),'{}_h.mp4'.format(scene_name))

        video_path = os.path.join(dataset.model_path, "infer", "ours_{}".format(iteration),'{}_{}.mp4'.format(scene_name,thres))
        #video_path = os.path.join(dataset.model_path, "infer", "ours_{}".format(iteration),'zoom_in_interp.mp4')
        images_to_video(image_path, video_path , fps=30)
        #if not skip_test:
        #render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)