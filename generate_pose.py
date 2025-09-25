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
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
import numpy as np
from utils.camera_utils import generate_interpolated_path, visualizer
from pathlib import Path
import matplotlib.pyplot as plt




def pose_generator(dataset, iteration, zoom_in):


    dataset.resolution = 1
    dataset.data_device = 'cuda'
    dataset.sh_degree = 3

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree,"")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    views = scene.getTrainCameras()
    org_pose = np.zeros((len(views), 4,4))

    for i,view in enumerate(views):
        org_pose[i,:3,:3] = view.R
        org_pose[i,:3,3] = view.T
    org_pose[:,3,3] = 1.0


    # no zoom in
    if zoom_in:
        org_pose_new = np.zeros((len(org_pose)*2, org_pose.shape[1],org_pose.shape[2]))
        for i in range(0,len(org_pose_new),2):
            org_pose_new[i] = org_pose[i//2]
        d = 3
        for i in range(0,len(org_pose_new),2):
            pose0 = org_pose_new[i].copy()
            pose1 = pose0
            pose1[:3,-1] = pose1[:3,-1] - d * pose1[:3,2]
            org_pose_new[i+1] = pose1
        org_pose_new = org_pose_new[:len(org_pose)]

    else:
        org_pose_new = np.zeros((len(org_pose), org_pose.shape[1],org_pose.shape[2]))
        org_pose_new = org_pose[:len(org_pose_new)]




    #def save_interpolate_pose(model_path, iter, n_views):
    model_path = Path(dataset.model_path)
    iter = args.iteration
    #org_pose = np.load(model_path / f"pose/ours_{iter}/pose_optimized.npy")
    pose_path = os.path.join(model_path, 'pose', 'ours_{}'.format(args.iteration))
    if zoom_in:
        pose_name = 'zoom_in_pose'
    else:
        pose_name = 'pose'
        
    os.makedirs(pose_path,exist_ok=True)
    visualizer(org_pose_new, ["green" for _ in org_pose], os.path.join(pose_path, '{}_optimized.png'.format(pose_name)))

    n_views = len(org_pose_new)
    n_interp = 10 #int(n_views * 30 / n_views)  # 10second, fps=30


    all_inter_pose = []
    for i in tqdm(range(n_views-1)):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose_new[i:i+2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.concatenate(all_inter_pose, axis=0)
    all_inter_pose = np.concatenate([all_inter_pose, org_pose[-1][:3, :].reshape(1, 3, 4)], axis=0)

    inter_pose_list = []
    for p in tqdm(all_inter_pose):
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    visualizer(inter_pose, ["blue" for _ in inter_pose], os.path.join(pose_path, '{}_interpolated.png'.format(pose_name)))
    np.save(os.path.join(pose_path, '{}_interpolated.npy'.format(pose_name)) , inter_pose)












if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Generate navigation pose")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--zoom_in", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    pose_generator(model.extract(args), args.iteration, args.zoom_in)