# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

import os
import os.path as osp
import sys
import pickle

import numpy as np
import open3d as o3d
import torch
import trimesh
from loguru import logger
from tqdm import tqdm

from smplx import build_layer

from .config import parse_args
from .data import build_dataloader
from .transfer_model import run_fitting
from .utils import read_deformation_transfer, np_mesh_to_o3d


def main() -> None:
    exp_cfg = parse_args()

    if torch.cuda.is_available() and exp_cfg["use_cuda"]:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        if exp_cfg["use_cuda"]:
            if input("use_cuda=True and GPU is not available, using CPU instead,"
                     " would you like to continue? (y/n)") != "y":
                sys.exit(3)

    logger.remove()
    logger.add(
        lambda x: tqdm.write(x, end=''), level=exp_cfg.logger_level.upper(),
        colorize=True)

    output_folder = osp.expanduser(osp.expandvars(exp_cfg.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    model_path = exp_cfg.body_model.folder
    body_model = build_layer(model_path, **exp_cfg.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=device)

    deformation_transfer_path = exp_cfg.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)

    # Read mask for valid vertex ids
    mask_ids_fname = osp.expandvars(exp_cfg.mask_ids_fname)
    mask_ids = None
    if osp.exists(mask_ids_fname):
        logger.info(f'Loading mask ids from: {mask_ids_fname}')
        mask_ids = np.load(mask_ids_fname)
        mask_ids = torch.from_numpy(mask_ids).to(device=device)
    else:
        logger.warning(f'Mask ids fname not found: {mask_ids_fname}')

    data_obj_dict = build_dataloader(exp_cfg)

    dataloader = data_obj_dict['dataloader']
    extras = {}

    for ii, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)
        if os.path.exists(exp_cfg.optim.shape_file):
            betas = np.load(exp_cfg.optim.shape_file)['betas']
            extras['betas'] = torch.from_numpy(betas).to(device=device).float().expand(len(batch['paths']), -1)
        elif os.path.exists(exp_cfg.optim.pose_file) and os.path.isdir(exp_cfg.optim.pose_file):
            for j in range(len(batch['paths'])):
                name = os.path.basename(batch['paths'][j])
                pose_file = os.path.join(exp_cfg.optim.pose_file, name.replace('.ply', '.npz'))
                pose = np.load(pose_file)
                for k, v in pose.items():
                    if k in extras:
                        extras[k].append(torch.from_numpy(v).to(device=device).float())
                    else:
                        extras[k] = [torch.from_numpy(v).to(device=device).float()]
            for k, v in extras.items():
                extras[k] = torch.cat(v, dim=0)
        var_dict = run_fitting(
            exp_cfg, batch, body_model, def_matrix, mask_ids, **extras)
        extras = {}
        paths = batch['paths']
        for k, v in var_dict.items():
            if torch.is_tensor(v):
                var_dict[k] = v.detach().cpu().numpy()

        for ii, path in enumerate(paths):
            _, fname = osp.split(path)
            data = dict()
            for k, v in var_dict.items():
                if k == 'faces':
                    data[k] = v
                elif k == 'loss':
                    data[k] = v
                else:
                    data[k] = v[ii:ii+1]
            # output_path = osp.join(
            #     output_folder, f'{osp.splitext(fname)[0]}.pkl')
            # with open(output_path, 'wb') as f:
            #     pickle.dump(var_dict, f)
            if not exp_cfg.optim.mesh_only:
                output_path = osp.join(
                    output_folder, f'{osp.splitext(fname)[0]}.npz')
                np.savez_compressed(output_path, **data)

            if exp_cfg.vis_mesh or exp_cfg.optim.mesh_only:
                output_path = osp.join(
                    output_folder, f'{osp.splitext(fname)[0]}.ply')
                # mesh = np_mesh_to_o3d(
                #     data['vertices'][0], data['faces'])
                # o3d.io.write_triangle_mesh(output_path, mesh)
                mesh = trimesh.Trimesh(vertices=data['vertices'][0], faces=data['faces'], process=False)
                mesh.export(output_path)

if __name__ == '__main__':
    main()
