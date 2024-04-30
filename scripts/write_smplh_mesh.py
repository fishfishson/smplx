import os
import argparse
from os.path import join
import json
import trimesh
import open3d as o3d
import torch

from transfer_model.utils import np_mesh_to_o3d

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_dotdict, to_tensor, to_cuda
from easyvolcap.utils.parallel_utils import parallel_execution
from easymocap.bodymodel.smplx import SMPLModel, SMPLHModel


def write_smplh_mesh(param, model, output_dir):
    with open(param, 'r') as f:
        data = json.load(f)['annots'][0]
    for k in data.keys():
        data[k] = torch.from_numpy(np.array(data[k])).float()
    data.pop('id')
    data.pop('R_handl3d')
    data.pop('R_handr3d')
    data.pop('T_handl3d')
    data.pop('T_handr3d')
    data.pop('handl')
    data.pop('handr')
    v = model.forward(**data).numpy()[0]
    f = model.faces_tensor.numpy()
    output_name = join(output_dir, os.path.basename(param).replace('.json', '.ply'))
    # mesh = np_mesh_to_o3d(v, f)
    # o3d.io.write_triangle_mesh(output_name, mesh)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.export(output_name)


def write_smplh_mesh_woRT(param, model, output_dir):
    with open(param, 'r') as f:
        data = json.load(f)['annots'][0]
    for k in data.keys():
        data[k] = torch.from_numpy(np.array(data[k])).float()
    data.pop('id')
    data.pop('R_handl3d')
    data.pop('R_handr3d')
    data.pop('T_handl3d')
    data.pop('T_handr3d')
    data.pop('handl')
    data.pop('handr')
    data.pop('Rh')
    data.pop('Th')
    v = model.forward(**data).numpy()[0]
    f = model.faces_tensor.numpy()
    output_name = join(output_dir, os.path.basename(param).replace('.json', '.ply'))
    # mesh = np_mesh_to_o3d(v, f)
    # o3d.io.write_triangle_mesh(output_name, mesh)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.export(output_name)


def write_cano_smplh_mesh(param, model, output_dir):
    with open(param, 'r') as f:
        data = json.load(f)['annots'][0]
    for k in data.keys():
        data[k] = torch.from_numpy(np.array(data[k])).float()
    data.pop('id')
    data.pop('R_handl3d')
    data.pop('R_handr3d')
    data.pop('T_handl3d')
    data.pop('T_handr3d')
    data.pop('handl')
    data.pop('handr')
    data['poses'] = torch.zeros_like(data['poses'])
    data['Rh'] = torch.zeros_like(data['Rh'])
    data['Th'] = torch.zeros_like(data['Th'])
    v = model.forward(**data).numpy()[0]
    f = model.faces_tensor.numpy()
    # mesh = np_mesh_to_o3d(v, f)
    # o3d.io.write_triangle_mesh(join(output_dir, 'cano.ply'), mesh)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.export(join(output_dir, 'cano.ply'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='smplh')
    parser.add_argument('--input_dir', type=str, default='output-output-smpl-3d/smplfull')
    parser.add_argument('--output_dir', type=str, default='output-output-smpl-3d/mesh')
    parser.add_argument('--woRT', action='store_true', default=False)
    parser.add_argument('--cano', action='store_true', default=False)
    args = parser.parse_args()

    SMPLH_CFG = dotdict()
    SMPLH_CFG.model_path = 'data/bodymodels/smplhv1.2/neutral/model.npz'
    SMPLH_CFG.regressor_path = 'data/smplx/J_regressor_body25_smplh.txt'
    SMPLH_CFG.mano_path = 'data/bodymodels/manov1.2'
    SMPLH_CFG.cfg_hand = dotdict()
    SMPLH_CFG.cfg_hand.use_pca = True
    SMPLH_CFG.cfg_hand.use_flat_mean = False
    SMPLH_CFG.cfg_hand.num_pca_comps = 12

    if args.model_type == 'smplh':
        model = SMPLHModel(**SMPLH_CFG, device='cpu')
    else:
        raise ValueError(f'Unknown model type: {args.model_type}')
    
    input_dir = join(args.data_root, args.input_dir)
    smpl_params = [join(input_dir, x) for x in sorted(os.listdir(input_dir))]
    # write_smplh_mesh(smpl_params[1000], model, output_dir)
    if args.woRT:
        output_dir = join(args.data_root, args.output_dir + '-woRT')
        os.makedirs(output_dir, exist_ok=True)
        parallel_execution(smpl_params, model=model, output_dir=output_dir, action=write_smplh_mesh_woRT, 
                           print_progress=True, sequential=False)
    elif args.cano:
        output_dir = join(args.data_root, args.output_dir, '..')
        write_cano_smplh_mesh(smpl_params[0], model=model, output_dir=output_dir)
    else:
        output_dir = join(args.data_root, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        parallel_execution(smpl_params, model=model, output_dir=output_dir, action=write_smplh_mesh, 
                           print_progress=True, sequential=False)
        # v = model.v_template.numpy()
        # f = model.faces_tensor.numpy()
        # mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        # mesh.export(join(output_dir, 'cano.ply'))

    # for i in tqdm(range(smpl_data['poses'].shape[0])):
    #     v = model.forward(poses=smpl_data['poses'][i:i+1], 
    #                       shapes=smpl_data['shapes'][i:i+1],
    #                       Rh=smpl_data['Rh'][i:i+1],
    #                       Th=smpl_data['Th'][i:i+1],).numpy()[0]
    #     f = model.faces_tensor.numpy()
    #     mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    #     mesh.export(join(args.data_root, output_dir, f'{i:06}.ply'))

    # output_dir = args.output_dir + '-woRT'
    # os.makedirs(join(args.data_root, output_dir), exist_ok=True)
    # for i in tqdm(range(smpl_data['poses'].shape[0])):
    #     v = model.forward(poses=smpl_data['poses'][i:i+1], 
    #                       shapes=smpl_data['shapes'][i:i+1]).numpy()[0]
    #     f = model.faces_tensor.numpy()
    #     mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    #     mesh.export(join(args.data_root, output_dir, f'{i:06}.ply'))