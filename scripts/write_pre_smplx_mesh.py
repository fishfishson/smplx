import os
import argparse
from os.path import join
import json
import trimesh
import torch
import smplx
import numpy as np

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_dotdict, to_tensor, to_cuda
from easyvolcap.utils.parallel_utils import parallel_execution

from transfer_model.utils import batch_rodrigues, batch_rot2aa, np_mesh_to_o3d


def write_pre_smplx_mesh(smpl_param, cano_param, body_model, output_dir):
    with open(smpl_param) as f:
        smpl_data = json.load(f)['annots'][0]
    for k in smpl_data.keys():
        smpl_data[k] = torch.from_numpy(np.array(smpl_data[k])).float()

    smplx_data = {}
    smplx_data['transl'] = cano_param['transl']
    smplx_data['global_orient'] = torch.eye(3).reshape(1, 1, 3, 3)
    smplx_data['body_pose'] = batch_rodrigues(smpl_data['poses'][:, 3:66].reshape(21, 3))[None]
    smplx_data['betas'] = cano_param['betas']
    smplx_data['left_hand_pose'] = batch_rodrigues(smpl_data['poses'][:, 66:111].reshape(15, 3))[None]
    smplx_data['right_hand_pose'] = batch_rodrigues(smpl_data['poses'][:, 111:156].reshape(15, 3))[None]
    smplx_data['leye_pose'] = torch.eye(3).reshape(1, 1, 3, 3)
    smplx_data['reye_pose'] = torch.eye(3).reshape(1, 1, 3, 3)
    smplx_data['jaw_pose'] = torch.eye(3).reshape(1, 1, 3, 3)
    smplx_data['expression'] = torch.zeros_like(cano_param['expression'])
    # smplx_data['global_orient'] = torch.zeros(1, 3)
    # smplx_data['body_pose'] = smpl_data['poses'][:, 3:66].reshape(21, 3)[None]
    # smplx_data['betas'] = cano_param['betas']
    # smplx_data['left_hand_pose'] = smpl_data['poses'][:, 66:111].reshape(15, 3)[None]
    # smplx_data['right_hand_pose'] = smpl_data['poses'][:, 111:156].reshape(15, 3)[None]
    # smplx_data['leye_pose'] = torch.zeros(1, 3)
    # smplx_data['reye_pose'] = torch.zeros(1, 3)
    # smplx_data['jaw_pose'] = torch.zeros(1, 3)
    # smplx_data['expression'] = torch.zeros(1, 10)
    # smplx_out = body_model(**smplx_data)
    # v = smplx_out.vertices.cpu().numpy().reshape(-1, 3)
    # f = body_model.faces
    # mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    # mesh.export(join(output_dir, os.path.basename(smpl_param).replace('.json', '.ply')))
    for k, v in smplx_data.items():
        smplx_data[k] = v.numpy()
    np.savez_compressed(join(output_dir, os.path.basename(smpl_param).replace('.json', '.npz')), **smplx_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--input_dir', type=str, default='output-output-smpl-3d/smplfull')
    parser.add_argument('--output_dir', type=str, default='output-output-smpl-3d/mesh-pre-smplx')
    args = parser.parse_args()

    body_model = smplx.SMPLXLayer(model_path='./models/smplx',
                                  gender='neutral',
                                  use_compressed=False,
                                  use_face_contour=True,
                                  num_betas=16,
                                  num_expression_coeffs=10)

    input_dir = join(args.data_root, args.input_dir)
    output_dir = join(args.data_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cano_file = join(args.data_root, 'output-output-smpl-3d/mesh-smplx/cano.npz')

    cano_param = np.load(cano_file)
    cano_param = {k: v for k, v in cano_param.items()}
    cano_param.pop('full_pose')
    cano_param.pop('joints')
    cano_param.pop('vertices')
    cano_param.pop('faces')
    cano_param = {k: torch.from_numpy(v).float() for k, v in cano_param.items()}
    # cano_output = body_model(betas=cano_param['betas'], transl=cano_param['transl'])
    # cano_pcd = cano_output.vertices.cpu().numpy().reshape(-1, 3)
    # cano_mesh = trimesh.Trimesh(vertices=cano_pcd, faces=body_model.faces, process=False)
    # cano_mesh.export('cano.ply')

    smpl_params = [join(input_dir, x) for x in sorted(os.listdir(input_dir))]
    parallel_execution(smpl_params, cano_param=cano_param, body_model=body_model, output_dir=output_dir, 
                       action=write_pre_smplx_mesh,
                       print_progress=True,
                       sequential=False)