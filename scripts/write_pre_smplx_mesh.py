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
                                  num_betas=10,
                                  num_expression_coeffs=100)

    input_dir = join(args.data_root, args.input_dir)
    output_dir = join(args.data_root, args.output_dir)
    
    cano_param = np.load(join(output_dir, 'cano.npz'))
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
    for smpl_param in smpl_params:
        with open(smpl_param) as f:
            smpl_data = json.load(f)['annots'][0]
        for k in smpl_data.keys():
            smpl_data[k] = torch.from_numpy(np.array(smpl_data[k])).float()

        smplx_data = {}
        import ipdb; ipdb.set_trace()      

