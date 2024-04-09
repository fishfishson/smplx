import os
import argparse
import numpy as np
import torch
import smplx
import trimesh
import json
from transfer_model.utils import batch_rodrigues, batch_rot2aa, np_mesh_to_o3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('smplx_path', type=str)
    parser.add_argument('smplh_path', type=str)
    parser.add_argument('--model_path', type=str, default='./models/smplx') 
    args = parser.parse_args()

    with open(args.smplh_path) as f:
        smplh_data = json.load(f)['annots'][0]
    Rh = batch_rodrigues(torch.from_numpy(np.array(smplh_data['Rh']))).float()
    Th = torch.from_numpy(np.array(smplh_data['Th'])).float()
    
    body_model = smplx.SMPLXLayer(model_path=args.model_path,
                                  gender='neutral',
                                  use_compressed=False,
                                  use_face_contour=True,
                                  num_betas=16,
                                  num_expression_coeffs=100)
    
    smplx_data = np.load(args.smplx_path)
    smplx_data = {k: v for k, v in smplx_data.items()}
    smplx_data.pop('full_pose')
    smplx_data.pop('joints')
    smplx_data.pop('vertices')
    smplx_data.pop('faces')
    smplx_data.pop('leye_pose')
    smplx_data.pop('reye_pose')
    smplx_data.pop('jaw_pose')
    smplx_data.pop('expression')
    smplx_data = {k: torch.from_numpy(v).float() for k, v in smplx_data.items()}

    smplx_output = body_model(**smplx_data)
    smplx_pcd = smplx_output.vertices
    smplx_pcd = (Rh @ smplx_pcd.mT + Th[:, :, None]).mT
    smplx_pcd = smplx_pcd.cpu().numpy().reshape(-1, 3)
    smplx_mesh = trimesh.Trimesh(vertices=smplx_pcd, faces=body_model.faces, process=False)
    smplx_mesh.export('smplx.ply')
