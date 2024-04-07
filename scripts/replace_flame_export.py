import os
import argparse
from os.path import join
from glob import glob
import numpy as np
import torch
import smplx
import open3d as o3d
import cv2
import pickle
from torchvision.utils import make_grid, save_image

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_dotdict, to_tensor, to_cuda
from easymocap.mytools.camera_utils import read_cameras

from transfer_model.utils import batch_rodrigues, batch_rot2aa, np_mesh_to_o3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/0325data')
    parser.add_argument('--input_dir', type=str, default='transfer-smplx-3d')
    parser.add_argument('--vis_dir', type=str, default='vis-smplx-3d')
    parser.add_argument('--output', type=str, default='smpl_params.pkl')
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()
    
    images_dir = 'images'
    masks_dir = 'masks'
    data_root = args.data_root
    input_dir = args.input_dir
    vis_dir = args.vis_dir
    # output_dir = args.output_dir

    os.makedirs(join(data_root, vis_dir), exist_ok=True)
    # os.makedirs(join(data_root, output_dir), exist_ok=True)

    body_model = smplx.SMPLXLayer(model_path='./models/smplx',
                                  gender='neutral',
                                  use_compressed=False,
                                  use_face_contour=True,
                                  num_betas=10,
                                  num_expression_coeffs=100)
    # body_model = smplx.SMPLX(model_path='./models/smplx',
    #                          gender='neutral',
    #                          use_compressed=False,
    #                          use_face_contour=True,
    #                          use_pca=False, 
    #                          num_pca_comps=45, 
    #                          flat_hand_mean=True,
    #                          num_betas=10,
    #                          num_expression_coeffs=100)
    print(body_model)

    cameras = read_cameras(data_root)
    exp_jaw = torch.load(join(data_root, 'exp_jaw.pt'))
    smplxs = sorted(glob(join(data_root, input_dir, '*.npz')))
    smpl_params = dotdict()
    for f in tqdm(smplxs):
        frame = int(os.path.basename(f).split('.')[0])
        exp = exp_jaw['exp_para'][frame]
        jaw = exp_jaw['jaw_pose'][frame]
        data = load_dotdict(f)
        
        full_pose = data.pop('full_pose')
        joints = data.pop('joints')
        vertices = data.pop('vertices')
        faces = data.pop('faces')
        
        data.leye_pose = batch_rodrigues(torch.from_numpy(data.leye_pose.reshape(1, 3))).reshape(1, 1, 3, 3).numpy()
        data.reye_pose = batch_rodrigues(torch.from_numpy(data.reye_pose.reshape(1, 3))).reshape(1, 1, 3, 3).numpy()
        data.jaw_pose = batch_rodrigues(jaw.reshape(1, 3)).reshape(1, 1, 3, 3).numpy()
        data.expression = exp.reshape(1, 100).numpy()
        # data.global_orient = batch_rot2aa(torch.from_numpy(data.global_orient.reshape(-1, 3, 3))).flatten()[None].numpy()
        # data.body_pose = batch_rot2aa(torch.from_numpy(data.body_pose.reshape(-1, 3, 3))).flatten()[None].numpy()
        # data.left_hand_pose = batch_rot2aa(torch.from_numpy(data.left_hand_pose.reshape(-1, 3, 3))).flatten()[None].numpy()
        # data.right_hand_pose = batch_rot2aa(torch.from_numpy(data.right_hand_pose.reshape(-1, 3, 3))).flatten()[None].numpy()
        # data.leye_pose = data.leye_pose.reshape(1, 3)
        # data.reye_pose = data.reye_pose.reshape(1, 3)
        # data.jaw_pose = jaw.reshape(1, 3).numpy()

        # np.savez_compressed(join(data_root, output_dir, f'{frame:06d}.npz'), **data)
        smpl_params[f'{frame:06d}'] = data

        if args.vis:
            data = to_tensor(data)
            out = body_model(return_full_pose=True, get_skin=True, **data)
            verts = out.vertices[0].detach().cpu().numpy()
            imgs = []
            for k, v in cameras.items():
                img = cv2.imread(join(data_root, images_dir, k, f'{frame:06d}.jpg'))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                msk = cv2.imread(join(data_root, masks_dir, k, f'{frame:06d}.jpg'))
                img = img * np.uint8(msk > 128)
                img = cv2.resize(img, None, fx=0.25, fy=0.25)

                R = v['R']
                T = v['T']
                v_cam = np.matmul(verts, R.transpose()) + T.transpose()

                new_K = v['K'].copy()
                new_K[:2] *= 0.25
                v_img = np.matmul(v_cam / v_cam[..., 2:], new_K.transpose())          

                v_img = np.round(v_img).astype(np.int32)
                v_img[..., 0] = np.clip(v_img[..., 0], 0, img.shape[1] - 1)
                v_img[..., 1] = np.clip(v_img[..., 1], 0, img.shape[0] - 1)

                for v in v_img:
                    img[v[1], v[0]] = np.array([255, 255, 255], dtype=np.uint8)
                
                imgs.append(img)
            
            imgs = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)
            imgs = make_grid(imgs, nrow=6)
            imgs = imgs.float() / 255.0
            save_image(imgs, join(data_root, vis_dir, f'{frame:06d}.jpg'))

    # np.savez_compressed(join(data_root, args.output), **smpl_params)
    with open(join(data_root, args.output), 'wb') as f:
        pickle.dump(smpl_params, f)