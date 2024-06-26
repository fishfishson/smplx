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

from typing import Optional, Dict, Callable
import sys
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from loguru import logger
from .utils import get_vertices_per_edge

from .optimizers import build_optimizer, minimize
from .utils import (
    Tensor, batch_rodrigues, apply_deformation_transfer, batch_rot2aa)
from .losses import build_loss


def summary_closure(gt_vertices, var_dict, body_model, mask_ids=None):
    param_dict = {}
    for key, var in var_dict.items():
        # Decode the axis-angles
        if 'pose' in key or 'orient' in key:
            param_dict[key] = batch_rodrigues(
                var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
        else:
            # Simply pass the variable
            param_dict[key] = var
    body_model_output = body_model(
        return_full_pose=True, get_skin=True, **param_dict)
    # est_vertices = body_model_output['vertices']
    est_vertices = body_model_output.vertices
    if mask_ids is not None:
        est_vertices = est_vertices[:, mask_ids]
        gt_vertices = gt_vertices[:, mask_ids]

    v2v = (est_vertices - gt_vertices).pow(2).sum(dim=-1).sqrt().mean()
    return {
        'Vertex-to-Vertex': v2v * 1000}


def build_model_forward_closure(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
    per_part: bool = True,
    part_key: Optional[str] = None,
    jidx: Optional[int] = None,
    part: Optional[Tensor] = None
) -> Callable:
    if per_part:
        cond = part is not None and part_key is not None and jidx is not None
        assert cond, (
            'When per-part is True, "part", "part_key", "jidx" must not be'
            ' None.'
        )

        def model_forward():
            param_dict = {}
            for key, var in var_dict.items():
                if part_key == key:
                    param_dict[key] = batch_rodrigues(
                        var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
                    param_dict[key][:, jidx] = batch_rodrigues(
                        part.reshape(-1, 3)).reshape(-1, 3, 3)
                else:
                    # Decode the axis-angles
                    if 'pose' in key or 'orient' in key:
                        param_dict[key] = batch_rodrigues(
                            var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
                    else:
                        # Simply pass the variable
                        param_dict[key] = var

            return body_model(
                return_full_pose=True, get_skin=True, **param_dict)
    else:
        def model_forward():
            param_dict = {}
            for key, var in var_dict.items():
                # Decode the axis-angles
                if 'pose' in key or 'orient' in key:
                    param_dict[key] = batch_rodrigues(
                        var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
                else:
                    # Simply pass the variable
                    param_dict[key] = var

            return body_model(return_full_pose=True, get_skin=True,
                              **param_dict)
    return model_forward


def build_edge_closure(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
    edge_loss: nn.Module,
    optimizer_dict,
    gt_vertices: Tensor,
    per_part: bool = True,
    part_key: Optional[str] = None,
    jidx: Optional[int] = None,
    part: Optional[Tensor] = None,
    params_to_opt: Optional[Tensor] = None,
) -> Callable:
    ''' Builds the closure for the edge objective
    '''
    optimizer = optimizer_dict['optimizer']
    create_graph = optimizer_dict['create_graph']

    if per_part:
        params_to_opt = [part]
    elif params_to_opt is None:
        params_to_opt = [p for key, p in var_dict.items() if 'pose' in key]

    model_forward = build_model_forward_closure(
        body_model, var_dict, per_part=per_part, part_key=part_key,
        jidx=jidx, part=part)

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = model_forward()
        # est_vertices = body_model_output['vertices']
        est_vertices = body_model_output.vertices

        loss = edge_loss(est_vertices, gt_vertices)
        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(
                    loss, params_to_opt, create_graph=True)
                torch.autograd.backward(
                    params_to_opt, grads, create_graph=True)
            else:
                loss.backward()

        return loss
    return closure


def build_vertex_closure(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
    optimizer_dict,
    gt_vertices: Tensor,
    vertex_loss: nn.Module,
    mask_ids=None,
    per_part: bool = True,
    part_key: Optional[str] = None,
    jidx: Optional[int] = None,
    part: Optional[Tensor] = None,
    params_to_opt: Optional[Tensor] = None,
) -> Callable:
    ''' Builds the closure for the vertex objective
    '''
    optimizer = optimizer_dict['optimizer']
    create_graph = optimizer_dict['create_graph']

    model_forward = build_model_forward_closure(
        body_model, var_dict, per_part=per_part, part_key=part_key,
        jidx=jidx, part=part)

    if params_to_opt is None:
        params_to_opt = [p for key, p in var_dict.items()]

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = model_forward()
        # est_vertices = body_model_output['vertices']
        est_vertices = body_model_output.vertices

        loss = vertex_loss(
            est_vertices[:, mask_ids] if mask_ids is not None else
            est_vertices,
            gt_vertices[:, mask_ids] if mask_ids is not None else gt_vertices)
        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(
                    loss, params_to_opt, create_graph=True)
                torch.autograd.backward(
                    params_to_opt, grads, create_graph=True)
            else:
                loss.backward()

        return loss
    return closure


def get_variables(
    batch_size: int,
    body_model: nn.Module,
    dtype: torch.dtype = torch.float32
) -> Dict[str, Tensor]:
    var_dict = {}

    device = next(body_model.buffers()).device

    if (body_model.name() == 'SMPL' or body_model.name() == 'SMPL+H' or
            body_model.name() == 'SMPL-X'):
        var_dict.update({
            'transl': torch.zeros(
                [batch_size, 3], device=device, dtype=dtype),
            'global_orient': torch.zeros(
                [batch_size, 1, 3], device=device, dtype=dtype),
            'body_pose': torch.zeros(
                [batch_size, body_model.NUM_BODY_JOINTS, 3],
                device=device, dtype=dtype),
            'betas': torch.zeros([batch_size, body_model.num_betas],
                                 dtype=dtype, device=device),
        })

    if body_model.name() == 'SMPL+H' or body_model.name() == 'SMPL-X':
        var_dict.update(
            left_hand_pose=torch.zeros(
                [batch_size, body_model.NUM_HAND_JOINTS, 3], device=device,
                dtype=dtype),
            right_hand_pose=torch.zeros(
                [batch_size, body_model.NUM_HAND_JOINTS, 3], device=device,
                dtype=dtype),
        )

    if body_model.name() == 'SMPL-X':
        var_dict.update(
            jaw_pose=torch.zeros([batch_size, 1, 3],
                                 device=device, dtype=dtype),
            leye_pose=torch.zeros([batch_size, 1, 3],
                                  device=device, dtype=dtype),
            reye_pose=torch.zeros([batch_size, 1, 3],
                                  device=device, dtype=dtype),
            expression=torch.zeros(
                [batch_size, body_model.num_expression_coeffs],
                device=device, dtype=dtype),
        )

    # Toggle gradients to True
    for key, val in var_dict.items():
        val.requires_grad_(True)

    return var_dict


def run_fitting(
    exp_cfg,
    batch: Dict[str, Tensor],
    body_model: nn.Module,
    def_matrix: Tensor,
    mask_ids: None,
    **kwargs,
) -> Dict[str, Tensor]:
    ''' Runs fitting
    '''
    vertices = batch['vertices']
    faces = batch['faces']

    batch_size = len(vertices)
    dtype, device = vertices.dtype, vertices.device
    summary_steps = exp_cfg.get('summary_steps')
    interactive = exp_cfg.get('interactive')

    # Get the parameters from the model
    var_dict = get_variables(batch_size, body_model)

    # Build the optimizer object for the current batch
    optim_cfg = exp_cfg.get('optim', {})
    if optim_cfg.shape_only:
        for k in var_dict.keys():
            if k == 'betas' or k == 'transl': continue
            else: var_dict[k].requires_grad_(False)
    else:
        if 'betas' in kwargs:
            var_dict['betas'].requires_grad_(False)
            var_dict['betas'][:] = kwargs['betas']
        if 'expression' in kwargs:
            var_dict['expression'].requires_grad_(False)
            var_dict['expression'][:] = kwargs['expression']
        if 'jaw_pose' in kwargs:
            var_dict['jaw_pose'].requires_grad_(False)
            var_dict['jaw_pose'][:] = batch_rot2aa(kwargs['jaw_pose'].reshape(-1, 3, 3)).reshape(batch_size, -1, 3)
        if 'leye_pose' in kwargs:
            var_dict['leye_pose'].requires_grad_(False)
            var_dict['leye_pose'][:] = batch_rot2aa(kwargs['leye_pose'].reshape(-1, 3, 3)).reshape(batch_size, -1, 3)
        if 'reye_pose' in kwargs:
            var_dict['reye_pose'].requires_grad_(False)
            var_dict['reye_pose'][:] = batch_rot2aa(kwargs['reye_pose'].reshape(-1, 3, 3)).reshape(batch_size, -1, 3)
        if 'global_orient' in kwargs:
            var_dict['global_orient'].requires_grad_(False)
            var_dict['global_orient'][:] = batch_rot2aa(kwargs['global_orient'].reshape(-1, 3, 3)).reshape(batch_size, -1, 3)
        if 'transl' in kwargs:
            var_dict['transl'].requires_grad_(False)
            var_dict['transl'][:] = kwargs['transl']
        if 'body_pose' in kwargs:
            body_pose = batch_rot2aa(kwargs['body_pose'].reshape(-1, 3, 3)).reshape(batch_size, -1, 3)
            var_dict['body_pose'].requires_grad_(False)
            var_dict['body_pose'][:] = body_pose
            var_dict['body_pose'].requires_grad_(True)
        if 'left_hand_pose' in kwargs:
            left_hand_pose = batch_rot2aa(kwargs['left_hand_pose'].reshape(-1, 3, 3)).reshape(batch_size, -1, 3)
            var_dict['left_hand_pose'].requires_grad_(False)
            var_dict['left_hand_pose'][:] = left_hand_pose
            var_dict['left_hand_pose'].requires_grad_(True)
        if 'right_hand_pose' in kwargs:
            right_hand_pose = batch_rot2aa(kwargs['right_hand_pose'].reshape(-1, 3, 3)).reshape(batch_size, -1, 3)
            var_dict['right_hand_pose'].requires_grad_(False)
            var_dict['right_hand_pose'][:] = right_hand_pose
            var_dict['right_hand_pose'].requires_grad_(True)

    def_vertices = apply_deformation_transfer(def_matrix, vertices, faces)
    if optim_cfg.mesh_only:
        mesh = {
            'vertices': def_vertices,
            'faces': body_model.faces
        }
        return mesh

    if mask_ids is None:
        f_sel = np.ones_like(body_model.faces[:, 0], dtype=np.bool_)
    else:
        f_per_v = [[] for _ in range(body_model.get_num_verts())]
        [f_per_v[vv].append(iff) for iff, ff in enumerate(body_model.faces)
         for vv in ff]
        f_sel = list(set(tuple(sum([f_per_v[vv] for vv in mask_ids], []))))
    vpe = get_vertices_per_edge(
        body_model.v_template.detach().cpu().numpy(), body_model.faces[f_sel])

    def log_closure():
        return summary_closure(def_vertices, var_dict, body_model,
                               mask_ids=mask_ids)

    edge_fitting_cfg = exp_cfg.get('edge_fitting', {})
    edge_loss = build_loss(type='vertex-edge', gt_edges=vpe, est_edges=vpe,
                           **edge_fitting_cfg)
    edge_loss = edge_loss.to(device=device)

    vertex_fitting_cfg = exp_cfg.get('vertex_fitting', {})
    vertex_loss = build_loss(**vertex_fitting_cfg)
    vertex_loss = vertex_loss.to(device=device)

    per_part = edge_fitting_cfg.get('per_part', True)
    logger.info(f'Per-part: {per_part}')
    # Optimize edge-based loss to initialize pose
    if per_part:
        for key, var in tqdm(var_dict.items(), desc='Parts'):
            if 'pose' not in key:
                continue

            for jidx in tqdm(range(var.shape[1]), desc='Joints'):
                part = torch.zeros(
                    [batch_size, 3], dtype=dtype, device=device,
                    requires_grad=True)
                # Build the optimizer for the current part
                optimizer_dict = build_optimizer([part], optim_cfg)
                closure = build_edge_closure(
                    body_model, var_dict, edge_loss, optimizer_dict,
                    def_vertices, per_part=per_part, part_key=key, jidx=jidx,
                    part=part)

                minimize(optimizer_dict['optimizer'], closure,
                         params=[part],
                         summary_closure=log_closure,
                         summary_steps=summary_steps,
                         interactive=interactive,
                         **optim_cfg)
                with torch.no_grad():
                    var[:, jidx] = part
    else:
        params = []
        for k, v in var_dict.items():
            if 'pose' in k and v.requires_grad: params.append(v)
        if len(params) > 0:
            optimizer_dict = build_optimizer(params, optim_cfg)
            closure = build_edge_closure(
                body_model, var_dict, edge_loss, optimizer_dict,
                def_vertices, per_part=per_part, params_to_opt=params)

            minimize(optimizer_dict['optimizer'], closure,
                    params=params,
                    summary_closure=log_closure,
                    summary_steps=summary_steps,
                    interactive=interactive,
                    **optim_cfg)

    if 'translation' in var_dict and var_dict['translation'].requires_grad:
        optimizer_dict = build_optimizer([var_dict['translation']], optim_cfg)
        closure = build_vertex_closure(
            body_model, var_dict,
            optimizer_dict,
            def_vertices,
            vertex_loss=vertex_loss,
            mask_ids=mask_ids,
            per_part=False,
            params_to_opt=[var_dict['translation']],
        )
        # Optimize translation
        minimize(optimizer_dict['optimizer'],
                 closure,
                 params=[var_dict['translation']],
                 summary_closure=log_closure,
                 summary_steps=summary_steps,
                 interactive=interactive,
                 **optim_cfg)

    #  Optimize all model parameters with vertex-based loss
    params = [x for x in list(var_dict.values()) if x.requires_grad]
    optimizer_dict = build_optimizer(params, optim_cfg)
    closure = build_vertex_closure(
        body_model, var_dict,
        optimizer_dict,
        def_vertices,
        vertex_loss=vertex_loss,
        per_part=False,
        mask_ids=mask_ids,
        params_to_opt=params)
    loss = minimize(optimizer_dict['optimizer'], closure,
             params=params,
             summary_closure=log_closure,
             summary_steps=summary_steps,
             interactive=interactive,
             **optim_cfg)

    param_dict = {}
    for key, var in var_dict.items():
        # Decode the axis-angles
        if 'pose' in key or 'orient' in key:
            param_dict[key] = batch_rodrigues(
                var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
        else:
            # Simply pass the variable
            param_dict[key] = var

    body_model_output = body_model(
        return_full_pose=True, get_skin=True, **param_dict)
    body_model_output_dict = {
        'betas': body_model_output.betas,
        'body_pose': body_model_output.body_pose,
        'expression': body_model_output.expression,
        'full_pose': body_model_output.full_pose,
        'global_orient': body_model_output.global_orient,
        'jaw_pose': body_model_output.jaw_pose,
        'joints': body_model_output.joints,
        'left_hand_pose': body_model_output.left_hand_pose,
        'right_hand_pose': body_model_output.right_hand_pose,
        'transl': body_model_output.transl,
        'vertices': body_model_output.vertices,
    }
    var_dict.update(body_model_output_dict)
    var_dict['faces'] = body_model.faces
    var_dict['loss'] = loss
    return var_dict
