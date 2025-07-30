# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling RVT or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# from pytorch3d import transforms as torch3d_tf
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary

from ...helpers import utils
from ...helpers.utils import visualise_voxel, stack_on_channel
from ...voxel.voxel_grid import VoxelGrid
from ...voxel.augmentation import apply_se3_augmentation
from einops import rearrange
from ...helpers.clip.core.clip import build_model, load_clip

import transformers
from ...helpers.optim.lamb import Lamb
from ...helpers.utils import point_to_voxel_index_tensor_batched

from torch.nn.parallel import DistributedDataParallel as DDP
from bridgevla.mvt.utils import plot_pcd, plot_voxel_grid_with_action, plot_voxel_grid_with_action_cubes
NAME = 'QAttentionAgent'


class QFunction(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 voxelizer: VoxelGrid,
                 bounds_offset: float,
                 rotation_resolution: float,
                 device,
                 training):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)

        # distributed training
        if training:
            self._qnet = DDP(self._qnet, device_ids=[device], find_unused_parameters=True)

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self, rgb_pcd, proprio, pcd, 
                # lang_goal_emb, lang_token_embs,
                bounds=None, prev_bounds=None, prev_layer_voxel_grid=None):
        # rgb_pcd will be list of list (list of [rgb, pcd])
        b = rgb_pcd[0][0].shape[0]

        # Combine all the point clouds into a single tensor: (B, N, 3)
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        # Combine all the RGBs into a single tensor: (B, N, 3)
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)

        # construct voxel grid (bounds are [-x, -y, -z, x, y, z])
        voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)
        # voxel_grid => (batch, X, Y, Z, features) => (b, 64, 64, 64, 10)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()
        # voxel_grid => (batch, features, X, Y, Z) => (b, 10, 64, 64, 64)

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass (_qnet is the Perceiver Encoder)
        q_trans, \
        q_rot_and_grip,\
        q_ignore_collisions = self._qnet(voxel_grid, proprio)   # This is the Perceiver Encoder

        return q_trans, q_rot_and_grip, q_ignore_collisions, voxel_grid


class QAttentionPerActBCAgent(Agent):

    def __init__(self,
                 layer: int,
                 coordinate_bounds: list,
                 perceiver_encoder: nn.Module,
                 camera_names: list,
                 batch_size: int,
                 voxel_size: int,
                 bounds_offset: float,
                 voxel_feature_size: int,
                 image_crop_size: int,
                 num_rotation_classes: int,
                 rotation_resolution: float,
                 lr: float = 0.0001,
                 lr_scheduler: bool = False,
                 training_iterations: int = 100000,
                 num_warmup_steps: int = 20000,
                 trans_loss_weight: float = 1.0,
                 rot_loss_weight: float = 1.0,
                 grip_loss_weight: float = 1.0,
                 collision_loss_weight: float = 1.0,
                 include_low_dim_state: bool = False,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 0.0,
                 transform_augmentation: bool = True,
                 transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                 transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                 transform_augmentation_rot_resolution: int = 5,
                 optimizer_type: str = 'adam',
                 num_devices: int = 1,
                 ):
        self._layer = layer
        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(np.array(transform_augmentation_xyz))
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type
        self._num_devices = num_devices
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._name = NAME + '_layer' + str(self._layer)

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if device is None:
            device = torch.device('cpu')

        # !!! Voxelize here
        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        # !!! This is the main Q-function that predicts the action
        self._q = QFunction(self._perceiver_encoder,
                            self._voxelizer,
                            self._bounds_offset,
                            self._rotation_resolution,
                            device,
                            training).to(device).train(training)

        grid_for_crop = torch.arange(0,
                                     self._image_crop_size,
                                     device=device).unsqueeze(0).repeat(self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0),
                                         grid_for_crop], dim=2).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._training:
            # optimizer
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception('Unknown optimizer type')

            # learning rate scheduler
            if self._lr_scheduler:
                self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self._optimizer,
                    num_warmup_steps=self._num_warmup_steps,
                    num_training_steps=self._training_iterations,
                    num_cycles=self._training_iterations // 10000,
                )

            # one-hot zero tensors
            self._action_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                            1,
                                                            self._voxel_size,
                                                            self._voxel_size,
                                                            self._voxel_size),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_x_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_y_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_z_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                           2),
                                                           dtype=int,
                                                           device=device)
            self._action_ignore_collisions_one_hot_zeros = torch.zeros((self._batch_size,
                                                                        2),
                                                                        dtype=int,
                                                                        device=device)

            # print total params
            logging.info('# Q Params: %d' % sum(
                p.numel() for name, p in self._q.named_parameters() \
                if p.requires_grad and 'clip' not in name))
        else:
            for param in self._q.parameters():
                param.requires_grad = False

            # load CLIP for encoding language goals during evaluation
            model, _ = load_clip("RN50", jit=False)
            self._clip_rn50 = build_model(model.state_dict())
            self._clip_rn50 = self._clip_rn50.float().to(device)
            self._clip_rn50.eval()
            del model

            self._voxelizer.to(device)
            self._q.to(device)

    def _extract_crop(self, pixel_action, observation):
        # Pixel action will now be (B, 2)
        # observation = stack_on_channel(observation)
        h = observation.shape[-1]
        top_left_corner = torch.clamp(
            pixel_action - self._image_crop_size // 2, 0,
            h - self._image_crop_size)
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0  # between -1 and 1
        # Used for cropping the images across a batch
        # swap fro y x, to x, y
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode='nearest',
                             align_corners=True)
        return crop

    def _preprocess_inputs(self, replay_sample):
        obs = []
        pcds = []
        self._crop_summary = []
        for n in self._camera_names:
            rgb = replay_sample[n]['rgb']
            pcd = replay_sample[n]['pcd']    

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _act_preprocess_inputs(self, observation):
        obs, pcds = [], []
        for n in self._camera_names:
            rgb = observation['%s_rgb' % n]
            pcd = observation['%s_point_cloud' % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _get_value_from_voxel_index(self, q, voxel_idx):
        b, c, d, h, w = q.shape
        q_trans_flat = q.view(b, c, d * h * w)
        flat_indicies = (voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2])[:, None].int()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_trans_flat.gather(2, highest_idxs)[..., 0]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        q_rot = torch.stack(torch.split(
            rot_grip_q[:, :-2], int(360 // self._rotation_resolution),
            dim=1), dim=1)  # B, 3, 72
        q_grip = rot_grip_q[:, -2:]
        rot_and_grip_values = torch.cat(
            [q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
             q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
             q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
             q_grip.gather(1, rot_and_grip_idx[:, 3:4])], -1)
        return rot_and_grip_values

    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes: 1*self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes: 2*self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes: 3*self._num_rotation_classes]
        q_grip_flat  = q_rot_grip[:, 3*self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([q_rot_x_flat_softmax,
                          q_rot_y_flat_softmax,
                          q_rot_z_flat_softmax,
                          q_grip_flat_softmax], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def update(self, step, replay_sample: dict) -> dict:

        # NOTE: This class does not have any neural network layers, 
        # it just uses the Perceiver Encoder to predict the action.
        # So, The Main Network is the Perceiver Encoder !!!
        
        # Main (x,y,z) action of the voxel grid. Assumes that the layers are listed sequentially along the 2nd axis, so we are indexing the action for this layer
        action_trans = replay_sample['gripper_pose'][:, :3]

        # Convert action_trans to voxel grid frame:
        action_trans_voxel = point_to_voxel_index_tensor_batched(
            action_trans, 
            self._voxel_size, 
            self._coordinate_bounds
        ).int()
        
        action_rot_grip = replay_sample['gripper_pose'][:, 3:6]
        
        # Convert action_rot_grip to bins
        
        action_gripper_pose = replay_sample['gripper_pose']
        
        
        action_ignore_collisions = replay_sample['ignore_collisions'].int()
        # lang_goal_emb = replay_sample['lang_goal_emb'].float()
        # lang_token_embs = replay_sample['lang_token_embs'].float()
        # prev_layer_voxel_grid = replay_sample.get('prev_layer_voxel_grid', None)
        # prev_layer_bounds = replay_sample.get('prev_layer_bounds', None)
        device = self._device

        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            cp = replay_sample['attention_coordinate_layer_%d' % (self._layer - 1)]
            bounds = torch.cat([cp - self._bounds_offset, cp + self._bounds_offset], dim=1)

        proprio = None
        if self._include_low_dim_state:
            proprio = replay_sample['low_dim_state']

        obs, pcd = self._preprocess_inputs(replay_sample)
        # obs is a list of each element as [rgb, pcd], pcd is a list of each element as just pcd

        # batch size
        bs = pcd[0].shape[0]

        # SE(3) augmentation of point clouds and actions
        # !!! Turning off for now to understand code, look through this later (we do need 3D augmentations)
        if self._transform_augmentation:
            action_trans, \
            action_rot_grip, \
            pcd = apply_se3_augmentation(pcd,
                                         action_gripper_pose,
                                         action_trans,
                                         action_rot_grip,
                                         bounds,
                                         self._layer,
                                         self._transform_augmentation_xyz,
                                         self._transform_augmentation_rpy,
                                         self._transform_augmentation_rot_resolution,
                                         self._voxel_size,
                                         self._rotation_resolution,
                                         self._device)

        # forward pass
        q_trans, q_rot_grip, \
        q_collision, \
        voxel_grid = self._q(obs,
                             proprio,
                             pcd,
                            #  lang_goal_emb,
                            #  lang_token_embs,
                             bounds,
                            #  prev_layer_bounds,
                            #  prev_layer_voxel_grid
                            )

        # argmax to choose best action
        coords, \
        rot_and_grip_indicies, \
        ignore_collision_indicies = self._q.choose_highest_action(q_trans, q_rot_grip, q_collision)


        # !!! --------- Debugging : Plot Voxel Grid --------- #
        
        # vis_b = 0
        # vis_grid = voxel_grid[vis_b].permute(1, 2, 3, 0)
        # mask = torch.norm(vis_grid[..., :3], dim = -1) > 0      # Could just use occupancy instead...

        # vis_pts = vis_grid[torch.where(mask)][..., :3]
        # vis_pts = torch.cat([vis_pts, action_trans[0][None]], dim=0)

        # vis_rgb = vis_grid[torch.where(mask)][..., 3:6]
        # vis_rgb_voxel = vis_rgb.clone()
        # vis_rgb = torch.cat([vis_rgb, torch.tensor([[255, 0, 0]]).float().to(vis_rgb.device)], dim=0)

        # vis_pts_voxel = vis_grid[torch.where(mask)][..., 6:9]
        # action_voxel_center = vis_grid[action_trans_voxel[vis_b, 0], action_trans_voxel[vis_b, 1], action_trans_voxel[vis_b, 2], 6:9]
        # vis_pts_voxel = torch.cat([vis_pts_voxel, action_voxel_center[None]], dim=0)
        # vis_rgb_voxel = torch.cat([vis_rgb_voxel, torch.tensor([[255, 0, 0]]).float().to(vis_rgb.device)], dim=0)
        
        # print(vis_pts.shape, vis_rgb.shape)
        # plot_pcd(vis_pts, vis_rgb)

        # --------------------------------------------------- #

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0., 0., 0., 0.

        # Preparing a one hot voxel grid for supervising the translation action:
        action_trans_one_hot = self._action_trans_one_hot_zeros.clone()
        for b in range(bs):
            gt_coord = action_trans[b, :].int()
            action_trans_one_hot[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

        # Flatten both the prediction and the ground truth, and apply cross entropy loss
        q_trans_flat = q_trans.view(bs, -1)
        action_trans_one_hot_flat = action_trans_one_hot.view(bs, -1)
        q_trans_loss = self._celoss(q_trans_flat, action_trans_one_hot_flat)

        # with_rot_and_grip = rot_and_grip_indicies is not None
        with_rot_and_grip = False
        if with_rot_and_grip:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone()

            for b in range(bs):
                gt_rot_grip = action_rot_grip[b, :].int()
                action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
                action_grip_one_hot[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes:1*self._num_rotation_classes]
            q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes:2*self._num_rotation_classes]
            q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes:3*self._num_rotation_classes]
            q_grip_flat =  q_rot_grip[:, 3*self._num_rotation_classes:]
            q_ignore_collisions_flat = q_collision

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat, action_rot_x_one_hot)
            q_rot_loss += self._celoss(q_rot_y_flat, action_rot_y_one_hot)
            q_rot_loss += self._celoss(q_rot_z_flat, action_rot_z_one_hot)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat, action_grip_one_hot)

            # collision loss
            q_collision_loss += self._celoss(q_ignore_collisions_flat, action_ignore_collisions_one_hot)

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight)
        total_loss = combined_losses.mean()

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        self._summaries = {
            'losses/total_loss': total_loss,
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip else 0.,
            'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip else 0.,
            'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip else 0.,
        }

        if self._lr_scheduler:
            self._scheduler.step()
            self._summaries['learning_rate'] = self._scheduler.get_last_lr()[0]

        self._vis_voxel_grid = voxel_grid[0]
        self._vis_translation_qvalue = self._softmax_q_trans(q_trans[0])
        self._vis_max_coordinate = coords[0]
        self._vis_gt_coordinate = action_trans[0]

        # Note: PerAct doesn't use multi-layer voxel grids like C2FARM
        # stack prev_layer_voxel_grid(s) from previous layers into a list
        # if prev_layer_voxel_grid is None:
        #     prev_layer_voxel_grid = [voxel_grid]
        # else:
        #     prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        # # stack prev_layer_bound(s) from previous layers into a list
        # if prev_layer_bounds is None:
        #     prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        # else:
        #     prev_layer_bounds = prev_layer_bounds + [bounds]

        if step % 99 == 0:
            # Visualization:
            action_voxels = torch.cat([coords[0][None], action_trans_voxel[0][None]], dim=0)
            action_colors = torch.zeros_like(action_voxels, dtype=torch.float32)
            action_colors[0, 0] = 255.
            action_colors[1, 1] = 255.
            # plot_voxel_grid_with_action(voxel_grid[0], action_voxels, action_colors)

            plot_voxel_grid_with_action_cubes(voxel_grid[0], action_voxels, action_colors)

        return {
            'total_loss': total_loss
        }


    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        deterministic = True
        bounds = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = observation.get('prev_layer_bounds', None)
        lang_goal_tokens = observation.get('lang_goal_tokens', None).long()

        # extract CLIP language embs
        with torch.no_grad():
            lang_goal_tokens = lang_goal_tokens.to(device=self._device)
            lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        max_rot_index = int(360 // self._rotation_resolution)
        proprio = None

        if self._include_low_dim_state:
            proprio = observation['low_dim_state']

        obs, pcd = self._act_preprocess_inputs(observation)     # !!! This is the last use of observation, just returns a dictionary of rgbs and pcds from each camera

        # correct batch size and device
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)] for o in obs]
        proprio = proprio[0].to(self._device)
        pcd = [p[0].to(self._device) for p in pcd]
        lang_goal_emb = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds = torch.as_tensor(bounds, device=self._device)
        prev_layer_voxel_grid = prev_layer_voxel_grid.to(self._device) if prev_layer_voxel_grid is not None else None
        prev_layer_bounds = prev_layer_bounds.to(self._device) if prev_layer_bounds is not None else None

        # inference
        q_trans, \
        q_rot_grip, \
        q_ignore_collisions, \
        vox_grid = self._q(obs,
                           proprio,
                           pcd,
                           lang_goal_emb,
                           lang_token_embs,
                           bounds,
                           prev_layer_bounds,
                           prev_layer_voxel_grid)

        # softmax Q predictions
        q_trans = self._softmax_q_trans(q_trans)
        q_rot_grip =  self._softmax_q_rot_grip(q_rot_grip) if q_rot_grip is not None else q_rot_grip
        q_ignore_collisions = self._softmax_ignore_collision(q_ignore_collisions) \
            if q_ignore_collisions is not None else q_ignore_collisions

        # argmax Q predictions
        coords, \
        rot_and_grip_indicies, \
        ignore_collisions = self._q.choose_highest_action(q_trans, q_rot_grip, q_ignore_collisions)

        rot_grip_action = rot_and_grip_indicies if q_rot_grip is not None else None
        ignore_collisions_action = ignore_collisions.int() if ignore_collisions is not None else None

        coords = coords.int()
        attention_coordinate = bounds[:, :3] + res * coords + res / 2

        # stack prev_layer_voxel_grid(s) into a list
        # NOTE: PerAct doesn't used multi-layer voxel grids like C2FARM
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [vox_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [vox_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        observation_elements = {
            'attention_coordinate': attention_coordinate,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }
        info = {
            'voxel_grid_depth%d' % self._layer: vox_grid,
            'q_depth%d' % self._layer: q_trans,
            'voxel_idx_depth%d' % self._layer: coords
        }
        self._act_voxel_grid = vox_grid[0]
        self._act_max_coordinate = coords[0]
        self._act_qvalues = q_trans[0].detach()
        return ActResult((coords, rot_grip_action, ignore_collisions_action),
                         observation_elements=observation_elements,
                         info=info)

    def update_summaries(self) -> List[Summary]:
        summaries = [
            ImageSummary('%s/update_qattention' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._vis_voxel_grid.detach().cpu().numpy(),
                             self._vis_translation_qvalue.detach().cpu().numpy(),
                             self._vis_max_coordinate.detach().cpu().numpy(),
                             self._vis_gt_coordinate.detach().cpu().numpy())))
        ]

        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))

        for (name, crop) in (self._crop_summary):
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            summaries.extend([
                ImageSummary('%s/crops/%s' % (self._name, name), crops)])

        for tag, param in self._q.named_parameters():
            # assert not torch.isnan(param.grad.abs() <= 1.0).all()
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (self._name, tag),
                                 param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (self._name, tag),
                                 param.data))

        return summaries

    def act_summaries(self) -> List[Summary]:
        return [
            ImageSummary('%s/act_Qattention' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._act_voxel_grid.cpu().numpy(),
                             self._act_qvalues.cpu().numpy(),
                             self._act_max_coordinate.cpu().numpy())))]

    def load_weights(self, savedir: str):
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        state_dict = torch.load(weight_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if '_voxelizer' in k:
                continue
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    logging.warning("key %s not found in checkpoint" % k)
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % weight_file)

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % self._name))
