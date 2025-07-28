from bridgevla.libs.peract.agents.peract_bc.perceiver_lang_io import PerceiverVoxelLangEncoder
from bridgevla.libs.peract.agents.peract_bc.qattention_peract_bc_agent import QAttentionPerActBCAgent


class PerActAgent:
    def __init__(self):
        
        self.perceiver_encoder = PerceiverVoxelLangEncoder(
            depth=6,                           # cfg.method.transformer_depth
            iterations=1,                      # cfg.method.transformer_iterations  
            voxel_size=100,                    # cfg.method.voxel_sizes[0] (first element)
            initial_dim=10,                    # 3 + 3 + 1 + 3 (hardcoded)
            low_dim_size=4,                    # hardcoded
            layer=0,                       # depth variable
            num_rotation_classes=72,           # 360/cfg.method.rotation_resolution (360/5=72)
            num_grip_classes=2,                # hardcoded
            num_collision_classes=2,           # hardcoded
            input_axis=3,                      # hardcoded
            num_latents=2048,                  # cfg.method.num_latents
            latent_dim=512,                    # cfg.method.latent_dim
            cross_heads=1,                     # cfg.method.cross_heads
            latent_heads=8,                    # cfg.method.latent_heads
            cross_dim_head=64,                 # cfg.method.cross_dim_head
            latent_dim_head=64,                # cfg.method.latent_dim_head
            weight_tie_layers=False,           # hardcoded
            activation='lrelu',                # cfg.method.activation
            pos_encoding_with_lang=False,      # cfg.method.pos_encoding_with_lang
            input_dropout=0.1,                 # cfg.method.input_dropout
            attn_dropout=0.1,                  # cfg.method.attn_dropout
            decoder_dropout=0.0,               # cfg.method.decoder_dropout
            lang_fusion_type='seq',            # cfg.method.lang_fusion_type
            voxel_patch_size=5,                # cfg.method.voxel_patch_size
            voxel_patch_stride=5,              # cfg.method.voxel_patch_stride
            no_skip_connection=False,          # cfg.method.no_skip_connection
            no_perceiver=False,                # cfg.method.no_perceiver
            no_language=False,                 # cfg.method.no_language
            final_dim=64,                      # cfg.method.final_dim
        )

        self.qattention_agent = QAttentionPerActBCAgent(
            layer=0,                                                    # hardcoded
            coordinate_bounds=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],        # cfg.rlbench.scene_bounds
            perceiver_encoder=self.perceiver_encoder,                        # from above
            camera_names=['front', 'left_shoulder', 'right_shoulder', 'wrist'],  # cfg.rlbench.cameras
            voxel_size=100,                                             # cfg.method.voxel_sizes[0]
            bounds_offset=0.15,                                         # cfg.method.bounds_offset[0]
            image_crop_size=64,                                         # cfg.method.image_crop_size
            lr=0.0005,                                                  # cfg.method.lr
            training_iterations=40000,                                  # cfg.framework.training_iterations
            lr_scheduler=False,                                         # cfg.method.lr_scheduler
            num_warmup_steps=3000,                                      # cfg.method.num_warmup_steps
            trans_loss_weight=1.0,                                      # cfg.method.trans_loss_weight
            rot_loss_weight=1.0,                                        # cfg.method.rot_loss_weight
            grip_loss_weight=1.0,                                       # cfg.method.grip_loss_weight
            collision_loss_weight=1.0,                                  # cfg.method.collision_loss_weight
            include_low_dim_state=True,                                 # hardcoded
            image_resolution=[128, 128],                                # cfg.rlbench.camera_resolution
            batch_size=8,                                               # cfg.replay.batch_size
            voxel_feature_size=3,                                       # hardcoded
            lambda_weight_l2=1.0e-06,                                   # cfg.method.lambda_weight_l2
            num_rotation_classes=72,                                    # 360/cfg.method.rotation_resolution
            rotation_resolution=5,                                      # cfg.method.rotation_resolution
            transform_augmentation=True,                                # cfg.method.transform_augmentation.apply_se3
            transform_augmentation_xyz=[0.125, 0.125, 0.125],          # cfg.method.transform_augmentation.aug_xyz
            transform_augmentation_rpy=[0.0, 0.0, 0.0],                # cfg.method.transform_augmentation.aug_rpy
            transform_augmentation_rot_resolution=5,                    # cfg.method.transform_augmentation.aug_rot_resolution
            optimizer_type='lamb',                                      # cfg.method.optimizer
            num_devices=1,                                              # cfg.ddp.num_devices
        )