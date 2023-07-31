import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import networks
from .geometry.dmtet import DMTetGeometry
from .geometry.skinning import estimate_bones, skinning
from .render import util
from .render import mesh
from .render import light


def lookat_forward_to_rot_matrix(vec_forward, up=[0,1,0]):
    # vec_forward = nn.functional.normalize(vec_forward, p=2, dim=-1)  # x right, y up, z forward  -- assumed normalized
    up = torch.FloatTensor(up).to(vec_forward.device)
    vec_right = up.expand_as(vec_forward).cross(vec_forward, dim=-1)
    vec_right = nn.functional.normalize(vec_right, p=2, dim=-1)
    vec_up = vec_forward.cross(vec_right, dim=-1)
    vec_up = nn.functional.normalize(vec_up, p=2, dim=-1)
    rot_mat = torch.stack([vec_right, vec_up, vec_forward], -2)
    return rot_mat


def sample_pose_hypothesis_from_quad_predictions(poses_raw, total_iter, rot_temp_scalar=1., num_hypos=4, naive_probs_iter=2000, best_pose_start_iter=6000, random_sample=True):
    rots_pred = poses_raw[..., :num_hypos*4].view(-1, num_hypos, 4)  # NxKx4
    N = len(rots_pred)
    rots_logits = rots_pred[..., 0]  # Nx4
    rots_pred = rots_pred[..., 1:4]
    trans_pred = poses_raw[..., -3:]
    temp = 1 / np.clip(total_iter / 1000 / rot_temp_scalar, 1., 100.)

    rots_probs = torch.nn.functional.softmax(-rots_logits / temp, dim=1)  # NxK
    naive_probs = torch.ones(num_hypos).to(rots_logits.device)
    naive_probs = naive_probs / naive_probs.sum()
    naive_probs_weight = np.clip(1 - (total_iter - naive_probs_iter) / 2000, 0, 1)
    rots_probs = naive_probs.view(1, num_hypos) * naive_probs_weight + rots_probs * (1 - naive_probs_weight)
    best_rot_idx = torch.argmax(rots_probs, dim=1)  # N

    if random_sample:
        rand_rot_idx = torch.randperm(N, device=poses_raw.device) % num_hypos  # N
        best_flag = (torch.randperm(N, device=poses_raw.device) / N < np.clip((total_iter - best_pose_start_iter)/2000, 0, 0.8)).long()
        rand_flag = 1 - best_flag
        rot_idx = best_rot_idx * best_flag + rand_rot_idx * (1 - best_flag)
    else:
        rand_flag = torch.zeros_like(best_rot_idx)
        rot_idx = best_rot_idx
    rot_pred = torch.gather(rots_pred, 1, rot_idx[:, None, None].expand(-1, 1, 3))[:, 0]  # Nx3
    pose_raw = torch.cat([rot_pred, trans_pred], -1)
    rot_prob = torch.gather(rots_probs, 1, rot_idx[:, None].expand(-1, 1))[:, 0]  # N
    rot_logit = torch.gather(rots_logits, 1, rot_idx[:, None].expand(-1, 1))[:, 0]  # N

    rot_mat = lookat_forward_to_rot_matrix(rot_pred, up=[0, 1, 0])
    pose = torch.cat([rot_mat.view(N, -1), pose_raw[:, 3:]], -1)  # flattened to Nx12
    pose_aux = {
        'rot_idx': rot_idx,
        'rot_prob': rot_prob,
        'rot_logit': rot_logit,
        'rots_probs': rots_probs,
        'rand_pose_flag': rand_flag
    }
    return pose_raw, pose, pose_aux


class PriorPredictor(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        mlp_hidden_size = cfgs.get('mlp_hidden_size', 64)
        embed_concat_pts = cfgs.get('embed_concat_pts', True)

        dmtet_grid_res = cfgs.get('dmtet_grid_res', 64)
        grid_scale = cfgs.get('grid_scale', 5)
        num_layers_shape = cfgs.get('num_layers_shape', 5)
        embedder_freq_shape = cfgs.get('embedder_freq_shape', 8)
        init_sdf = cfgs.get('init_sdf', None)
        jitter_grid = cfgs.get('jitter_grid', 0.)
        sym_prior_shape = cfgs.get('sym_prior_shape', False)
        self.netShape = DMTetGeometry(
            dmtet_grid_res,
            grid_scale,
            num_layers=num_layers_shape,
            hidden_size=mlp_hidden_size,
            embedder_freq=embedder_freq_shape,
            embed_concat_pts=embed_concat_pts,
            init_sdf=init_sdf,
            jitter_grid=jitter_grid,
            symmetrize=sym_prior_shape
        )

        num_layers_dino = cfgs.get("num_layers_dino", 5)
        dino_feature_recon_dim = cfgs.get('dino_feature_recon_dim', 64)
        dino_minmax = torch.FloatTensor(cfgs.get('dino_minmax', [0., 1.])).repeat(dino_feature_recon_dim, 1)  # Nx2
        embedder_freq_dino = cfgs.get('embedder_freq_dino', 8)
        embedder_scalar = 2 * np.pi / grid_scale * 0.9  # originally (-0.5, 0.5) * grid_scale rescale to (-pi, pi) * 0.9
        sym_dino = cfgs.get("sym_dino", False)
        self.netDINO = networks.CoordMLP(
            3,  # x, y, z coordinates
            dino_feature_recon_dim,
            num_layers_dino,
            nf=mlp_hidden_size,
            dropout=0,
            activation="sigmoid",
            min_max=dino_minmax,
            n_harmonic_functions=embedder_freq_dino,
            embedder_scalar=embedder_scalar,
            embed_concat_pts=embed_concat_pts,
            extra_feat_dim=0,
            symmetrize=sym_dino
        )
        
    def forward(self, total_iter=None, is_training=True):
        prior_shape = self.netShape.getMesh(total_iter=total_iter, jitter_grid=is_training)
        return prior_shape, self.netDINO


class InstancePredictor(nn.Module):
    def __init__(self, cfgs, tet_bbox=None):
        super().__init__()
        self.cfgs = cfgs
        self.grid_scale = cfgs.get('grid_scale', 5)
        mlp_hidden_size = cfgs.get('mlp_hidden_size', 64)
        embedder_scalar = 2 * np.pi / self.grid_scale * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9
        embed_concat_pts = cfgs.get('embed_concat_pts', True)

        ## Image encoder
        which_vit = cfgs.get('which_vit', 'dino_vits8')
        if which_vit == 'dino_vits8':
            vit_feat_dim = 384
        elif which_vit == 'dino_vitb8':
            vit_feat_dim = 768
        encoder_latent_dim = cfgs.get('latent_dim', 256)
        encoder_pretrained = cfgs.get('encoder_pretrained', False)
        encoder_frozen = cfgs.get('encoder_frozen', False)
        vit_final_layer_type = cfgs.get('vit_final_layer_type', 'conv')
        self.netEncoder = networks.ViTEncoder(
            cout=encoder_latent_dim,
            which_vit=which_vit,
            pretrained=encoder_pretrained,
            frozen=encoder_frozen,
            final_layer_type=vit_final_layer_type
        )
        
        ## Texture network
        num_layers_tex = cfgs.get("num_layers_tex", 5)
        texture_out_dim = 9  # in practice, only first three channels are used as albedo RGB
        kd_minmax = torch.FloatTensor(cfgs.get('kd_minmax', [[0., 0.], [0., 0.], [0., 0.]]))  # 3x2
        ks_minmax = torch.FloatTensor(cfgs.get('ks_minmax', [[0., 0.], [0., 0.], [0., 0.]]))  # 3x2
        nrm_minmax = torch.FloatTensor(cfgs.get('nrm_minmax', [[-1., 1.], [-1., 1.], [0., 1.]]))  # 3x2
        texture_min_max = torch.cat((kd_minmax, ks_minmax, nrm_minmax), dim=0)  # 9x2
        embedder_freq_tex = cfgs.get('embedder_freq_tex', 10)
        sym_texture = cfgs.get("sym_texture", False)
        self.netTexture = networks.CoordMLP(
            3,  # x, y, z coordinates
            texture_out_dim,
            num_layers_tex,
            nf=mlp_hidden_size,
            dropout=0,
            activation="sigmoid",
            min_max=texture_min_max,
            n_harmonic_functions=embedder_freq_tex,
            embedder_scalar=embedder_scalar,
            embed_concat_pts=embed_concat_pts,
            extra_feat_dim=encoder_latent_dim,
            symmetrize=sym_texture
        )

        ## Pose network
        self.pose_arch = cfgs.get('pose_arch', 'encoder_dino_patch_key')
        self.cam_pos_z_offset = cfgs.get('cam_pos_z_offset', 10.)
        self.cam_pos_offset = torch.FloatTensor([0, 0, -self.cam_pos_z_offset])
        self.fov = cfgs.get('fov', 25)
        half_range = np.tan(self.fov /2 /180 * np.pi) * self.cam_pos_z_offset  # 2.22
        self.max_trans_xy_range = half_range * cfgs.get('max_trans_xy_range_ratio', 1.)
        self.max_trans_z_range = half_range * cfgs.get('max_trans_z_range_ratio', 1.)
        self.lookat_zeroy = cfgs.get('lookat_zeroy', False)
        self.rot_temp_scalar = cfgs.get('rot_temp_scalar', 1.)
        self.naive_probs_iter = cfgs.get('naive_probs_iter', 2000)
        self.best_pose_start_iter = cfgs.get('best_pose_start_iter', 6000)
        self.rot_rep = cfgs.get('rot_rep', 'euler_angle')
        if self.rot_rep == 'euler_angle':
            pose_cout = 6
        elif self.rot_rep == 'quaternion':
            pose_cout = 7
        elif self.rot_rep == 'lookat':
            pose_cout = 6
        elif self.rot_rep == 'quadlookat':
            self.num_pose_hypos = 4
            pose_cout = (3 + 1) * self.num_pose_hypos + 3  # forward vector + prob logits for each hypothesis, 3 for translation
            self.orthant_signs = torch.FloatTensor([[1,1,1], [-1,1,1], [-1,1,-1], [1,1,-1]])  # 4x3
        elif self.rot_rep == 'octlookat':
            self.num_pose_hypos = 8
            pose_cout = (3 + 1) * self.num_pose_hypos + 3  # forward vector + prob logits for each hypothesis, 3 for translation
            self.orthant_signs = torch.stack(torch.meshgrid([torch.arange(1, -2, -2)] *3), -1).view(-1, 3)  # 8x3
        else:
            raise NotImplementedError
        self.netPose = networks.Encoder32(cin=vit_feat_dim, cout=pose_cout, nf=256, activation=None)  # vit patches are 32x32
        
        ## Deformation network
        self.enable_deform = cfgs.get('enable_deform', False)
        if self.enable_deform:
            num_layers_deform = cfgs.get('num_layers_deform', 5)
            self.deform_epochs = np.arange(*cfgs.get('deform_epochs', [0, 0]))
            embedder_freq_deform = cfgs.get('embedder_freq_deform', 10)
            sym_deform = cfgs.get("sym_deform", False)
            self.netDeform = networks.CoordMLP(
                3,  # x, y, z coordinates
                3,  # dx, dy, dz deformation
                num_layers_deform,
                nf=mlp_hidden_size,
                dropout=0,
                activation=None,
                min_max=None,
                n_harmonic_functions=embedder_freq_deform,
                embedder_scalar=embedder_scalar,
                embed_concat_pts=embed_concat_pts,
                extra_feat_dim=encoder_latent_dim,
                symmetrize=sym_deform
            )

        ## Articulation network
        self.enable_articulation = cfgs.get('enable_articulation', False)
        if self.enable_articulation:
            self.articulation_epochs = np.arange(*cfgs.get('articulation_epochs', [0, 0]))
            self.num_body_bones = cfgs.get('num_body_bones', 4)
            self.body_bones_mode = cfgs.get('body_bones_mode', 'z_minmax')
            self.num_legs = cfgs.get('num_legs', 0)
            self.num_leg_bones = cfgs.get('num_leg_bones', 0)
            self.num_bones = self.num_body_bones + self.num_legs * self.num_leg_bones
            self.attach_legs_to_body_epochs = np.arange(*cfgs.get('attach_legs_to_body_epochs', [0, 0]))
            self.static_root_bones = cfgs.get('static_root_bones', False)
            self.skinning_temperature = cfgs.get('skinning_temperature', 1)
            self.max_arti_angle = cfgs.get('max_arti_angle', 60)
            self.constrain_legs = cfgs.get('constrain_legs', False)
            self.articulation_multiplier = cfgs.get('articulation_multiplier', 1)
            num_layers_arti = cfgs.get('num_layers_arti', 4)
            articulation_arch = cfgs.get('articulation_arch', 'mlp')
            self.articulation_feature_mode = cfgs.get('articulation_feature_mode', 'global')
            embedder_freq_arti = cfgs.get('embedder_freq_arti', 8)
            if self.articulation_feature_mode == 'global':
                feat_dim = encoder_latent_dim
            elif self.articulation_feature_mode == 'sample':
                feat_dim = vit_feat_dim
            elif self.articulation_feature_mode == 'sample+global':
                feat_dim = encoder_latent_dim + vit_feat_dim
            else:
                raise NotImplementedError
            pos_embedder_scalar = np.pi * 0.9  # originally (-1, 1) rescale to (-pi, pi) * 0.9
            self.netArticulation = networks.ArticulationNetwork(
                articulation_arch,
                feat_dim,
                pos_dim=1+2+3*2,  # bone index + 2D mid bone position + 3D joint locations
                num_layers=num_layers_arti,
                nf=mlp_hidden_size,
                n_harmonic_functions=embedder_freq_arti,
                embedder_scalar=pos_embedder_scalar
            )
            self.kinematic_tree_epoch = -1  # initialize to -1 to force compute kinematic tree at first epoch
        
        ## Lighting network
        self.enable_lighting = cfgs.get('enable_lighting', False)
        if self.enable_lighting:
            num_layers_light = cfgs.get('num_layers_light', 5)
            amb_diff_minmax = torch.FloatTensor(cfgs.get('amb_diff_minmax', [[0.0, 1.0], [0.5, 1.0]]))
            self.netLight = light.DirectionalLight(
                encoder_latent_dim,
                num_layers_light,
                mlp_hidden_size,
                intensity_min_max=amb_diff_minmax
            )

    def forward_encoder(self, images):
        images_in = images.view(-1, *images.shape[2:]) * 2 - 1  # (B*F)xCxHxW rescale to (-1, 1)
        patch_out = patch_key = None
        feat_out, feat_key, patch_out, patch_key = self.netEncoder(images_in, return_patches=True)
        return feat_out, feat_key, patch_out, patch_key
    
    def forward_pose(self, patch_out, patch_key):
        if self.pose_arch == 'encoder_dino_patch_key':
            pose = self.netPose(patch_key)  # Shape: (B, latent_dim)
        elif self.pose_arch == 'encoder_dino_patch_out':
            pose = self.netPose(patch_out)  # Shape: (B, latent_dim)
        else:
            raise NotImplementedError
        trans_pred = pose[...,-3:].tanh() * torch.FloatTensor([self.max_trans_xy_range, self.max_trans_xy_range, self.max_trans_z_range]).to(pose.device)

        if self.rot_rep == 'euler_angle':
            max_rot_x_range = self.cfgs.get("max_rot_x_range", 180)
            max_rot_y_range = self.cfgs.get("max_rot_y_range", 180)
            max_rot_z_range = self.cfgs.get("max_rot_z_range", 180)
            max_rot_range = torch.FloatTensor([max_rot_x_range, max_rot_y_range, max_rot_z_range]).to(pose.device)
            rot_pred = pose[...,:3].tanh()
            rot_pred = rot_pred * max_rot_range /180 * np.pi

        elif self.rot_rep == 'quaternion':
            quat_init = torch.FloatTensor([0.01,0,0,0]).to(pose.device)
            rot_pred = pose[...,:4] + quat_init
            rot_pred = nn.functional.normalize(rot_pred, p=2, dim=-1)
            # rot_pred = torch.cat([rot_pred[...,:1].abs(), rot_pred[...,1:]], -1)  # make real part non-negative
            rot_pred = rot_pred * rot_pred[...,:1].sign()  # make real part non-negative

        elif self.rot_rep == 'lookat':
            vec_forward_raw = pose[...,:3]
            if self.lookat_zeroy:
                vec_forward_raw = vec_forward_raw * torch.FloatTensor([1,0,1]).to(pose.device)
            vec_forward_raw = nn.functional.normalize(vec_forward_raw, p=2, dim=-1)  # x right, y up, z forward
            rot_pred = vec_forward_raw

        elif self.rot_rep in ['quadlookat', 'octlookat']:
            rots_pred = pose[..., :self.num_pose_hypos*4].view(-1, self.num_pose_hypos, 4)  # (B*F, K, 4)
            rots_logits = rots_pred[..., :1]
            vec_forward_raw = rots_pred[..., 1:4]
            xs, ys, zs = vec_forward_raw.unbind(-1)
            margin = 0.
            xs = nn.functional.softplus(xs, beta=np.log(2)/(0.5+margin)) - margin  # initialize to 0.5
            if self.rot_rep == 'octlookat':
                ys = nn.functional.softplus(ys, beta=np.log(2)/(0.5+margin)) - margin  # initialize to 0.5
            if self.lookat_zeroy:
                ys = ys * 0
            zs = nn.functional.softplus(zs, beta=2*np.log(2))  # initialize to 0.5
            vec_forward_raw = torch.stack([xs, ys, zs], -1)
            vec_forward_raw = vec_forward_raw * self.orthant_signs.to(pose.device)
            vec_forward_raw = nn.functional.normalize(vec_forward_raw, p=2, dim=-1)  # x right, y up, z forward
            rot_pred = torch.cat([rots_logits, vec_forward_raw], -1).view(-1, self.num_pose_hypos*4)  # (B*F, K*4)

        else:
            raise NotImplementedError
        
        pose = torch.cat([rot_pred, trans_pred], -1)
        return pose
    
    def forward_deformation(self, shape, feat=None):
        original_verts = shape.v_pos
        num_verts = original_verts.shape[1]
        if feat is not None:
            deform_feat = feat[:, None, :].repeat(1, num_verts, 1)  # Shape: (B, num_verts, latent_dim)
            original_verts = original_verts.repeat(len(feat), 1, 1)
        deformation = self.netDeform(original_verts, deform_feat) * 0.1  # Shape: (B, num_verts, 3), multiply by 0.1 to minimize descruption when enabled
        shape = shape.deform(deformation)
        return shape, deformation
    
    def forward_articulation(self, shape, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch):
        verts = shape.v_pos
        if len(verts) == batch_size * num_frames:
            verts = verts.view(batch_size, num_frames, *verts.shape[1:])
        else:
            verts = verts[None]
        
        ## recompute kinematic tree at the beginning of each epoch
        if self.kinematic_tree_epoch != epoch:
            attach_legs_to_body = epoch in self.attach_legs_to_body_epochs
            bones, self.kinematic_tree, self.bone_aux = estimate_bones(verts.detach(), self.num_body_bones, n_legs=self.num_legs, n_leg_bones=self.num_leg_bones, body_bones_mode=self.body_bones_mode, compute_kinematic_chain=True, attach_legs_to_body=attach_legs_to_body)
            self.kinematic_tree_epoch = epoch
        else:
            bones = estimate_bones(verts.detach(), self.num_body_bones, n_legs=self.num_legs, n_leg_bones=self.num_leg_bones, body_bones_mode=self.body_bones_mode, compute_kinematic_chain=False, aux=self.bone_aux)

        ## bone mid point 2D location
        bones_pos = bones  # Shape: (B, F, K, 2, 3)
        if batch_size > bones_pos.shape[0] or num_frames > bones_pos.shape[1]:
            assert bones_pos.shape[0] == 1 and bones_pos.shape[1] == 1, "canonical mesh should have batch_size=1 and num_frames=1"
            bones_pos = bones_pos.repeat(batch_size, num_frames, 1, 1, 1)
        num_bones = bones_pos.shape[2]
        bones_pos = bones_pos.view(batch_size*num_frames, num_bones, 2, 3)  # NxKx2x3
        bones_mid_pos = bones_pos.mean(2)  # NxKx3
        bones_mid_pos_world4 = torch.cat([bones_mid_pos, torch.ones_like(bones_mid_pos[..., :1])], -1)  # NxKx4
        bones_mid_pos_clip4 = bones_mid_pos_world4 @ mvp.transpose(-1, -2)
        bones_mid_pos_2d = bones_mid_pos_clip4[..., :2] / bones_mid_pos_clip4[..., 3:4]
        bones_mid_pos_2d = bones_mid_pos_2d.detach()  # we don't want gradient to flow through the camera projection

        ## two bone end points 3D locations in camera space
        bones_pos_world4 = torch.cat([bones_pos, torch.ones_like(bones_pos[..., :1])], -1)  # NxKx2x4
        bones_pos_cam4 = bones_pos_world4 @ w2c[:,None].transpose(-1, -2)
        bones_pos_cam3 = bones_pos_cam4[..., :3] / bones_pos_cam4[..., 3:4]
        bones_pos_cam3 = bones_pos_cam3 + torch.FloatTensor([0, 0, self.cam_pos_z_offset]).to(bones_pos_cam3.device).view(1, 1, 1, 3)
        bones_pos_3d = bones_pos_cam3.view(batch_size*num_frames, num_bones, 2*3) / self.grid_scale * 2  # (-1, 1), NxKx(2*3)
        
        ## bone index
        bones_idx = torch.arange(num_bones).to(bones_pos.device)
        bones_idx_in = ((bones_idx[None, :, None] + 0.5) / num_bones * 2 - 1).repeat(batch_size * num_frames, 1, 1)  # (-1, 1)
        bones_pos_in = torch.cat([bones_mid_pos_2d, bones_pos_3d, bones_idx_in], -1)
        bones_pos_in = bones_pos_in.detach()  # we don't want gradient to flow through the camera pose

        if self.articulation_feature_mode == 'global':
            bones_feat = feat[:, None].repeat(1, num_bones, 1)  # (BxF, K, feat_dim)
        elif self.articulation_feature_mode == 'sample':
            bones_feat = F.grid_sample(patch_feat, bones_mid_pos_2d.view(batch_size * num_frames, 1, -1, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # (BxF, K, feat_dim)
        elif self.articulation_feature_mode == 'sample+global':
            bones_feat = F.grid_sample(patch_feat, bones_mid_pos_2d.view(batch_size * num_frames, 1, -1, 2), mode='bilinear').squeeze(dim=-2).permute(0, 2, 1)  # (BxF, K, feat_dim)
            bones_feat = torch.cat([feat[:, None].repeat(1, num_bones, 1), bones_feat], -1)
        else:
            raise NotImplementedError

        articulation_angles = self.netArticulation(bones_feat, bones_pos_in).view(batch_size, num_frames, num_bones, 3)  # (B, F, K, 3)
        articulation_angles = articulation_angles * self.articulation_multiplier
        articulation_angles = articulation_angles.tanh()

        if self.static_root_bones:
            root_bones = [self.num_body_bones // 2 - 1, self.num_body_bones - 1]  # middle two bones, assuming an even number of bones
            tmp_mask = torch.ones_like(articulation_angles)
            tmp_mask[:, :, root_bones] = 0
            articulation_angles = articulation_angles * tmp_mask

        if self.constrain_legs:
            leg_bones_idx = self.num_body_bones + np.arange(self.num_leg_bones * self.num_legs)

            tmp_mask = torch.zeros_like(articulation_angles)
            tmp_mask[:, :, leg_bones_idx, 2] = 1  # twist / rotation around z axis
            articulation_angles = tmp_mask * (articulation_angles * 0.3) + (1 - tmp_mask) * articulation_angles  # limit to (-0.3, 0.3)

            tmp_mask = torch.zeros_like(articulation_angles)
            tmp_mask[:, :, leg_bones_idx, 1] = 1  # side bending / rotation around y axis
            articulation_angles = tmp_mask * (articulation_angles * 0.3) + (1 - tmp_mask) * articulation_angles  # limit to (-0.3, 0.3)
        
        articulation_angles = articulation_angles * self.max_arti_angle / 180 * np.pi
        
        verts_articulated, aux = skinning(verts, bones, self.kinematic_tree, articulation_angles, output_posed_bones=True, temperature=self.skinning_temperature)
        verts_articulated = verts_articulated.view(batch_size*num_frames, *verts_articulated.shape[2:])
        v_tex = shape.v_tex
        if len(v_tex) != len(verts_articulated):
            v_tex = v_tex.repeat(len(verts_articulated), 1, 1)
        articulated_shape = mesh.make_mesh(verts_articulated, shape.t_pos_idx, v_tex, shape.t_tex_idx, shape.material)
        return articulated_shape, articulation_angles, aux
    
    def get_camera_extrinsics_from_pose(self, pose, znear=0.1, zfar=1000.):
        N = len(pose)
        pose_R = pose[:, :9].view(N, 3, 3).transpose(2, 1)  # to be compatible with pytorch3d
        pose_T = pose[:, -3:] + self.cam_pos_offset.to(pose.device)
        pose_T = pose_T.view(N, 3, 1)
        pose_RT = torch.cat([pose_R, pose_T], axis=2)  # Nx3x4
        w2c = torch.cat([pose_RT, torch.FloatTensor([0, 0, 0, 1]).repeat(N, 1, 1).to(pose.device)], axis=1)  # Nx4x4
        proj = util.perspective(self.fov / 180 * np.pi, 1, znear, zfar)[None].to(pose.device)  # assuming square images
        mvp = torch.matmul(proj, w2c)
        campos = -torch.matmul(pose_R.transpose(2, 1), pose_T).view(N, 3)
        return mvp, w2c, campos

    def forward(self, images=None, prior_shape=None, epoch=None, total_iter=None, is_training=True):
        batch_size, num_frames = images.shape[:2]
        feat_out, feat_key, patch_out, patch_key = self.forward_encoder(images)  # first two dimensions are collapsed N=(B*F)
        shape = prior_shape
        texture = self.netTexture

        poses_raw = self.forward_pose(patch_out, patch_key)
        pose_raw, pose, multi_hypothesis_aux = sample_pose_hypothesis_from_quad_predictions(poses_raw, total_iter, rot_temp_scalar=self.rot_temp_scalar, num_hypos=self.num_pose_hypos, naive_probs_iter=self.naive_probs_iter, best_pose_start_iter=self.best_pose_start_iter, random_sample=is_training)
        mvp, w2c, campos = self.get_camera_extrinsics_from_pose(pose)

        deformation = None
        if self.enable_deform and epoch in self.deform_epochs:
            shape, deformation = self.forward_deformation(shape, feat_key)
        
        arti_params, articulation_aux = None, {}
        if self.enable_articulation and epoch in self.articulation_epochs:
            shape, arti_params, articulation_aux = self.forward_articulation(shape, feat_key, patch_key, mvp, w2c, batch_size, num_frames, epoch)
        
        light = self.netLight if self.enable_lighting else None

        aux = multi_hypothesis_aux
        aux.update(articulation_aux)

        return shape, pose_raw, pose, mvp, w2c, campos, texture, feat_out, deformation, arti_params, light, aux
