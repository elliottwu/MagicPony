import numpy as np
import torch
import torch.nn as nn
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
import os
import os.path as osp
from einops import rearrange
from .utils import misc
from .dataloaders import get_sequence_loader, get_image_loader
from .render import util
from .render import render
from .predictor import PriorPredictor, InstancePredictor


def validate_tensor_to_device(x, device):
    if torch.any(torch.isnan(x)):
        return None
    else:
        return x.to(device)


def collapseBF(x):
    return None if x is None else rearrange(x, 'b f ... -> (b f) ...')
def expandBF(x, b, f):
    return None if x is None else rearrange(x, '(b f) ... -> b f ...', b=b, f=f)


def get_optimizer(model, lr=0.0001, betas=(0.9, 0.999), weight_decay=0):
    return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, betas=betas, weight_decay=weight_decay)


class MagicPony:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = cfgs.get('device', 'cpu')

        self.num_epochs = cfgs.get('num_epochs', 10)
        self.lr = cfgs.get('lr', 1e-4)
        self.use_scheduler = cfgs.get('use_scheduler', False)
        if self.use_scheduler:
            scheduler_milestone = cfgs.get('scheduler_milestone', [1,2,3,4,5])
            scheduler_gamma = cfgs.get('scheduler_gamma', 0.5)
            self.make_scheduler = lambda optim: torch.optim.lr_scheduler.MultiStepLR(optim, milestones=scheduler_milestone, gamma=scheduler_gamma)
        
        self.netPrior = PriorPredictor(self.cfgs)
        self.prior_lr = cfgs.get('prior_lr', self.lr)
        self.prior_weight_decay = cfgs.get('prior_weight_decay', 0.)
        self.backward_prior = cfgs.get('backward_prior', True)
        self.netInstance = InstancePredictor(self.cfgs, tet_bbox=self.netPrior.netShape.getAABB())
        self.texture_epochs = np.arange(*cfgs.get('texture_epochs', [0, self.num_epochs]))
        self.dmtet_grid_res_smaller_epoch = cfgs.get('dmtet_grid_res_smaller_epoch', 0)
        self.dmtet_grid_res_smaller = cfgs.get('dmtet_grid_res_smaller', 128)
        self.dmtet_grid_res = cfgs.get('dmtet_grid_res', 256)
        self.background_mode = cfgs.get('background_mode', 'none')
        self.dino_feature_recon_dim = cfgs.get('dino_feature_recon_dim', 64)
        self.sdf_reg_decay_start_iter = cfgs.get('sdf_reg_decay_start_iter', 999999)
        self.arti_reg_loss_epochs = np.arange(*cfgs.get('arti_reg_loss_epochs', [0, self.num_epochs]))
        self.render_flow = self.cfgs.get('flow_loss_weight', 0.) > 0.
        
        self.glctx = dr.RasterizeGLContext()
        self.in_image_size = cfgs.get('in_image_size', 128)
        self.out_image_size = cfgs.get('out_image_size', 128)
        self.cam_pos_z_offset = cfgs.get('cam_pos_z_offset', 10.)
        self.fov = cfgs.get("fov", 25)
        self.extra_renders = cfgs.get('extra_renders', [])
        self.renderer_spp = cfgs.get('renderer_spp', 1)

        self.total_loss = 0.
    
    @staticmethod
    def get_data_loaders(cfgs, data_type='sequence', in_image_size=256, out_image_size=256, batch_size=64, num_workers=4, train_data_dir=None, val_data_dir=None, test_data_dir=None):
        train_loader = val_loader = test_loader = None
        random_shuffle_train_samples = cfgs.get('random_shuffle_train_samples', False)
        random_sample_train_frames = cfgs.get('random_sample_train_frames', False)
        random_sample_val_frames = cfgs.get('random_sample_val_frames', False)
        random_xflip_train = cfgs.get('random_xflip_train', False)
        load_flow = cfgs.get('load_flow', False)
        load_background = cfgs.get('background_mode', 'none') == 'background'
        load_dino_feature = cfgs.get('load_dino_feature', False)
        load_dino_cluster = cfgs.get('load_dino_cluster', False)
        dino_feature_dim = cfgs.get('dino_feature_dim', 64)

        if data_type == 'sequence':
            skip_beginning = cfgs.get('skip_beginning', 4)
            skip_end = cfgs.get('skip_end', 4)
            num_frames = cfgs.get('num_frames', 2)
            min_seq_len = cfgs.get('min_seq_len', 10)
            get_loader = lambda is_train, random_sample_frames=False, **kwargs: get_sequence_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                in_image_size=in_image_size,
                out_image_size=out_image_size,
                skip_beginning=skip_beginning,
                skip_end=skip_end,
                num_frames=num_frames,
                min_seq_len=min_seq_len,
                load_flow=load_flow,
                load_background=load_background,
                load_dino_feature=load_dino_feature,
                load_dino_cluster=load_dino_cluster,
                dino_feature_dim=dino_feature_dim,
                random_xflip=random_xflip_train if is_train else False,
                random_sample=random_sample_frames,
                dense_sample=is_train,
                shuffle=random_shuffle_train_samples if is_train else False,
                **kwargs)
        
        elif data_type == 'image':
            get_loader = lambda is_train, random_sample=False, **kwargs: get_image_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                in_image_size=in_image_size,
                out_image_size=out_image_size,
                load_background=load_background,
                load_dino_feature=load_dino_feature,
                load_dino_cluster=load_dino_cluster,
                dino_feature_dim=dino_feature_dim,
                random_xflip=random_xflip_train if is_train else False,
                shuffle=random_shuffle_train_samples if is_train else False,
                **kwargs)

        else:
            raise ValueError(f"Unexpected data type: {data_type}")

        if train_data_dir is not None:
            assert osp.isdir(train_data_dir), f"Training data directory does not exist: {train_data_dir}"
            print(f"Loading training data from {train_data_dir}")
            train_loader = get_loader(is_train=True, random_sample_frames=random_sample_train_frames, data_dir=train_data_dir)

        if val_data_dir is not None:
            assert osp.isdir(val_data_dir), f"Validation data directory does not exist: {val_data_dir}"
            print(f"Loading validation data from {val_data_dir}")
            val_loader = get_loader(is_train=False, random_sample_frames=random_sample_val_frames, data_dir=val_data_dir)

        if test_data_dir is not None:
            assert osp.isdir(test_data_dir), f"Testing data directory does not exist: {test_data_dir}"
            print(f"Loading testing data from {test_data_dir}")
            test_loader = get_loader(is_train=False, random_sample_frames=False, data_dir=test_data_dir)

        return train_loader, val_loader, test_loader

    def load_model_state(self, cp):
        self.netPrior.load_state_dict(cp["netPrior"])
        self.netInstance.load_state_dict(cp["netInstance"])

    def load_optimizer_state(self, cp):
        self.optimizerPrior.load_state_dict(cp["optimizerPrior"])
        self.optimizerInstance.load_state_dict(cp["optimizerInstance"])
        if self.use_scheduler:
            if 'schedulerPrior' in cp:
                self.schedulerPrior.load_state_dict(cp["schedulerPrior"])
            if 'schedulerInstance' in cp:
                self.schedulerInstance.load_state_dict(cp["schedulerInstance"])

    def get_model_state(self):
        state = {"netPrior": self.netPrior.state_dict(),
                 "netInstance": self.netInstance.state_dict()}
        return state

    def get_optimizer_state(self):
        state = {"optimizerPrior": self.optimizerPrior.state_dict(),
                 "optimizerInstance": self.optimizerInstance.state_dict()}
        if self.use_scheduler:
            state["schedulerPrior"] = self.schedulerPrior.state_dict()
            state["schedulerInstance"] = self.schedulerInstance.state_dict()
        return state

    def to(self, device):
        self.device = device
        self.netPrior.to(device)
        self.netInstance.to(device)

    def set_train(self):
        self.netPrior.train()
        self.netInstance.train()

    def set_eval(self):
        self.netPrior.eval()
        self.netInstance.eval()

    def reset_optimizers(self):
        print("Resetting optimizers...")
        self.optimizerPrior = get_optimizer(self.netPrior, lr=self.prior_lr, weight_decay=self.prior_weight_decay)
        self.optimizerInstance = get_optimizer(self.netInstance, self.lr)
        if self.use_scheduler:
            self.schedulerPrior = self.make_scheduler(self.optimizerPrior)
            self.schedulerInstance = self.make_scheduler(self.optimizerInstance)

    def backward(self):
        self.optimizerInstance.zero_grad()
        if self.backward_prior:
            self.optimizerPrior.zero_grad()
        self.total_loss.backward()
        self.optimizerInstance.step()
        if self.backward_prior:
            self.optimizerPrior.step()
        self.total_loss = 0.

    def scheduler_step(self):
        if self.use_scheduler:
            self.schedulerPrior.step()
            self.schedulerInstance.step()

    def render(self, render_modes, shape, texture, mvp, w2c, campos, resolution, background=None, im_features=None, light=None, prior_shape=None, dino_net=None, bsdf='diffuse', two_sided_shading=True, num_frames=None, spp=None):
        h, w = resolution
        N = len(mvp)
        if background is None:
            background = self.background_mode
        if spp is None:
            spp = self.renderer_spp
        
        if background in ['none', 'black']:
            bg_image = torch.zeros((N, h, w, 3), device=mvp.device)
        elif background == 'white':
            bg_image = torch.ones((N, h, w, 3), device=mvp.device)
        elif background == 'checkerboard':
            bg_image = torch.FloatTensor(util.checkerboard((h, w), 8), device=mvp.device).repeat(N, 1, 1, 1)  # NxHxWxC
        else:
            raise NotImplementedError

        rendered = render.render_mesh(
            self.glctx,
            shape,
            mtx_in=mvp,
            w2c=w2c,
            view_pos=campos,
            material=texture,
            lgt=light,
            resolution=resolution,
            spp=spp,
            num_layers=1,
            msaa=True,
            background=bg_image,
            bsdf=bsdf,
            feat=im_features,
            render_modes=render_modes,
            prior_mesh=prior_shape,
            two_sided_shading=two_sided_shading,
            dino_net=dino_net,
            num_frames=num_frames)
        return rendered

    def compute_reconstruction_losses(self, image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt, dino_feat_im_pred, background_mode='none', reduce=False):
        losses = {}
        batch_size, num_frames, _, h, w = image_pred.shape  # BxFxCxHxW

        ## mask L2 loss
        mask_pred_valid = mask_pred * mask_valid
        mask_loss = (mask_pred_valid - mask_gt) ** 2
        losses['mask_loss'] = mask_loss.view(batch_size, num_frames, -1).mean(2)
        losses['mask_dt_loss'] = (mask_pred * mask_dt[:,:,1]).view(batch_size, num_frames, -1).mean(2)
        losses['mask_inv_dt_loss'] = ((1-mask_pred) * mask_dt[:,:,0]).view(batch_size, num_frames, -1).mean(2)

        mask_pred_binary = (mask_pred_valid > 0.).float().detach()
        mask_both_binary = collapseBF(mask_pred_binary * mask_gt)  # BFxHxW
        mask_both_binary = (nn.functional.avg_pool2d(mask_both_binary.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float().detach()  # erode by 1 pixel
        mask_both_binary = expandBF(mask_both_binary, b=batch_size, f=num_frames)  # BxFxHxW

        ## RGB L1 loss
        rgb_loss = (image_pred - image_gt).abs()
        if background_mode in ['background', 'input']:
            pass
        else:
            rgb_loss = rgb_loss * mask_both_binary.unsqueeze(2)
        losses['rgb_loss'] = rgb_loss.view(batch_size, num_frames, -1).mean(2)

        ## flow loss between consecutive frames
        if flow_pred is not None:
            flow_loss = (flow_pred - flow_gt) ** 2.
            flow_loss_mask = mask_both_binary[:,:-1].unsqueeze(2).expand_as(flow_gt)

            ## ignore frames where GT flow is too large (likely inaccurate)
            large_flow = (flow_gt.abs() > 0.5).float() * flow_loss_mask
            large_flow = (large_flow.view(batch_size, num_frames-1, -1).sum(2) > 0).float()
            self.large_flow = large_flow

            flow_loss = flow_loss * flow_loss_mask * (1 - large_flow[:,:,None,None,None])
            num_mask_pixels = flow_loss_mask.reshape(batch_size, num_frames-1, -1).sum(2).clamp(min=1)
            losses['flow_loss'] = (flow_loss.reshape(batch_size, num_frames-1, -1).sum(2) / num_mask_pixels)

        ## DINO feature loss
        if dino_feat_im_pred is not None and dino_feat_im_gt is not None:
            dino_feat_loss = (dino_feat_im_pred - dino_feat_im_gt) ** 2
            dino_feat_loss = dino_feat_loss * mask_both_binary.unsqueeze(2)
            losses['dino_feat_im_loss'] = dino_feat_loss.reshape(batch_size, num_frames, -1).mean(2)

        if reduce:
            for k, v in losses.item():
                losses[k] = v.mean()
        return losses

    def compute_regularizers(self, arti_params=None, deformation=None):
        losses = {}
        aux = {}
        losses.update(self.netPrior.netShape.get_sdf_reg_loss())
        if arti_params is not None:
            losses['arti_reg_loss'] = (arti_params ** 2).mean()
        if deformation is not None:
            losses['deformation_reg_loss'] = (deformation ** 2).mean()
        return losses, aux
    
    def forward(self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None, logger_prefix='', is_training=True):
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, seq_idx, frame_idx = (*map(lambda x: validate_tensor_to_device(x, self.device), batch),)
        global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx8
        mask_gt = (mask_gt[:, :, 0, :, :] > 0.9).float()  # BxFxHxW
        mask_dt = mask_dt / self.in_image_size
        batch_size, num_frames, _, _, _ = input_image.shape  # BxFxCxHxW
        h = w = self.out_image_size
        aux_viz = {}

        dino_feat_im_gt = None if dino_feat_im is None else expandBF(torch.nn.functional.interpolate(collapseBF(dino_feat_im), size=[h, w], mode="bilinear"), batch_size, num_frames)[:, :, :self.dino_feature_recon_dim]
        dino_cluster_im_gt = None if dino_cluster_im is None else expandBF(torch.nn.functional.interpolate(collapseBF(dino_cluster_im), size=[h, w], mode="nearest"), batch_size, num_frames)
        
        ## GT image
        image_gt = input_image
        if self.out_image_size != self.in_image_size:
            image_gt = expandBF(torch.nn.functional.interpolate(collapseBF(image_gt), size=[h, w], mode='bilinear'), batch_size, num_frames)
            if flow_gt is not None:
                flow_gt = expandBF(torch.nn.functional.interpolate(collapseBF(flow_gt), size=[h, w], mode="bilinear"), batch_size, num_frames-1)

        ## predict prior shape and DINO
        dmtet_grid_res = self.dmtet_grid_res_smaller if epoch < self.dmtet_grid_res_smaller_epoch else self.dmtet_grid_res
        if self.netPrior.netShape.grid_res != dmtet_grid_res:
            self.netPrior.netShape.load_tets(dmtet_grid_res)
        prior_shape, dino_net = self.netPrior(total_iter=total_iter, is_training=is_training)

        ## predict instance specific parameters
        shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(input_image, prior_shape, epoch, total_iter, is_training=is_training)  # first two dim dimensions already collapsed N=(B*F)
        rot_logit = forward_aux['rot_logit']
        rot_idx = forward_aux['rot_idx']
        rot_prob = forward_aux['rot_prob']
        aux_viz.update(forward_aux)

        ## render images
        render_flow = self.render_flow and num_frames > 1
        render_modes = ['shaded', 'dino_pred']
        if render_flow:
            render_modes += ['flow']
        renders = self.render(render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light, prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames)
        renders = map(lambda x: expandBF(x, batch_size, num_frames), renders)
        if render_flow:
            shaded, dino_feat_im_pred, flow_pred = renders
            flow_pred = flow_pred[:, :-1]  # Bx(F-1)x2xHxW
        else:
            shaded, dino_feat_im_pred = renders
            flow_pred = None
        image_pred = shaded[:, :, :3]
        mask_pred = shaded[:, :, 3]

        ## compute reconstruction losses
        losses = self.compute_reconstruction_losses(image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt, dino_feat_im_pred, background_mode=self.background_mode, reduce=False)
        
        ## supervise the rotation logits directly with reconstruction loss
        logit_loss_target = torch.zeros_like(expandBF(rot_logit, batch_size, num_frames))
        final_losses = {}
        for name, loss in losses.items():
            loss_weight = self.cfgs.get(f"{name}_weight", 0.)
            if name in ['dino_feat_im_loss']:
                ## increase the importance of dino loss for viewpoint hypothesis selection (directly increasing dino recon loss leads to stripe artifacts)
                loss_weight = loss_weight * self.cfgs.get("logit_loss_dino_feat_im_loss_multiplier", 1.)
            if loss_weight > 0:
                logit_loss_target += loss * loss_weight
            
            ## multiply the loss with probability of the rotation hypothesis (detached)
            if self.netInstance.rot_rep in ['quadlookat', 'octlookat']:
                loss_prob = rot_prob.detach().view(batch_size, num_frames)[:, :loss.shape[1]]  # handle edge case for flow loss with one frame less
                loss = loss * loss_prob *self.netInstance.num_pose_hypos
            ## only compute flow loss for frames with the same rotation hypothesis
            if name == 'flow_loss' and num_frames > 1:
                ri = rot_idx.view(batch_size, num_frames)
                same_rot_idx = (ri[:, 1:] == ri[:, :-1]).float()
                loss = loss * same_rot_idx
            ## update the final prob-adjusted losses
            final_losses[name] = loss.mean()

        logit_loss_target = collapseBF(logit_loss_target).detach()  # detach the gradient for the loss target
        final_losses['logit_loss'] = ((rot_logit - logit_loss_target)**2.).mean()

        ## regularizers
        regularizers, aux = self.compute_regularizers(arti_params, deformation)
        final_losses.update(regularizers)
        aux_viz.update(aux)

        ## compute final losses
        total_loss = 0
        for name, loss in final_losses.items():
            loss_weight = self.cfgs.get(f"{name}_weight", 0.)
            if loss_weight <= 0:
                continue
            if (epoch not in self.texture_epochs) and (name in ['rgb_loss']):
                continue
            if (epoch not in self.arti_reg_loss_epochs) and (name in ['arti_reg_loss']):
                continue
            if (total_iter >= self.sdf_reg_decay_start_iter) and (name in ['sdf_bce_reg_loss', 'sdf_gradient_reg_loss']):
                decay_rate = max(0, 1 - (total_iter-self.sdf_reg_decay_start_iter) / 10000)
                loss_weight = max(loss_weight * decay_rate, self.cfgs.get(f"{name}_min_weight", 0.))
            total_loss += loss * loss_weight
        self.total_loss += total_loss  # reset to 0 in backward step

        if torch.isnan(self.total_loss):
            print("NaN in loss...")
            import pdb; pdb.set_trace()
        
        final_losses['logit_loss_target'] = logit_loss_target.mean()
        metrics = {'loss': total_loss, **final_losses}

        ## log visuals
        if logger is not None:
            b0 = max(min(batch_size, 16//num_frames), 1)
            def log_image(name, image):
                logger.add_image(logger_prefix+'image/'+name, misc.image_grid(collapseBF(image[:b0,:]).detach().cpu().clamp(0,1)), total_iter)
            def log_video(name, frames):
                logger.add_video(logger_prefix+'animation/'+name, frames.detach().cpu().unsqueeze(0).clamp(0,1), total_iter, fps=2)
            
            log_image('image_gt', input_image)
            log_image('image_pred', image_pred)
            log_image('mask_gt', mask_gt.unsqueeze(2).repeat(1,1,3,1,1))
            log_image('mask_pred', mask_pred.unsqueeze(2).repeat(1,1,3,1,1))

            if dino_feat_im_gt is not None:
                log_image('dino_feat_im_gt', dino_feat_im_gt[:,:,:3])
            if dino_feat_im_pred is not None:
                log_image('dino_feat_im_pred', dino_feat_im_pred[:,:,:3])
            if dino_cluster_im_gt is not None:
                log_image('dino_cluster_im_gt', dino_cluster_im_gt)
                
            if self.render_flow and flow_gt is not None:
                flow_gt_viz = torch.nn.functional.pad(flow_gt, pad=[0, 0, 0, 0, 0, 1])  # add a dummy channel for visualization
                flow_gt_viz = flow_gt_viz + 0.5  # -0.5~1.5
                flow_gt_viz = torch.nn.functional.pad(flow_gt_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # add a dummy frame for visualization

                ## draw marker on large flow frames
                large_flow_marker_mask = torch.zeros_like(flow_gt_viz)
                large_flow_marker_mask[:,:,:,:8,:8] = 1.
                large_flow = torch.cat([self.large_flow, self.large_flow[:,:1] *0.], 1)
                large_flow_marker_mask = large_flow_marker_mask * large_flow[:,:,None,None,None]
                red = torch.FloatTensor([1,0,0]).view(1,1,3,1,1).to(flow_gt_viz.device)
                flow_gt_viz = large_flow_marker_mask * red + (1-large_flow_marker_mask) * flow_gt_viz
                log_image('flow_gt', flow_gt_viz)
            
            if self.render_flow and flow_pred is not None:
                flow_pred_viz = torch.nn.functional.pad(flow_pred, pad=[0, 0, 0, 0, 0, 1])  # add a dummy channel for visualization
                flow_pred_viz = flow_pred_viz + 0.5  # -0.5~1.5
                flow_pred_viz = torch.nn.functional.pad(flow_pred_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # add a dummy frame for visualization
                log_image('flow_pred', flow_pred_viz)

            if arti_params is not None:
                logger.add_histogram(logger_prefix+'arti_params', arti_params, total_iter)
            
            if deformation is not None:
                logger.add_histogram(logger_prefix+'deformation', deformation, total_iter)
            
            rot_rep = self.netInstance.rot_rep
            if rot_rep == 'euler_angle':
                for i, name in enumerate(['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']):
                    logger.add_histogram(logger_prefix+'pose/'+name, pose[...,i], total_iter)
            elif rot_rep == 'quaternion':
                for i, name in enumerate(['qt_0', 'qt_1', 'qt_2', 'qt_3', 'trans_x', 'trans_y', 'trans_z']):
                    logger.add_histogram(logger_prefix+'pose/'+name, pose[...,i], total_iter)
                rot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(pose.detach().cpu()[...,:4]), convention='XYZ')
                for i, name in enumerate(['rot_x', 'rot_y', 'rot_z']):
                    logger.add_histogram(logger_prefix+'pose/'+name, rot_euler[...,i], total_iter)
            elif rot_rep in ['lookat', 'quadlookat', 'octlookat']:
                for i, name in enumerate(['fwd_x', 'fwd_y', 'fwd_z']):
                    logger.add_histogram(logger_prefix+'pose/'+name, pose_raw[...,i], total_iter)
                for i, name in enumerate(['trans_x', 'trans_y', 'trans_z']):
                    logger.add_histogram(logger_prefix+'pose/'+name, pose_raw[...,-3+i], total_iter)
            
            if rot_rep in ['quadlookat', 'octlookat']:
                for i, rp in enumerate(forward_aux['rots_probs'].unbind(-1)):
                    logger.add_histogram(logger_prefix+'pose/rot_prob_%d'%i, rp, total_iter)

            logger.add_histogram(logger_prefix+'sdf', self.netPrior.netShape.get_sdf(), total_iter)
            logger.add_histogram(logger_prefix+'coordinates', shape.v_pos, total_iter)

            render_modes = ['geo_normal', 'kd', 'shading']
            rendered = self.render(render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light, prior_shape=prior_shape)
            geo_normal, albedo, shading = map(lambda x: expandBF(x, batch_size, num_frames), rendered)

            if light is not None:
                param_names = ['dir_x', 'dir_y', 'dir_z', 'int_ambient', 'int_diffuse']
                for name, param in zip(param_names, light.light_params.unbind(-1)):
                    logger.add_histogram(logger_prefix+'light/'+name, param, total_iter)
                log_image('albedo', albedo)
                log_image('shading', shading.repeat(1,1,3,1,1) /2.)

            ## add bone visualizations
            if 'posed_bones' in aux_viz:
                rendered_bone_image = self.render_bones(mvp, aux_viz['posed_bones'], (h, w))
                rendered_bone_image_mask = (rendered_bone_image < 1).any(1, keepdim=True).float()
                geo_normal = rendered_bone_image_mask*0.8 * rendered_bone_image + (1-rendered_bone_image_mask*0.8) * geo_normal

            ## draw marker on images with randomly sampled pose
            if rot_rep in ['quadlookat', 'octlookat']:
                rand_pose_flag = forward_aux['rand_pose_flag']
                rand_pose_marker_mask = torch.zeros_like(geo_normal)
                rand_pose_marker_mask[:,:,:,:16,:16] = 1.
                rand_pose_marker_mask = rand_pose_marker_mask * rand_pose_flag.view(batch_size, num_frames, 1, 1, 1)
                red = torch.FloatTensor([1,0,0]).view(1,1,3,1,1).to(geo_normal.device)
                geo_normal = rand_pose_marker_mask * red + (1-rand_pose_marker_mask) * geo_normal

            log_image('instance_geo_normal', geo_normal)
            
            rot_frames = self.render_rotation_frames('geo_normal', shape, texture, light, (h, w), im_features=im_features, prior_shape=prior_shape, num_frames=15, b=1)
            log_video('instance_normal_rotation', rot_frames)
            
            rot_frames = self.render_rotation_frames('shaded', prior_shape, texture, light, (h, w), im_features=im_features, num_frames=15, b=1)
            log_video('prior_image_rotation', rot_frames)
            
            rot_frames = self.render_rotation_frames('geo_normal', prior_shape, texture, light, (h, w), im_features=im_features, num_frames=15, b=1)
            log_video('prior_normal_rotation', rot_frames)

        if save_results:
            b0 = self.cfgs.get('num_to_save_from_each_batch', batch_size*num_frames)
            fnames = [f'{total_iter:07d}_{fid:10d}' for fid in collapseBF(global_frame_id.int())][:b0]
            def save_image(name, image):
                misc.save_images(save_dir, collapseBF(image)[:b0].clamp(0,1).detach().cpu().numpy(), suffix=name, fnames=fnames)

            save_image('image_gt', image_gt)
            save_image('image_pred', image_pred)
            save_image('mask_gt', mask_gt.unsqueeze(2).repeat(1,1,3,1,1))
            save_image('mask_pred', mask_pred.unsqueeze(2).repeat(1,1,3,1,1))

            if self.render_flow and flow_gt is not None:
                flow_gt_viz = torch.cat([flow_gt, torch.zeros_like(flow_gt[:,:,:1])], 2) + 0.5  # -0.5~1.5
                flow_gt_viz = flow_gt_viz.view(-1, *flow_gt_viz.shape[2:])
                save_image('flow_gt', flow_gt_viz)
            if flow_pred is not None:
                flow_pred_viz = torch.cat([flow_pred, torch.zeros_like(flow_pred[:,:,:1])], 2) + 0.5  # -0.5~1.5
                flow_pred_viz = flow_pred_viz.view(-1, *flow_pred_viz.shape[2:])
                save_image('flow_pred', flow_pred_viz)
            
            tmp_shape = shape.first_n(b0).clone()
            tmp_shape.material = texture
            feat = im_features[:b0] if im_features is not None else None
            misc.save_obj(save_dir, tmp_shape, save_material=False, feat=feat, suffix="mesh", fnames=fnames)
            misc.save_txt(save_dir, pose[:b0].detach().cpu().numpy(), suffix='pose', fnames=fnames)

        return metrics

    def render_rotation_frames(self, render_mode, mesh, texture, light, resolution, background=None, im_features=None, prior_shape=None, num_frames=36, b=None):
        if b is None:
            b = len(mesh)
        else:
            mesh = mesh.first_n(b)
            feat = im_features[:b] if im_features is not None else None
        
        delta_angle = np.pi / num_frames * 2
        delta_rot_matrix = torch.FloatTensor([
            [np.cos(delta_angle),  0, np.sin(delta_angle), 0],
            [0,                    1, 0,                   0],
            [-np.sin(delta_angle), 0, np.cos(delta_angle), 0],
            [0,                    0, 0,                   1],
        ]).to(self.device).repeat(b, 1, 1)

        w2c = torch.FloatTensor(np.diag([1., 1., 1., 1]))
        w2c[:3, 3] = torch.FloatTensor([0, 0, -self.cam_pos_z_offset *1.1])
        w2c = w2c.repeat(b, 1, 1).to(self.device)
        proj = util.perspective(self.fov / 180 * np.pi, 1, n=0.1, f=1000.0).repeat(b, 1, 1).to(self.device)
        mvp = torch.bmm(proj, w2c)
        campos = -w2c[:, :3, 3]

        def rotate_pose(mvp, campos):
            mvp = torch.matmul(mvp, delta_rot_matrix)
            campos = torch.matmul(delta_rot_matrix[:,:3,:3].transpose(2,1), campos[:,:,None])[:,:,0]
            return mvp, campos

        frames = []
        for _ in range(num_frames):
            rendered = self.render([render_mode], mesh, texture, mvp, w2c, campos, resolution, background=background, im_features=feat, light=light, prior_shape=prior_shape)
            shaded = rendered[0]
            frames += [misc.image_grid(shaded[:, :3])]
            mvp, campos = rotate_pose(mvp, campos)
        return torch.stack(frames, dim=0)  # Shape: (T, C, H, W)

    def render_bones(self, mvp, bones_pred, size=(256, 256)):
        bone_world4 = torch.concat([bones_pred, torch.ones_like(bones_pred[..., :1]).to(bones_pred.device)], dim=-1)
        b, f, num_bones = bone_world4.shape[:3]
        bones_clip4 = (bone_world4.view(b, f, num_bones*2, 1, 4) @ mvp.transpose(-1, -2).reshape(b, f, 1, 4, 4)).view(b, f, num_bones, 2, 4)
        bones_uv = bones_clip4[..., :2] / bones_clip4[..., 3:4]  # b, f, num_bones, 2, 2
        dpi = 32
        fx, fy = size[1] // dpi, size[0] // dpi

        rendered = []
        for b_idx in range(b):
            for f_idx in range(f):
                frame_bones_uv = bones_uv[b_idx, f_idx].cpu().numpy()
                fig = plt.figure(figsize=(fx, fy), dpi=dpi, frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                for bone in frame_bones_uv:
                    ax.plot(bone[:, 0], bone[:, 1], marker='o', linewidth=8, markersize=20)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.invert_yaxis()
                # Convert to image
                fig.add_axes(ax)
                fig.canvas.draw_idle()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                image.resize(h, w, 3)
                rendered += [image / 255.]
        
        rendered = expandBF(torch.FloatTensor(np.stack(rendered, 0)).permute(0, 3, 1, 2).to(bones_pred.device), b, f)
        return rendered
