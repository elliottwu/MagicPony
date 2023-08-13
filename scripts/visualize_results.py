import sys
from os import path as osp
sys.path.append(osp.abspath(osp.join(__file__, "../../")))

import argparse
from tqdm import tqdm
import nvdiffrast.torch as dr
import torchvision
import cv2
from PIL import Image
import torch
import glob
import os
import shutil
import numpy as np
from magicpony.model import MagicPony
from magicpony import setup_runtime
from magicpony.geometry.skinning import estimate_bones, skinning, euler_angles_to_matrix
from magicpony.render import util


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def save_images(images, mask_pred, base_names, out_dir, suffix="", mode="transparent"):
    if mask_pred is None:
        mask_pred = [None] * len(images)
    for img, mask, base_name in zip(images, mask_pred, base_names):
        img = img.cpu().numpy()
        img = np.clip(img, 0, 1)
        if mask is not None:
            if mode == "white":
                img = np.transpose(img, (1, 2, 0))
                mask = mask.cpu().numpy()
                mask = np.transpose(mask, (1, 2, 0))
                img = np.flip(img, -1)
                img = img * mask + (1 - mask)
                img = (img * 255).astype(np.uint8)
            else:
                img = (img * 255).astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))
                mask = mask.cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = np.transpose(mask, (1, 2, 0))
                channel_a = mask[..., 0:1]
                img = np.concatenate([np.flip(img, -1), channel_a], axis=-1)
        else:
            img = (img * 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))
            img = np.flip(img, -1)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(osp.join(out_dir, base_name + suffix + ".png"), img)


class FixedDirectionLight(torch.nn.Module):
    def __init__(self, direction, amb, diff):
        super(FixedDirectionLight, self).__init__()
        self.light_dir = direction
        self.amb = amb
        self.diff = diff
        self.is_hacking = not (isinstance(self.amb, float)
                               or isinstance(self.amb, int))

    def forward(self, feat):
        batch_size = feat.shape[0]
        if self.is_hacking:
            return torch.concat([self.light_dir, self.amb, self.diff], -1)
        else:
            return torch.concat([self.light_dir, torch.FloatTensor([self.amb, self.diff]).to(self.light_dir.device)], -1).expand(batch_size, -1)

    def shade(self, feat, kd, normal):
        light_params = self.forward(feat)
        light_dir = light_params[..., :3][:, None, None, :]
        int_amb = light_params[..., 3:4][:, None, None, :]
        int_diff = light_params[..., 4:5][:, None, None, :]
        shading = (int_amb + int_diff *
                   torch.clamp(util.dot(light_dir, normal), min=0.0))
        shaded = shading * kd
        return shaded, shading


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')


def main(args):
    assert torch.cuda.is_available(), "Cuda is required to run this script."

    resolution = (args.resolution, args.resolution)
    render_mode = args.render_mode
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    batch_size = args.batch_size
    if args.finetune_texture:
        batch_size = 1
    input_image_dir = args.input_image_dir
    
    # 0. Initialize model.
    model_cfgs = setup_runtime(args)
    model = MagicPony(model_cfgs)

    cp = torch.load(checkpoint_path)
    model.load_model_state(cp)
    epoch = cp.get('epoch', 999)
    total_iter = cp.get('total_iter', 999999)

    device = "cuda"
    model.to(device)
    model.set_eval()

    image_paths = sorted(glob.glob(osp.join(input_image_dir, "*_rgb.png")))
    save_basenames = [osp.splitext(osp.basename(p))[0].replace(
        '_rgb', "") for p in image_paths]
    total_num = len(image_paths)

    for i in tqdm(range(0, total_num, batch_size)):

        # 1. Load input images.
        images = [pil_loader(p)
                  for p in image_paths[i:min(i + batch_size, total_num + 1)]]
        input_image = torch.stack(
            [torchvision.transforms.ToTensor()(img) for img in images], dim=0).to(device)
        in_image_size = model_cfgs.get("in_image_size", 256)
        input_image = torch.nn.functional.interpolate(
            input_image, size=(in_image_size, in_image_size), mode='bilinear', align_corners=False)
        input_image = input_image[:, None, :, :]

        # 1.5. Finetune the model if needed.
        if args.finetune_texture:
            num_iters = args.finetune_iters
            lr = args.finetune_lr
            model.set_train()
            optimizer = torch.optim.Adam(model.netInstance.netTexture.parameters(), lr=lr)

            # Create single-image dataset
            single_data_dir = osp.join(input_image_dir, "single1", "0")
            os.makedirs(single_data_dir, exist_ok=True)
            shutil.copyfile(image_paths[i], osp.join(single_data_dir, "0_rgb.png"))
            shutil.copyfile(image_paths[i].replace("_rgb", "_mask"), osp.join(single_data_dir, "0_mask.png"))
            shutil.copyfile(image_paths[i].replace("_rgb.png", "_box.txt"), osp.join(single_data_dir, "0_box.txt"))
            model_cfgs["load_dino_feature"] = False
            model_cfgs["load_dino_cluster"] = False
            dataloader, _, _ = MagicPony.get_data_loaders(model_cfgs, train_data_dir=osp.dirname(single_data_dir))

            for _ in range(num_iters):
                for batch in dataloader:
                    optimizer.zero_grad()
                    _ = model.forward(batch, epoch=epoch, total_iter=total_iter, is_training=True)
                    model.total_loss.backward()
                    model.total_loss = 0
                    optimizer.step()

            shutil.rmtree(single_data_dir, ignore_errors=True)

        with torch.no_grad():

            # 2. Run model and save renderings.
            prior_shape, dino_pred = model.netPrior(is_training=False)
            shape, _, _, mvp, w2c, campos, texture_pred, im_features, deform, all_arti_params, light, forward_aux = \
                model.netInstance(input_image, prior_shape,
                                  epoch, total_iter, is_training=False)

            if "input_view" in render_mode:
                shaded, shading, albedo = \
                    model.render(["shaded", "shading", "kd"], shape, texture_pred, mvp, w2c, campos, resolution,
                                 im_features=im_features, light=light, prior_shape=prior_shape,
                                 dino_net=dino_pred, spp=4, num_frames=1)
                image_pred = shaded[:, :3, :, :]
                mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                shading = shading.expand_as(image_pred)
                save_images(input_image.squeeze(1), None, save_basenames[i:min(
                    i+batch_size, total_num + 1)], output_dir, suffix="_input_image")
                save_images(image_pred, mask_pred, save_basenames[i:min(
                    i+batch_size, total_num + 1)], output_dir, suffix="_input_view_textured")
                if shading is not None:
                    save_images(shading / 2, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_input_view_shading")

                gray_light = FixedDirectionLight(direction=torch.FloatTensor(
                    [0, 0, 1]).to(device), amb=0.2, diff=0.7)
                shaded, shading = \
                    model.render(["shaded", "shading"], shape, texture_pred, mvp, w2c, campos, resolution,
                                 im_features=im_features, light=gray_light, prior_shape=prior_shape,
                                 dino_net=dino_pred, spp=4)
                
                mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                shading = shading.expand_as(image_pred)
                if shading is not None:
                    save_images(shading, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_input_view_mesh")

            if "other_views" in render_mode:
                current_batch = shape.v_pos.shape[0]
                pose_canon = torch.concat(
                    [torch.eye(3), torch.zeros(1, 3)], dim=0).view(-1)[None].to(device)
                mvp_canon, w2c_canon, campos_canon = model.netInstance.get_camera_extrinsics_from_pose(
                    pose_canon, offset_extra=5.5)
                
                instance_rotate_angles = [torch.FloatTensor([0, angle, 0]) / 180 * np.pi for angle in range(0, 360, 30)]
                
                gray_light = FixedDirectionLight(direction=torch.FloatTensor(
                    [0, 0, 1]).to(device), amb=0.2, diff=0.7)
                for angle_id, rot_angle in enumerate(instance_rotate_angles):
                    mtx = torch.eye(4).to(device)
                    mtx[:3, :3] = euler_angles_to_matrix(rot_angle, "XYZ")
                    cur_w2c = torch.matmul(w2c_canon, mtx[None])
                    cur_mvp = torch.matmul(mvp_canon, mtx[None])
                    cur_campos = campos_canon @ torch.linalg.inv(mtx[:3, :3]).T

                    shaded, shading, albedo = \
                        model.render(["shaded", "shading", "kd"], shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                    im_features=im_features, light=gray_light, prior_shape=prior_shape,
                                    dino_net=dino_pred, spp=4, num_frames=1)
                    image_pred = shaded[:, :3, :, :]
                    mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                    shading = shading.expand_as(image_pred)
                    save_images(shading, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_other_view_mesh_%d" % angle_id, mode="transparent")

                    _ = model.render(["shaded"], shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                     im_features=im_features, light=light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4)

                    ori_light_dir = light.light_params[..., :3]
                    final_dir = torch.matmul(ori_light_dir, w2c[:, :3, :3])
                    final_dir = torch.matmul(
                        final_dir, cur_w2c[:, :3, :3].transpose(2, 1))[:, 0]
                    amb, diff = light.light_params[...,
                                                   3:4], light.light_params[..., 4:5]
                    cur_light = FixedDirectionLight(
                        direction=final_dir, amb=amb, diff=diff)

                    shaded, shading = \
                        model.render(["shaded", "shading"], shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                     im_features=im_features, light=cur_light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4)
                    image_pred = shaded[:, :3, :, :]
                    mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                    save_images(image_pred, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_other_view_textured_%d" % angle_id, mode="transparent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str,
                        help='Specify a GPU device')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Specify the number of worker threads for data loaders')
    parser.add_argument('--seed', default=0, type=int,
                        help='Specify a random seed')
    parser.add_argument('--input_image_dir', required=True, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--config', default=None,
                        type=str)  # Model config path
    parser.add_argument('--checkpoint_path', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--render_mode', nargs="+", type=str, default=["input_view", "other_views"])
    parser.add_argument('--video', action="store_true")
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--finetune_texture', action="store_true")
    parser.add_argument('--finetune_iters', default=200, type=int)
    parser.add_argument('--finetune_lr', default=0.001, type=float)
    args = parser.parse_args()
    main(args)
