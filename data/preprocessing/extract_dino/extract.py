import os
from os import path as osp
import logging
from pathlib import Path
from glob import glob

import configargparse
import faiss
import numpy as np
import torch
from PIL import Image
import PIL
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange

from extractor import ViTExtractor
from feature_storage import save_feat_as_img


logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%I:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)


def arrange(images):
    rows = []
    for row in images:
        rows += [np.concatenate(row, axis=1)]
    image = np.concatenate(rows, axis=0)
    return image


def create_dataset(root, debug_subset=False, n_random_samples=None, img_postfix='_rgb.jpg', mask_postfix=None, transform=None, transform_no_norm=None, mask_transform=None):
    img_names = sorted(glob(osp.join(root, "*", "*" + img_postfix)))
    if mask_postfix is not None:
        mask_names = sorted(list(Path(root).rglob('*' + mask_postfix)))
        assert len(img_names) == len(mask_names)
    else:
        mask_names = None
    dataset = FilesListDataset(img_names, masks_paths=mask_names, transform=transform, transform_no_norm=transform_no_norm, mask_transform=mask_transform)
    if n_random_samples:
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), n_random_samples))
    if debug_subset:
        dataset = torch.utils.data.Subset(dataset, np.arange(50))
    return dataset


def create_dataset_from_multiple(roots, img_postfix, mask_postfix, **kwargs):
    if not isinstance(img_postfix, list):
        img_postfix = len(roots) * [img_postfix]
    if not isinstance(mask_postfix, list):
        mask_postfix = len(roots) * [mask_postfix]
    datasets = [create_dataset(root, img_postfix=img_postfix_, mask_postfix=mask_postfix_, **kwargs) for root, img_postfix_, mask_postfix_ in zip(roots, img_postfix, mask_postfix)]
    return torch.utils.data.ConcatDataset(datasets)


def create_dataloader(root, batch_size, img_postfix, mask_postfix, shuffle=False, **kwargs):
    dataset = create_dataset_from_multiple(root, img_postfix, mask_postfix, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def save_features(features_out_dir, img_path, img_postfix, name_depth, features_out, save_features_as_npy, dim_in_filename=False):
    name = img_path.split('/')[-1][:-len(img_postfix)]
    dirs = img_path.split('/')[-(name_depth + 1):-1]
    file_name = f'{name}_feat'
    if dim_in_filename:
        file_name += f'{features_out.shape[-1]}'
    if features_out_dir is None:
        # save the image features in the same directory as the input image
        out_path = Path(img_path).parent / file_name
    else:
        out_path = Path(os.path.join(features_out_dir, *dirs, file_name))
    out_path.parent.mkdir(exist_ok=True, parents=True)
    features_out_bad_idx = ((np.sum(np.absolute(features_out) >= 1, axis=-1)) > 0)
    features_out[features_out_bad_idx] = features_out[features_out_bad_idx] / np.linalg.norm(features_out[features_out_bad_idx], axis=-1, keepdims=True)
    save_feat_as_img(str(out_path) + '.png', features_out / np.linalg.norm(features_out, axis=-1, keepdims=True))
    if save_features_as_npy:
        np.save(str(out_path) + '.npy', features_out)


def save_visuals(img_path, img_postfix, img_no_norm, name_depth, features_out, image_size, mask, vis_out_dir):
    name = img_path.split('/')[-1][:-len(img_postfix)]
    img_np = img_no_norm.permute(1, 2, 0).cpu().numpy() * 255
    features_out = torch.nn.functional.interpolate(torch.from_numpy(features_out[None]).permute(0, 3, 1, 2), size=[image_size, image_size]).permute(0, 2, 3, 1).squeeze(0).numpy()
    features_out = (features_out + 1) / 2
    features_list = [features_out[..., k * 3:k * 3 + 3] * 255 for k in range(min(features_out.shape[-1], 9) // 3)]
    visual = [img_np] + features_list

    if mask is not None:
        mask = torch.nn.functional.interpolate(torch.from_numpy(mask[None]), size=[image_size, image_size]).permute(0, 2, 3, 1).squeeze(0).numpy()
        mask = np.repeat(mask, 3, axis=-1)
        visual += [mask * 255]

    visual = arrange([visual])

    Path(vis_out_dir).mkdir(exist_ok=True, parents=True)
    filename = f'{name}.png'
    Image.fromarray((visual).astype(np.uint8)).save(Path(vis_out_dir) / filename)


def extract_features(args, dataloader, model_type, model_path, device, load_mask=False, stride=4, facet='key', layer=11, pca_mat=None, log_features=False, features_out_dir=None, vis_out_dir=None, img_postfix='_rgb.jpg'):
    with torch.no_grad():
        extractor = ViTExtractor(model_type=model_type, stride=stride, model_path=model_path, device=device)

        all_features = []
        all_masks = []

        i = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            img, img_no_norm, img_path = (batch[k] for k in ['img', 'img_no_norm', 'img_path'])
            img = img.to(device)

            img_padded = torch.nn.functional.pad(img, 4 * [args.img_pad], mode='reflect')
            features = extractor.extract_descriptors(img_padded, facet=facet, layer=layer, bin=False)
            features = features.reshape(-1, features.shape[-1]).cpu().numpy()
            if args.normalize_features:
                features = features / np.linalg.norm(features, axis=-1, keepdims=True)

            if load_mask:
                mask = batch['mask']
                mask = torch.nn.functional.interpolate(mask, size=extractor.num_patches).cpu().numpy()

            # export image features and save visuals
            if pca_mat is not None:
                features_out = pca_mat.apply_py(features)
            else:
                features_out = features.copy()
            features_out = features_out.reshape(-1, *extractor.num_patches, features_out.shape[-1])
            for j, img_no_norm_ in enumerate(img_no_norm):
                # save features
                if features_out_dir is not None:
                    save_features(features_out_dir, img_path[j], img_postfix, args.name_depth, features_out[j], args.save_features_as_npy, dim_in_filename=args.dim_in_filename)
                # visuals
                save_visuals_frq = max(int(len(dataloader.dataset) / args.n_images_vis), 1)
                if vis_out_dir is not None and i % save_visuals_frq == 0:
                    mask_ = mask[j] if load_mask else None
                    save_visuals(img_path[j], img_postfix, img_no_norm[j], args.name_depth, features_out[j], args.image_size, mask_, vis_out_dir)
                i += 1

            # log features
            if log_features:
                all_features += [features]
                mask = rearrange(mask, 'b 1 h w -> b (h w)')
                all_masks += [mask]

        if log_features:
            all_features = np.concatenate(all_features, axis=0)
            if all_masks:
                all_masks = np.concatenate(all_masks, axis=0)

        return all_features, all_masks


class FilesListDataset(torch.utils.data.Dataset):
    def __init__(self, files_list, masks_paths=None, transform=transforms.ToTensor(), transform_no_norm=transforms.ToTensor(), mask_transform=transforms.ToTensor(),):
        self.files_list = files_list
        self.masks_paths = masks_paths
        self.transform = transform
        self.transform_no_norm = transform_no_norm
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img_path = self.files_list[idx]
        try:
            img = Image.open(img_path)
        except PIL.UnidentifiedImageError as e:
            logger.warning(f'Error: {e}')
            img = Image.new('RGB', (256, 256))
        img_transform = self.transform(img)
        img_transform_no_norm = self.transform_no_norm(img)
        result = {'img': img_transform, 'img_no_norm': img_transform_no_norm, 'img_path': str(img_path)}

        if self.masks_paths is not None:
            mask_path = self.masks_paths[idx]
            mask = Image.open(mask_path)
            mask_transformed = self.mask_transform(mask)[:1]
            result['mask'] = mask_transformed

        return result


def main(args):
    exp_name = args.exp_name
    model_type = args.model_type
    model_path = args.model_path
    image_size = args.image_size
    batch_size = args.batch_size
    device = args.device
    features_out_dir = Path(args.features_out_root) / exp_name
    pca_out_path = Path(args.results_info_root) / exp_name / 'pca.faiss'
    vis_out_dir = Path(args.vis_out_root) / exp_name

    image_norm_params = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
    interpolation = 3

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(image_norm_params['mean']),
            std=torch.tensor(image_norm_params['std']))
    ])
    transform_no_norm = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation),
        transforms.ToTensor(),
    ])

    if args.use_pca:
        if args.load_pca_path:
            # check if we were provided with a path PCA matrix
            logger.info(f'Loading PCA from {args.load_pca_path}')
            pca_mat = faiss.read_VectorTransform(args.load_pca_path)
        else:
            # if not -> train PCA on training data (pca data)
            mask_postfix_train = args.mask_postfix_train if args.load_mask else None
            train_dataloader = create_dataloader(
                args.train_root, batch_size, args.img_postfix_train, mask_postfix_train, 
                n_random_samples=args.n_train_random_samples, debug_subset=args.debug_subset, 
                shuffle=args.shuffle_data, 
                transform=transform, transform_no_norm=transform_no_norm, mask_transform=mask_transform)

            logger.info('Extracting features for PCA')
            features, masks = extract_features(
                args, train_dataloader, model_type, model_path, device,
                load_mask=args.load_mask, stride=args.stride, facet=args.facet, layer=args.layer,
                log_features=True)
            logger.info('Extracting features for PCA done')

            if args.load_mask:
                masks = masks > 0
                features = features * masks.reshape(-1, 1)

            logger.info('PCA on features')
            if args.load_pca_path:
                logger.info(f'Loading PCA from {args.load_pca_path}')
                pca_mat = faiss.read_VectorTransform(args.load_pca_path)
            else:
                features_dim = features.shape[1]
                pca_dim = args.pca_dim
                pca_mat = faiss.PCAMatrix(features_dim, pca_dim)
                pca_mat.train(features)
            assert pca_mat.is_trained
            logger.info('PCA on features done')

            # save pca mat
            pca_out_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_VectorTransform(pca_mat, str(pca_out_path))
    else:
        pca_mat = None

    # extract and save features
    mask_postfix_test = args.mask_postfix_test if args.load_mask else None
    test_dataloader = create_dataloader(
        args.test_root, batch_size, args.img_postfix_test, mask_postfix_test,
        debug_subset=args.debug_subset, shuffle=args.shuffle_data, transform=transform,
        transform_no_norm=transform_no_norm, mask_transform=mask_transform)

    logger.info('Extracting features')
    extract_features(
        args, test_dataloader, model_type, model_path, device, load_mask=args.load_mask,
        stride=args.stride, facet=args.facet, layer=args.layer, pca_mat=pca_mat,
        features_out_dir=features_out_dir, vis_out_dir=vis_out_dir, img_postfix=args.img_postfix_test)
    logger.info('Extracting features done')


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', type=str, is_config_file=True, help='Specify a config file path')
    parser.add_argument('--device', default='cuda:0', type=str, help='')
    parser.add_argument('--train_root', required=True, type=str, nargs='+', help='')
    parser.add_argument('--test_root', required=True, type=str, nargs='+', help='')
    parser.add_argument('--exp_name', required=True, type=str, help='')
    parser.add_argument('--results_info_root', required=True, type=str, help='')
    parser.add_argument('--features_out_root', default=None, type=str, help='')
    parser.add_argument('--model_type', default='dino_vits8', type=str, help='')
    parser.add_argument('--vis_out_root', default=None, type=str, help='')
    parser.add_argument('--name_depth', default=None, type=int, help='')
    parser.add_argument('--model_path', default=None, type=str, help='')
    parser.add_argument('--load_pca_path', default=None, type=str, help='')
    parser.add_argument('--img_postfix_train', default='_rgb.jpg', nargs='+', type=str, help='')
    parser.add_argument('--img_postfix_test', default='_rgb.jpg', nargs='+', type=str, help='')
    parser.add_argument('--mask_postfix_train', default='_mask.png', nargs='+', type=str, help='')
    parser.add_argument('--mask_postfix_test', default='_mask.png', nargs='+', type=str, help='')
    parser.add_argument('--image_size', default=224, type=int, help='')
    parser.add_argument('--img_pad', default=0, type=int, help='')
    parser.add_argument('--stride', default=4, type=int, help='')
    parser.add_argument('--facet', default='key', type=str, help='')
    parser.add_argument('--layer', default=11, type=int, help='')
    parser.add_argument('--batch_size', default=16, type=int, help='')
    parser.add_argument('--use_pca', action='store_true', help='')
    parser.add_argument('--pca_dim', default=9, type=int, help='')
    parser.add_argument('--save_features_as_npy', action='store_true', help='')
    parser.add_argument('--dim_in_filename', action='store_true', help='')
    parser.add_argument('--load_mask', action='store_true', help='')
    parser.add_argument('--normalize_features', action='store_true', help='')
    parser.add_argument('--debug_subset', action='store_true', help='')
    parser.add_argument('--shuffle_data', action='store_true', help='')
    parser.add_argument('--n_images_vis', default=1000, type=int, help='')
    parser.add_argument('--n_train_random_samples', default=5000, type=int, help='')
    args, _ = parser.parse_known_args()

    main(args)
