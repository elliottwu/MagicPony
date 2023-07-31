import os
from glob import glob
import random
import numpy as np
import re
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.datasets.folder
import torchvision.transforms as transforms
from einops import rearrange


def compute_distance_transform(mask):
    mask_dt = []
    for m in mask:
        dt = torch.FloatTensor(cv2.distanceTransform(np.uint8(m[0]), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        inv_dt = torch.FloatTensor(cv2.distanceTransform(np.uint8(1 - m[0]), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        mask_dt += [torch.stack([dt, inv_dt], 0)]
    return torch.stack(mask_dt, 0)  # Bx2xHxW


def crop_image(image, boxs, size):
    crops = []
    for box in boxs:
        crop_x0, crop_y0, crop_w, crop_h = box
        crop = transforms.functional.resized_crop(image, crop_y0, crop_x0, crop_h, crop_w, size)
        crop = transforms.functional.to_tensor(crop)
        crops += [crop]
    return torch.stack(crops, 0)  # BxCxHxW


def box_loader(fpath):
    box = np.loadtxt(fpath, 'str')
    box[0] = box[0].split('_')[0]
    return box.astype(np.float32)


def read_feat_from_img(path, n_channels):
    feat = np.array(Image.open(path))
    return dencode_feat_from_img(feat, n_channels)


def dencode_feat_from_img(img, n_channels):
    n_addon_channels = int(np.ceil(n_channels / 3) * 3) - n_channels
    n_tiles = int((n_channels + n_addon_channels) / 3)
    feat = rearrange(img, 'h (t w) c -> h w (t c)', t=n_tiles, c=3)
    feat = feat[:, :, :-n_addon_channels]
    feat = feat.astype('float32') / 255
    return feat.transpose(2, 0, 1)  # CxHxW


def dino_loader(fpath, n_channels):
    dino_map = read_feat_from_img(fpath, n_channels)
    return dino_map


def get_valid_mask(boxs, image_size):
    valid_masks = []
    for box in boxs:
        crop_x0, crop_y0, crop_w, crop_h, full_w, full_h = box[1:7].int().numpy()
        margin_w = int(crop_w * 0.02)  # discard a small margin near the boundary
        margin_h = int(crop_h * 0.02)
        mask_full = torch.ones(full_h-margin_h*2, full_w-margin_w*2)
        mask_full_pad = torch.nn.functional.pad(mask_full, (crop_w+margin_w, crop_w+margin_w, crop_h+margin_h, crop_h+margin_h), mode='constant', value=0.0)
        mask_full_crop = mask_full_pad[crop_y0+crop_h:crop_y0+crop_h*2, crop_x0+crop_w:crop_x0+crop_w*2]
        mask_crop = torch.nn.functional.interpolate(mask_full_crop[None, None, :, :], image_size, mode='nearest')[0,0]
        valid_masks += [mask_crop]
    return torch.stack(valid_masks, 0)  # NxHxW


def horizontal_flip_box(box):
    frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = box.unbind(1)
    box[:,1] = full_w - crop_x0 - crop_w  # x0
    return box


def none_to_nan(x):
    return torch.FloatTensor([float('nan')]) if x is None else x


class BaseSequenceDataset(Dataset):
    def __init__(self, root, skip_beginning=4, skip_end=4, min_seq_len=10):
        super().__init__()

        self.skip_beginning = skip_beginning
        self.skip_end = skip_end
        self.min_seq_len = min_seq_len
        self.sequences = self._make_sequences(root)
        self.samples = []

    def _make_sequences(self, path):
        result = []
        for d in sorted(os.scandir(path), key=lambda e: e.name):
            if d.is_dir():
                files = self._parse_folder(d)
                if len(files) >= self.min_seq_len:
                    result.append(files)
        return result

    def _parse_folder(self, path):
        image_path_suffix = self.image_loader[0]
        result = sorted(glob(os.path.join(path, '*'+image_path_suffix)))
        if '*' in image_path_suffix:
            image_path_suffix = re.findall(image_path_suffix, result[0])[0]
            self.image_loader[0] = image_path_suffix
        result = [p.replace(image_path_suffix, '{}') for p in result]

        if len(result) <= self.skip_beginning + self.skip_end:
            return []
        if self.skip_end == 0:
            return result[self.skip_beginning:]
        return result[self.skip_beginning:-self.skip_end]

    def _load_ids(self, path_patterns, loader, transform=None):
        result = []
        for p in path_patterns:
            x = loader[1](p.format(loader[0]), *loader[2:])
            if transform:
                x = transform(x)
            result.append(x)
        return tuple(result)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise NotImplemented("This is a base class and should not be used directly")


class NFrameSequenceDataset(BaseSequenceDataset):
    def __init__(self, root, num_frames=2, skip_beginning=4, skip_end=4, min_seq_len=10, in_image_size=256, out_image_size=256, random_sample=False, dense_sample=True, shuffle=False, load_flow=False, load_background=False, random_xflip=False, load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64):
        self.image_loader = ["rgb.*", torchvision.datasets.folder.default_loader]
        self.mask_loader = ["mask.png", torchvision.datasets.folder.default_loader]
        self.bbox_loader = ["box.txt", box_loader]
        super().__init__(root, skip_beginning, skip_end, min_seq_len)
        if load_flow and num_frames > 1:
            self.flow_loader = ["flow.png", cv2.imread, cv2.IMREAD_UNCHANGED]
        else:
            self.flow_loader = None

        self.num_frames = num_frames
        self.random_sample = random_sample
        if self.random_sample:
            self.samples = self.sequences
        else:
            for i, s in enumerate(self.sequences):
                stride = 1 if dense_sample else self.num_frames
                self.samples += [(i, k) for k in range(0, len(s), stride)]
        if shuffle:
            random.shuffle(self.samples)
        
        self.in_image_size = in_image_size
        self.out_image_size = out_image_size
        self.image_transform = transforms.Compose([transforms.Resize(self.in_image_size, interpolation=Image.BILINEAR), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(self.out_image_size, interpolation=Image.NEAREST), transforms.ToTensor()])
        if self.flow_loader is not None:
            def flow_transform(x):
                x = torch.FloatTensor(x.astype(np.float32)).flip(2)[:,:,:2]  # HxWx2
                x = x / 65535. * 2 - 1  # -1~1
                x = torch.nn.functional.interpolate(x.permute(2,0,1)[None], size=self.out_image_size, mode="bilinear")[0]  # 2xHxW
                return x
            self.flow_transform = flow_transform
        self.load_dino_feature = load_dino_feature
        if load_dino_feature:
            self.dino_feature_loader = [f"feat{dino_feature_dim}.png", dino_loader, dino_feature_dim]
        self.load_dino_cluster = load_dino_cluster
        if load_dino_cluster:
            self.dino_cluster_loader = ["clusters.png", torchvision.datasets.folder.default_loader]
        self.load_flow = load_flow
        self.load_background = load_background
        self.random_xflip = random_xflip

    def __getitem__(self, index):
        if self.random_sample:
            seq_idx = index % len(self.samples)
            seq = self.samples[seq_idx]
            if len(seq) < self.num_frames:
                start_frame_idx = 0
            else:
                start_frame_idx = np.random.randint(len(seq)-self.num_frames+1)
        else:
            seq_idx, start_frame_idx = self.samples[index % len(self.samples)]
            seq = self.sequences[seq_idx]
            ## handle edge case: when only last frame is left, sample last two frames, except if the sequence only has one frame
            if len(seq) <= start_frame_idx +1:
                start_frame_idx = max(0, start_frame_idx-1)
        
        paths = seq[start_frame_idx:start_frame_idx+self.num_frames]  # length can be shorter than num_frames
        images = torch.stack(self._load_ids(paths, self.image_loader, transform=self.image_transform), 0)  # load all images
        masks = torch.stack(self._load_ids(paths, self.mask_loader, transform=self.mask_transform), 0)  # load all images
        mask_dt = compute_distance_transform(masks)
        bboxs = torch.stack(self._load_ids(paths, self.bbox_loader, transform=torch.FloatTensor), 0)   # load bounding boxes for all images
        mask_valid = get_valid_mask(bboxs, (self.out_image_size, self.out_image_size))  # exclude pixels cropped outside the original image
        if self.load_flow and len(paths) > 1:
            flows = torch.stack(self._load_ids(paths[:-1], self.flow_loader, transform=self.flow_transform), 0)  # load flow from current frame to next, (N-1)x(x,y)xHxW, -1~1
        else:
            flows = None
        if self.load_background:
            bg_fpath = os.path.join(os.path.dirname(paths[0]), 'background_frame.jpg')
            assert os.path.isfile(bg_fpath)
            bg_image = torchvision.datasets.folder.default_loader(bg_fpath)
            bg_images = crop_image(bg_image, bboxs[:, 1:5].int().numpy(), (self.out_image_size, self.out_image_size))
        else:
            bg_images = None
        if self.load_dino_feature:
            dino_features = torch.stack(self._load_ids(paths, self.dino_feature_loader, transform=torch.FloatTensor), 0)  # Fx64x224x224
        else:
            dino_features = None
        if self.load_dino_cluster:
            dino_clusters = torch.stack(self._load_ids(paths, self.dino_cluster_loader, transform=transforms.ToTensor()), 0)  # Fx3x55x55
        else:
            dino_clusters = None
        seq_idx = torch.LongTensor([seq_idx])
        frame_idx = torch.arange(start_frame_idx, start_frame_idx+len(paths)).long()

        ## random horizontal flip
        if self.random_xflip and np.random.rand() < 0.5:
            xflip = lambda x: None if x is None else x.flip(-1)
            images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters = (*map(xflip, (images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters)),)
            if flows is not None:
                flows[:,0] *= -1  # invert delta x
            bboxs = horizontal_flip_box(bboxs)  # NxK

        ## pad shorter sequence
        if len(paths) < self.num_frames:
            num_pad = self.num_frames - len(paths)
            pad_front = lambda x: None if x is None else torch.cat([x[:1]] *num_pad + [x], 0)
            images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, frame_idx = (*map(pad_front, (images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, frame_idx)),)
            if flows is not None:
                flows[:num_pad] = 0  # setting flow to zeros for replicated frames

        out = (*map(none_to_nan, (images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, seq_idx, frame_idx)),)  # for batch collation
        return out


def get_sequence_loader(data_dir, batch_size=256, num_workers=4, in_image_size=256, out_image_size=256, num_frames=2, skip_beginning=4, skip_end=4, min_seq_len=10, random_sample=False, dense_sample=True, shuffle=False, load_flow=False, load_background=False, random_xflip=False, load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64):
    dataset = NFrameSequenceDataset(data_dir, num_frames=num_frames, skip_beginning=skip_beginning, skip_end=skip_end, min_seq_len=min_seq_len, in_image_size=in_image_size, out_image_size=out_image_size, random_sample=random_sample, dense_sample=dense_sample, shuffle=shuffle, load_flow=load_flow, load_background=load_background, random_xflip=random_xflip, load_dino_feature=load_dino_feature, load_dino_cluster=load_dino_cluster, dino_feature_dim=dino_feature_dim)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


class ImageDataset(Dataset):
    def __init__(self, root, in_image_size=256, out_image_size=256, shuffle=False, load_background=False, random_xflip=False, load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64):
        super().__init__()
        self.image_loader = ["rgb.*", torchvision.datasets.folder.default_loader]
        self.mask_loader = ["mask.png", torchvision.datasets.folder.default_loader]
        self.bbox_loader = ["box.txt", box_loader]
        self.samples = self._parse_folder(root)
        if shuffle:
            random.shuffle(self.samples)
        self.in_image_size = in_image_size
        self.out_image_size = out_image_size
        self.image_transform = transforms.Compose([transforms.Resize(self.in_image_size, interpolation=Image.BILINEAR), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(self.out_image_size, interpolation=Image.NEAREST), transforms.ToTensor()])
        self.load_dino_feature = load_dino_feature
        if load_dino_feature:
            self.dino_feature_loader = [f"feat{dino_feature_dim}.png", dino_loader, dino_feature_dim]
        self.load_dino_cluster = load_dino_cluster
        if load_dino_cluster:
            self.dino_cluster_loader = ["clusters.png", torchvision.datasets.folder.default_loader]
        self.load_background = load_background
        self.random_xflip = random_xflip

    def _parse_folder(self, path):
        image_path_suffix = self.image_loader[0]
        result = sorted(glob(os.path.join(path, '**/*'+image_path_suffix), recursive=True))
        if '*' in image_path_suffix:
            image_path_suffix = re.findall(image_path_suffix, result[0])[0]
            self.image_loader[0] = image_path_suffix
        result = [p.replace(image_path_suffix, '{}') for p in result]
        return result

    def _load_ids(self, path, loader, transform=None):
        x = loader[1](path.format(loader[0]), *loader[2:])
        if transform:
            x = transform(x)
        return x

    def __len__(self):
        return len(self.samples)
    
    def set_random_xflip(self, random_xflip):
        self.random_xflip = random_xflip

    def __getitem__(self, index):
        path = self.samples[index % len(self.samples)]
        images = self._load_ids(path, self.image_loader, transform=self.image_transform).unsqueeze(0)
        masks = self._load_ids(path, self.mask_loader, transform=self.mask_transform).unsqueeze(0)
        mask_dt = compute_distance_transform(masks)
        bboxs = self._load_ids(path, self.bbox_loader, transform=torch.FloatTensor).unsqueeze(0)
        mask_valid = get_valid_mask(bboxs, (self.out_image_size, self.out_image_size))  # exclude pixels cropped outside the original image
        flows = None
        if self.load_background:
            bg_fpath = os.path.join(os.path.dirname(path), 'background_frame.jpg')
            assert os.path.isfile(bg_fpath)
            bg_image = torchvision.datasets.folder.default_loader(bg_fpath)
            bg_images = crop_image(bg_image, bboxs[:, 1:5].int().numpy(), (self.out_image_size, self.out_image_size))
        else:
            bg_images = None
        if self.load_dino_feature:
            dino_features = torch.stack(self._load_ids(path, self.dino_feature_loader, transform=torch.FloatTensor), 0)  # 64x224x224
        else:
            dino_features = None
        if self.load_dino_cluster:
            dino_clusters = torch.stack(self._load_ids(path, self.dino_cluster_loader, transform=transforms.ToTensor()), 0)  # 3x55x55
        else:
            dino_clusters = None
        seq_idx = torch.LongTensor([index])
        frame_idx = torch.LongTensor([0])

        ## random horizontal flip
        if self.random_xflip and np.random.rand() < 0.5:
            xflip = lambda x: None if x is None else x.flip(-1)
            images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters = (*map(xflip, (images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters)),)
            bboxs = horizontal_flip_box(bboxs)  # NxK

        out = (*map(none_to_nan, (images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, seq_idx, frame_idx)),)  # for batch collation
        return out


def get_image_loader(data_dir, batch_size=256, num_workers=4, in_image_size=256, out_image_size=256, shuffle=False, load_background=False, random_xflip=False, load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64):
    dataset = ImageDataset(data_dir, in_image_size=in_image_size, out_image_size=out_image_size, load_background=load_background, random_xflip=random_xflip, load_dino_feature=load_dino_feature, load_dino_cluster=load_dino_cluster, dino_feature_dim=dino_feature_dim)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
