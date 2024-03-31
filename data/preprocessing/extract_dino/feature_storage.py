import numpy as np
from einops import rearrange
from PIL import Image
from pathlib import Path


def save_feat_as_img(out_path, feat):
    feat_enc = encode_feat_to_img(feat)
    Image.fromarray(feat_enc).save(out_path)


def read_feat_from_img(path, n_channels):
    feat = np.array(Image.open(path))
    return dencode_feat_from_img(feat, n_channels)


def encode_feat_to_img(feat):
    """
    Assumes features normalized between -1 and 1

    Returns 3 channel image as uint8 array with features tiled across the width axis
    """
    #  convert to byte int8
    assert feat.min() >= -1 and feat.max() <= 1
    feat = np.round(((feat + 1) * 127)).astype('uint8')

    # append channels to make it devisible by 3
    n_channels = feat.shape[2]
    n_addon_channels = int(np.ceil(n_channels / 3) * 3) - n_channels
    feat = np.concatenate([feat, np.zeros([feat.shape[0], feat.shape[0], n_addon_channels], dtype=feat.dtype)], axis=-1)

    # tile to form an image
    feat = rearrange(feat, 'h w (t c) -> h (t w) c', c=3)

    return feat


def dencode_feat_from_img(img, n_channels):
    n_addon_channels = int(np.ceil(n_channels / 3) * 3) - n_channels
    n_tiles = int((n_channels + n_addon_channels) / 3)

    feat = rearrange(img, 'h (t w) c -> h w (t c)', t=n_tiles, c=3)
    feat = feat[:, :, :-n_addon_channels]

    feat = feat.astype('float32') / 127 - 1

    return feat


def test():
    # %%
    feat_path = '/scratch/shared/beegfs/tomj/dove/datasets/cub/dino/cub+birds-20c-4s-5k_rnd-norm-pca64-3/preprocessed-all/train/004.Groove_billed_Ani/0000092_feat64.npy'
    out_path = 'feat.png'
    n_channels = 64

    # %%
    feat = np.load(feat_path)
    feat_orig = feat.copy()

    feat_enc = encode_feat_to_img(feat)
    Image.fromarray(feat_enc).save(out_path)

    # %%
    # decode
    feat = np.array(Image.open(out_path))
    feat = dencode_feat_from_img(feat, n_channels)
    print(np.allclose(feat, feat_orig, rtol=0, atol=1e-2, equal_nan=False))


if __name__ == "__main__":
    from tqdm import tqdm

    # %%
    root = '/scratch/shared/beegfs/tomj/dove/datasets/cub/dino/cub+birds-20c-4s-5k_rnd-norm-pca64-3'
    feat_file_suffix = '_feat64.npy'
    enc_feat_file_suffix = '_feat64.png'

    # %%
    feat_names = sorted(list(Path(root).rglob(f'*{feat_file_suffix}')))

    # %%

    for feat_path in tqdm(feat_names):
        feat = np.load(feat_path)
        feat_enc = encode_feat_to_img(feat)
        out_path = str(feat_path)[:-len(feat_file_suffix)] + enc_feat_file_suffix
        Image.fromarray(feat_enc).save(out_path)
