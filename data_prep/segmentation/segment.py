import argparse
import cv2
from glob import glob
import numpy as np
import os
from os import path as osp
from PIL import Image
import torch
from tqdm import tqdm

from GroundingDINO.groundingdino.datasets import transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from segment_anything import build_sam, SamPredictor

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def segment_images(source_dir, target_dir, category, with_example=False, pass_model=False, gd_model=None, sam_model=None, rename=False, prefix=['.png']):
    config_file = '/path/to/GroundingDINO_config_file.py'  # e.g., 'utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    grounded_checkpoint = '/path/to/groundingdino_ckpt.pth'  # e.g., 'groundingdino_swint_ogc.pth'
    sam_checkpoint = '/path/to/sam_ckpt.pth'  # e.g., 'sam_vit_h_4b8939.pth'
    box_threshold = 0.3
    text_threshold = 0.25
    device = 'cuda'
    
    if pass_model:
        assert gd_model is not None
        assert sam_model is not None
        
        model = gd_model
        predictor = sam_model
    else:
        model = load_model(config_file, grounded_checkpoint, device)
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    image_paths = []
    if '.png' in prefix:
        image_paths += sorted(glob(osp.join(source_dir, "*.png")))
    if '.jpg' in prefix:
        image_paths += sorted(glob(osp.join(source_dir, "*.jpg")))

    for idx, image_path in tqdm(enumerate(image_paths)):
        try:
            image_pil, image = load_image(image_path)
        except:
            continue
        
        # Run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, category, box_threshold, text_threshold, device=device
        )

        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            continue
        predictor.set_image(image)

        W, H = image_pil.size[:2]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        boxes_filt = boxes_filt[:1]
        if boxes_filt.shape[0] == 0:
            continue
        
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )

        mask = masks[0, 0].cpu().int().numpy()
        
        crop_size = int((mask.sum()) ** 0.5)
        ys, xs = np.where(mask > 0.5)
        wc = int(xs.mean())
        hc = int(ys.mean())

        mask_pad = cv2.copyMakeBorder(mask, crop_size, crop_size, crop_size, crop_size, cv2.BORDER_CONSTANT, 0)
        mask_crop = mask_pad[hc:hc+crop_size*2, wc:wc+crop_size*2]
        mask_crop = np.uint8(np.repeat(mask_crop[:,:,None], 3, axis=2))
        mask_crop_resized = cv2.resize(mask_crop, (256, 256), interpolation=cv2.INTER_NEAREST)

        img_pad = cv2.copyMakeBorder(np.array(image_pil), crop_size, crop_size, crop_size, crop_size, cv2.BORDER_CONSTANT, 0)
        img_crop = img_pad[hc:hc+crop_size*2, wc:wc+crop_size*2]
        
        # img_crop_resized = cv2.resize(img_crop, (256, 256), interpolation=cv2.INTER_LINEAR)
        # use PIL's safe downsampling
        pil_img_crop = Image.fromarray(img_crop)
        pil_img_crop_resized = pil_img_crop.resize((256, 256), Image.BILINEAR)
        img_crop_resized = np.array(pil_img_crop_resized)

        img_crop_resized = cv2.cvtColor(img_crop_resized, cv2.COLOR_RGB2BGR)

        if rename:
            category_of_img = image_path.split('/')[-2]
            name_start = f"{category_of_img}"
            name_end = "_%06d.png" % idx
            name = name_start + name_end
        else:
            name = osp.basename(image_path)
        
        if name.endswith('.png'):
            NAME_PREFIX = '.png'
        elif name.endswith('.jpg'):
            NAME_PREFIX = '.jpg'

        cv2.imwrite(osp.join(target_dir, f'{name.replace(NAME_PREFIX, "")}_rgb.png'), img_crop_resized.astype(int))
        cv2.imwrite(osp.join(target_dir, f'{name.replace(NAME_PREFIX, "")}_mask.png'), (mask_crop_resized * 255).astype(int))
        with open(osp.join(target_dir, name.replace(NAME_PREFIX, "_box.txt")), 'w') as f:
            f.write(f'{idx} {(wc-crop_size):0.2f} {(hc-crop_size):0.2f} {(crop_size*2):0.2f} {(crop_size*2):0.2f} {W:0.2f} {H:0.2f} {0:0.2f}')
        
        if with_example:
            # create a mask-ed out image which is easy to check
            rgb_name = osp.join(target_dir, f'{name.replace(NAME_PREFIX, "")}_rgb.png')
            mask_name = osp.join(target_dir, f'{name.replace(NAME_PREFIX, "")}_mask.png')
            rgb_img = cv2.imread(rgb_name)
            mask_img = cv2.imread(mask_name)
            mask_img = mask_img == 255
            mask_img = mask_img.astype(int)
            mask_rgb = rgb_img * mask_img
            combine = np.concatenate((mask_rgb, rgb_img), axis=1)
            cv2.imwrite(rgb_name.replace('_rgb.png', '_combine.png'), combine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    args = parser.parse_args()

    segment_images(args.source_dir, args.target_dir, args.category)
