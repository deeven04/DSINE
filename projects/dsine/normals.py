import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.append('../../')
import utils.utils as utils
import projects.dsine.config as config
from utils.projection import intrins_from_fov, intrins_from_txt

def calculate_normals(img_path, model, device):
    ext = os.path.splitext(img_path)[1]
    img = Image.open(img_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # pad input
    _, _, orig_H, orig_W = img.shape
    lrtb = utils.get_padding(orig_H, orig_W)
    img = F.pad(img, lrtb, mode="constant", value=0.0)
    
    # Normalize the image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)

    # get intrinsics
    intrins_path = img_path.replace(ext, '.txt')
    if os.path.exists(intrins_path):
        intrins = intrins_from_txt(intrins_path, device=device).unsqueeze(0)
    else:
        intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)
    intrins[:, 0, 2] += lrtb[0]
    intrins[:, 1, 2] += lrtb[2]

    # Calculate the normals
    pred_norm = model(img, intrins=intrins)[-1]
    pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

    # Convert to tensor and return
    pred_norm_tensor = pred_norm.detach().cpu()
    return pred_norm_tensor

def load_model():
    device = torch.device('cpu')
    args = config.get_args( test=True)
    assert os.path.exists(args.ckpt_path)

    if args.NNET_architecture == 'v00':
        from models.dsine.v00 import DSINE_v00 as DSINE
    elif args.NNET_architecture == 'v01':
        from models.dsine.v01 import DSINE_v01 as DSINE
    elif args.NNET_architecture == 'v02':
        from models.dsine.v02 import DSINE_v02 as DSINE
    elif args.NNET_architecture == 'v02_kappa':
        from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise Exception('invalid arch')

    model = DSINE(args).to(device)
    model = utils.load_checkpoint(args.ckpt_path, model)
    model.eval()
    return model, device

if __name__ == '__main__':
    model, device = load_model()

    img_paths = glob.glob('./samples/img/*.hdr') + glob.glob('./samples/img/*.jpg')
    img_paths.sort()

    with torch.no_grad():
        for img_path in img_paths:
            print(f"Processing: {img_path}")
            normals_tensor = calculate_normals(img_path, model, device)
            print(f"Generated normals tensor shape: {normals_tensor.shape}")
