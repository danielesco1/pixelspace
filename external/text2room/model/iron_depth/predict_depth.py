import os
import numpy as np
from argparse import Namespace
import torch
from PIL import Image
from torchvision import transforms

import text2room.model.iron_depth.utils.utils as utils
from text2room.model.iron_depth.models_normal.NNET import NNET
from text2room.model.iron_depth.models.IronDepth import IronDepth

from diffusers import MarigoldDepthPipeline
import gdown

def download_checkpoints():
    os.makedirs('checkpoints', exist_ok=True)

    # Google Drive File IDs
    normal_id = '1joC2lim4Zf9BywAfwQAdB9Funec4EAtm'
    iron_id = '1janrxs5bcu1Djx_clBBvb9r_kv7jN1EJ'

    # Output filenames
    scannet_filename = 'normal_scannet.pt'
    irondepth_filename = 'irondepth_scannet.pt'

    scannet_filepath = os.path.join('checkpoints', scannet_filename)
    irondepth_filepath = os.path.join('checkpoints', irondepth_filename)

    # Check if the files already exist before downloading
    if not os.path.exists(scannet_filepath):
        gdown.download(f'https://drive.google.com/uc?id={normal_id}', scannet_filepath, quiet=False)
    else:
        print(f"{scannet_filename} already exists.")

    if not os.path.exists(irondepth_filepath):
        gdown.download(f'https://drive.google.com/uc?id={iron_id}', irondepth_filepath, quiet=False)
    else:
        print(f"{irondepth_filename} already exists.")

def load_iron_depth_model(type="scannet", iters=20, checkpoints_path="checkpoints", device="cuda:0"):
    # Ensure the checkpoints are available
    download_checkpoints()
    
    args = Namespace(**{
        "train_iter": 3,
        "test_iter": iters
    })
    
    if type == 'scannet':
        args.NNET_architecture = 'BN'
        args.NNET_ckpt = os.path.join(checkpoints_path, 'normal_scannet.pt')
        args.IronDepth_ckpt = os.path.join(checkpoints_path, 'irondepth_scannet.pt')
    elif type == 'nyuv2':
        args.NNET_architecture = 'GN'
        args.NNET_ckpt = os.path.join(checkpoints_path, 'normal_nyuv2.pt')
        args.IronDepth_ckpt = os.path.join(checkpoints_path, 'irondepth_nyuv2.pt')
    else:
        raise NotImplementedError("type=", type)

    device = torch.device(device)

    # define N_NET (surface normal estimation network)
    n_net = NNET(args).to(device)
    #print('loading N-Net weights from %s' % args.NNET_ckpt)
    n_net = utils.load_checkpoint(args.NNET_ckpt, n_net)
    n_net.eval()

    # define IronDepth
    model = IronDepth(args).to(device)
    #print('loading IronDepth weights from %s' % args.IronDepth_ckpt)
    model = utils.load_checkpoint(args.IronDepth_ckpt, model)
    model.eval()
    
    """
    ##using Marigold
    depth_model_path = "prs-eth/marigold-depth-lcm-v1-0"
    model = MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
            ).to(device)
    model.enable_model_cpu_offload()
    n_net = None
    """
    return n_net, model


def prepare_input(image: Image, K=None, device="cuda:0"):
    W, H = image.size
    device = torch.device(device)

    if K is None:
        K = np.eye(3).astype(np.float32)
        K[0, 0] = 500.0
        K[1, 1] = 500.0
        K[0, 2] = W // 2
        K[1, 2] = H // 2

    if isinstance(K, torch.Tensor):
        K = K.squeeze().detach().cpu().numpy()

    def get_pos(intrins, W, H):
        pos = np.ones((H, W, 2))
        x_range = np.concatenate([np.arange(W).reshape(1, W)] * H, axis=0)
        y_range = np.concatenate([np.arange(H).reshape(H, 1)] * W, axis=1)
        pos[:, :, 0] = x_range + 0.5
        pos[:, :, 1] = y_range + 0.5
        pos[:, :, 0] = np.arctan((pos[:, :, 0] - intrins[0, 2]) / intrins[0, 0])
        pos[:, :, 1] = np.arctan((pos[:, :, 1] - intrins[1, 2]) / intrins[1, 1])
        pos = torch.from_numpy(pos.astype(np.float32)).permute(2, 0, 1).to(device).unsqueeze(0)  # (1, 2, H, W)
        return pos

    # pos is an array of rays with unit depth
    pos = get_pos(K, W, H)

    img = np.array(image).astype(np.float32) / 255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img).permute(2, 0, 1)).to(device).unsqueeze(0)  # (1, 3, H, W)

    sample = {
        'img': img,
        'pos': pos
    }

    return sample


@torch.no_grad()
def predict_iron_depth(image, K=None, device="cuda:0", model=None, n_net=None, input_depth=None, input_mask=None, fix_input_depth=False):
    """
    Predict depth with IronDepth method.

    :param image: image from which to predict depth (PIL Image in range 0..255)
    :param K: intrinsics
    :param device: torch device
    :param model: iron_depth model
    :param n_net: iron_depth_n_net model
    :param input_depth: guidance depth that should be inpainted
    :param input_mask: where the guidance depth should be not trusted/inpainted. 1: these values should be inpainted. 0: these values should be trusted/fixed.
    :param fix_input_depth: if the guidance depth should remain unchanged or only initialized in the first iteration of iron_depth
    :return: predicted depth, predicted normal uncertainty
    """
    
    if model is None or n_net is None:
        model, n_net = load_iron_depth_model()

    data_dict = prepare_input(image, K, device)
    img = data_dict['img']
    pos = data_dict['pos']

    # surface normal prediction
    norm_out = n_net(img)
    pred_norm = norm_out[:, :3, :, :]
    pred_kappa = norm_out[:, 3:, :, :]
    alpha, alpha_deg = utils.kappa_to_alpha_torch(pred_kappa.permute(0, 2, 3, 1)[0, :, :, 0])

    input_dict = {
        'img': img,
        'pred_norm': pred_norm,
        'pred_kappa': pred_kappa,
        'pos': pos
    }

    if input_depth is not None:
        input_dict['input_depth'] = input_depth.unsqueeze(0).unsqueeze(0)
        input_dict['input_mask'] = input_mask.unsqueeze(0).unsqueeze(0)
        input_dict['fix_input_depth'] = fix_input_depth

    # IronDepth forward pass
    pred_list = model(input_dict, 'test')
    pred_depth = pred_list[-1].squeeze()
    
    """
    depth = model(image).prediction[0]
    print(depth.shape)
    pred_depth = torch.tensor(depth).to(device).squeeze()
    alpha = None
    """
    return pred_depth, alpha
