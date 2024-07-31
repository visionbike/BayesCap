import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from super_resolution_task import *
from bayescap import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser("BayesCap for Super Resolution Task")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=3000)
    args = parser.parse_args()

    if args.device == "cuda":
        if "," in args.gpu_id:
            raise ValueError("Multiple GPUs training does not supported!")
        device = f"{args.device}:{args.gpu_id}"
    else:
        device = args.device

    print("# Load datasets...")
    dataset_train = SRDataset(data_root="../data/SRGAN_ImageNet", image_size=(84, 84), upscale_factor=4)
    dataset_val = SRDataset(data_root="../data/Set5/original", image_size=(256, 256), upscale_factor=4)
    dataset_test = SRDataset(data_root="../data/Set5/original", image_size=(256, 256), upscale_factor=4)

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=1, pin_memory=True, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=1, pin_memory=True, shuffle=False)

    print("Training BayesCap...")
    NetG = SRGenerator()
    NetG.load_state_dict(torch.load("../ckpt/srgan-ImageNet-bc347d67.pth", map_location=device))
    model_parameters = filter(lambda p: True, NetG.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of Parameters:", params)
    #
    NetC = SRBayesCap(in_channels=3, out_channels=3)
    #
    train_BayesCap(
        NetC,
        NetG,
        loader_train,
        loader_val,
        cri=TempCombLoss(alpha_eps=1e-5, beta_eps=1e-2),
        device=device,
        dtype=torch.cuda.FloatTensor,
        init_lr=1e-4,
        num_epochs=args.epoch,
        eval_every=2,
        ckpt_path="../ckpt/BayesCap_SRGAN",
        task="super-resolution",
        viz=False
    )

    print("Evaluating BayesCap...")
    NetG = SRGenerator()
    NetG.load_state_dict(torch.load("../ckpt/srgan-ImageNet-bc347d67.pth", map_location=device))
    NetG.to(args.device)
    NetG.eval()
    #
    NetC = SRBayesCap(in_channels=3, out_channels=3)
    NetC.load_state_dict(torch.load('../ckpt/BayesCap_SRGAN_best.pth', map_location=device))
    NetC.to(args.device)
    NetC.eval()
    #
    eval_BayesCap(
        NetC,
        NetG,
        loader_test,
        device=device,
        dtype=torch.cuda.FloatTensor,
        task="super-resolution",
        viz=False,
        test=True)
