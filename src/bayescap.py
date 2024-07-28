import functools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import *
from metric import *
from utils import *
from viz import *

__all__ = [
    "train_BayesCap",
    "eval_BayesCap"
]


def eval_BayesCap(
        NetC: nn.Module,
        NetG: nn.Module,
        eval_loader: DataLoader,
        device: str = "cuda",
        dtype: torch.dtype = torch.cuda.FloatTensor,
        task: str | None = None,
        xMask: torch.Tensor | None = None,
) -> float:
    """

    :param NetC:
    :param NetG:
    :param eval_loader:
    :param device:
    :param dtype:
    :param task:
    :param xMask:
    :return:
    """
    NetC.to(device)
    NetC.eval()
    NetG.to(device)
    NetG.eval()
    #
    mean_ssim = 0.
    mean_psnr = 0.
    mean_mse = 0.
    mean_mae = 0.
    num_imgs = 0.
    list_error = []
    list_var = []
    with tqdm(eval_loader, unit="batch") as tepoch:
        for (idx, batch) in enumerate(tepoch):
            tepoch.set_description("Validating ...")
            #
            xLR, xHR = batch[0].to(device), batch[1].to(device)
            xLR, xHR = xLR.type(dtype), xHR.type(dtype)
            if task == "inpainting":
                if xMask is None:
                    xMask = random_mask(xLR.shape[0], (xLR.shape[2], xLR.shape[3]))
                    xMask = xMask.to(device).type(dtype)
                else:
                    xMask = xMask.to(device).type(dtype)
            # pass them through the network
            with torch.no_grad():
                if task == "inpainting":
                    _, xSR = NetG(xLR, xMask)
                elif task == "depth":
                    xSR = NetG(xLR)[("disp", 0)]
                else:
                    xSR = NetG(xLR)
                xSRC_mu, xSRC_alpha, xSRC_beta = NetC(xSR)
            a_map = (1 / (xSRC_alpha + 1e-5)).to("cpu").data
            b_map = xSRC_beta.to("cpu").data
            xSRvar = (a_map ** 2) * (torch.exp(torch.lgamma(3 / (b_map + 1e-2))) / torch.exp(torch.lgamma(1 / (b_map + 1e-2))))
            n_batch = xSRC_mu.shape[0]
            if task == "depth":
                xHR = xSR
            for j in range(n_batch):
                num_imgs += 1
                mean_ssim += compute_image_ssim(xSRC_mu[j], xHR[j])
                mean_psnr += compute_image_psnr(xSRC_mu[j], xHR[j])
                mean_mse += compute_image_mse(xSRC_mu[j], xHR[j])
                mean_mae += compute_image_mae(xSRC_mu[j], xHR[j])

                show_SR_w_uncertainties(xLR[j], xHR[j], xSR[j], xSRvar[j])

                error_map = torch.mean(torch.pow(torch.abs(xSR[j] - xHR[j]), 2), dim=0).to('cpu').data.reshape(-1)
                var_map = xSRvar[j].to('cpu').data.reshape(-1)
                list_error.extend(list(error_map.numpy()))
                list_var.extend(list(var_map.numpy()))
        #
        mean_ssim /= num_imgs
        mean_psnr /= num_imgs
        mean_mse /= num_imgs
        mean_mae /= num_imgs
        print(f"Avg. SSIM: {mean_ssim} | Avg. PSNR: {mean_psnr} | Avg. MSE: {mean_mse} | Avg. MAE: {mean_mae}")
    # print(len(list_error), len(list_var))
    print(f"UCE: {compute_uce(list_error[::10], list_var[::10], num_bins=500)[1]}")
    print(f"C.Coeff: {compute_correlation_coefficient(list_error[::10], list_var[::10])}")
    return mean_ssim


def train_BayesCap(
        NetC: nn.Module,
        NetG: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        Cri: nn.Module | functools.partial = TempCombLoss,
        device: str = "cuda",
        dtype: torch.dtype = torch.cuda.FloatTensor,
        init_lr: float = 1e-4,
        num_epochs: int = 100,
        eval_every: int = 1,
        ckpt_path: str = "../ckpt/BayesCap",
        t1: float = 1e0,
        t2: float = 5e-2,
        task: str | None = None,
) -> None:
    """

    :param NetC:
    :param NetG:
    :param train_loader:
    :param eval_loader:
    :param Cri:
    :param device:
    :param dtype:
    :param init_lr:
    :param num_epochs:
    :param eval_every:
    :param ckpt_path:
    :param t1:
    :param t2:
    :param task:
    :return:
    """
    NetC.to(device)
    NetC.train()
    NetG.to(device)
    NetG.eval()
    optimizer = torch.optim.Adam(list(NetC.parameters()), lr=init_lr)
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    #
    score = -1e8
    all_loss = []
    for epoch in range(num_epochs):
        loss_epoch = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for (idx, batch) in enumerate(tepoch):
                if idx > 2000:
                    break
                tepoch.set_description(f"Epoch {epoch}")
                #
                xLR, xHR = batch[0].to(device), batch[1].to(device)
                xLR, xHR = xLR.type(dtype), xHR.type(dtype)
                if task == "inpainting":
                    xMask = random_mask(xLR.shape[0], (xLR.shape[2], xLR.shape[3]))
                    xMask = xMask.to(device).type(dtype)
                # pass them through the network
                with torch.no_grad():
                    if task == "inpainting":
                        _, xSR1 = NetG(xLR, xMask)
                    elif task == "depth":
                        xSR1 = NetG(xLR)[("disp", 0)]
                    else:
                        xSR1 = NetG(xLR)
                # with torch.autograd.set_detect_anomaly(True):
                xSR = xSR1.clone()
                xSRC_mu, xSRC_alpha, xSRC_beta = NetC(xSR)
                # print(xSRC_alpha)
                optimizer.zero_grad()
                loss = Cri(xSRC_mu, xSRC_alpha, xSRC_beta, xSR, xHR, t1=t1, t2=t2)
                # print(loss)
                loss.backward()
                optimizer.step()
                #
                loss_epoch += loss.item()
                tepoch.set_postfix(loss=loss.item())
            loss_epoch /= len(train_loader)
            all_loss.append(loss_epoch)
            print(f"Avg. loss: {loss_epoch}")
        # evaluate and save the models
        torch.save(NetC.state_dict(), ckpt_path + "_last.pth")
        if (epoch % eval_every) == 0:
            curr_score = eval_BayesCap(
                NetC,
                NetG,
                eval_loader,
                device=device,
                dtype=dtype,
                task=task,
            )
            print(f"current score: {curr_score} | Last best score: {score}")
            if curr_score >= score:
                score = curr_score
                torch.save(NetC.state_dict(), ckpt_path + "_best.pth")
    optim_scheduler.step()
