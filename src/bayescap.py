import functools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from super_resolution_task import *
from metric import *
from viz import *

__all__ = [
    "train_BayesCap",
    "eval_BayesCap"
]


def eval_BayesCap(
        net_cap: nn.Module,
        net_gen: nn.Module,
        eval_loader: DataLoader,
        device: str = "cuda",
        dtype: torch.dtype = torch.cuda.FloatTensor,
        task: str | None = None,
        x_mask: torch.Tensor | None = None,
        viz: bool = False,
        test: bool = False,
) -> float:
    """

    :param net_cap:
    :param net_gen:
    :param eval_loader:
    :param device:
    :param gpu_id:
    :param dtype:
    :param task:
    :param x_mask:
    :param viz:
    :param test:
    :return:
    """
    net_cap.to(device)
    net_cap.eval()
    net_gen.to(device)
    net_gen.eval()
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
            x_a, x_b = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
            if task == "inpainting":
                if x_mask is None:
                    x_mask = random_mask(x_a.shape[0], (x_a.shape[2], x_a.shape[3]))
                    x_mask = x_mask.to(device).type(dtype)
                else:
                    x_mask = x_mask.to(device).type(dtype)
            # pass them through the network
            with torch.no_grad():
                if task == "inpainting":
                    _, y_b = net_gen(x_a, x_mask)
                elif task == "depth":
                    y_b = net_gen(x_a)[("disp", 0)]
                else:
                    y_b = net_gen(x_a)
                y_bc_mu, y_bc_alpha, y_bc_beta = net_cap(y_b)
            #
            map_alpha = (1.0 / (y_bc_alpha + 1e-5)).to("cpu").data
            map_beta = y_bc_beta.to("cpu").data
            map_error = torch.mean(torch.pow(torch.abs(y_b - x_b), 2), dim=1).to("cpu").data
            map_var = (map_alpha ** 2) * (torch.exp(torch.lgamma(3 / (map_beta + 1e-2))) / torch.exp(torch.lgamma(1 / (map_beta + 1e-2)))).to("cpu").data
            n_batch = y_bc_mu.shape[0]
            if task == "depth":
                x_b = y_b
            #
            for j in range(n_batch):
                num_imgs += 1
                if not test:
                    mean_ssim += compute_image_ssim(y_bc_mu[j], x_b[j])
                    mean_psnr += compute_image_psnr(y_bc_mu[j], x_b[j])
                    mean_mse += compute_image_mse(y_bc_mu[j], x_b[j])
                    mean_mae += compute_image_mae(y_bc_mu[j], x_b[j])
                else:
                    mean_ssim += compute_image_ssim(y_b[j], x_b[j])
                    mean_psnr += compute_image_psnr(y_b[j], x_b[j])
                    mean_mse += compute_image_mse(y_b[j], x_b[j])
                    mean_mae += compute_image_mae(y_b[j], x_b[j])
                #
                list_error.extend(map_error[j].tolist())
                list_var.extend(map_var[j].tolist())
                #
                if viz:
                    show_outputs_w_uncertainties(x_a[j], x_b[j], y_b[j],
                                                 y_bc_mu[j], map_alpha[j], map_beta[j],
                                                 map_error[j], map_var[j])
        #
        mean_ssim /= num_imgs
        mean_psnr /= num_imgs
        mean_mse /= num_imgs
        mean_mae /= num_imgs
        print(f"Avg. SSIM: {mean_ssim} | Avg. PSNR: {mean_psnr} | Avg. MSE: {mean_mse} | Avg. MAE: {mean_mae}")
        print(f"UCE: {compute_uce(list_error[::10], list_var[::10], num_bins=500)[1]}")
        print(f"C.Coeff: {compute_correlation_coefficient(list_error[::10], list_var[::10])}")
    return mean_ssim


def train_BayesCap(
        net_cap: nn.Module,
        net_gen: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        cri: nn.Module | functools.partial = TempCombLoss,
        device: str = "cuda",
        dtype: torch.dtype = torch.cuda.FloatTensor,
        init_lr: float = 1e-4,
        num_epochs: int = 100,
        eval_every: int = 1,
        ckpt_path: str = "../ckpt/BayesCap",
        t1: float = 1e0,
        t2: float = 5e-2,
        task: str | None = None,
        viz: bool = False
) -> None:
    """

    :param net_cap:
    :param net_gen:
    :param train_loader:
    :param eval_loader:
    :param cri:
    :param device:
    :param dtype:
    :param init_lr:
    :param num_epochs:
    :param eval_every:
    :param ckpt_path:
    :param t1:
    :param t2:
    :param task:
    :param viz:
    :return:
    """
    net_cap.to(device)
    net_cap.train()
    net_gen.to(device)
    net_gen.eval()
    optimizer = torch.optim.Adam(list(net_cap.parameters()), lr=init_lr)
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    #
    score = -1e8
    all_loss = []
    for epoch in range(num_epochs):
        loss_epoch = 0
        with tqdm(train_loader, unit="batch") as pbar:
            for (idx, batch) in enumerate(pbar):
                if idx > 2000:
                    break
                pbar.set_description(f"Epoch {epoch}")
                #
                x_a, x_b = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
                if task == "inpainting":
                    x_mask = random_mask(x_a.shape[0], (x_a.shape[2], x_a.shape[3]))
                    x_mask = x_mask.to(device).type(dtype)
                # pass them through the network
                with torch.no_grad():
                    if task == "inpainting":
                        _, y_b = net_gen(x_a, x_mask)
                    elif task == "depth":
                        y_b = net_gen(x_a)[("disp", 0)]
                    else:
                        y_b = net_gen(x_a)
                y_bc_mu, y_bc_alpha, y_bc_beta = net_cap(y_b)
                #
                optimizer.zero_grad()
                loss_iter = cri(y_bc_mu, y_bc_alpha, y_bc_beta, y_b, x_b, t1=t1, t2=t2)
                loss_iter.backward()
                optimizer.step()
                loss_epoch += loss_iter.item()
                pbar.set_postfix(loss=loss_iter.item())
            loss_epoch /= len(train_loader)
            all_loss.append(loss_epoch)
            print(f"Avg. loss: {loss_epoch}")
        # evaluate and save the models
        torch.save(net_cap.state_dict(), ckpt_path + "_last.pth")
        if (epoch % eval_every) == 0:
            score_curr = eval_BayesCap(
                net_cap,
                net_gen,
                eval_loader,
                device=device,
                dtype=dtype,
                task=task,
                viz=viz,
            )
            print(f"current score: {score_curr} | Last best score: {score}")
            if score_curr >= score:
                score = score_curr
                torch.save(net_cap.state_dict(), ckpt_path + "_best.pth")
    optim_scheduler.step()
