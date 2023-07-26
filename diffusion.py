"""
Based onhttps://arxiv.org/abs/2006.11239
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import json

from dataset import dataset_map,DatasetWithLogger
from model import model_map,ModelBase

import argparse

def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1 # つまり新たに加えるノイズの分散は線形に増加する
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }



class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        _ts = torch.randint(1, self.n_T, (x_0.shape[0],)).to(
            x_0.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x_0)  # eps ~ N(0, 1) , deviceも合わせる

        x_dim_num=len(x_0.shape) # batch 次元 1 + データ次元 ex) 1(batch) + 1(channel dim) + 1(width) + 1 (height)
        shape=[len(x_0)]+[1]*(x_dim_num-1) # (len(x),1,1,....) 1はx_dim_num-1だけほしい
        x_t = (
            self.sqrtab[_ts].reshape(shape)* x_0
            + self.sqrtmab[_ts].reshape(shape) * eps
        )
        eps_pred=self.eps_model(x_t, _ts / self.n_T)
        return self.criterion(eps, eps_pred)

    def sample(self, n_sample: int, size, device,
        is_log=False,epoch_rate=None,writer:SummaryWriter=None,dataset:DatasetWithLogger=None
        ) -> torch.Tensor:
        # epoch_rate ... 0, 0.1, 0.2, ..., 0.9, 1.0 のいずれか
        if is_log:
            assert epoch_rate!=None and writer!=None and dataset!=None

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        for step in range(self.n_T, 0, -1): # step = self.n_T , self.n_T-1 , ... , 1 
            eps_pred = self.eps_model(x_i, step / self.n_T)
            x_0_pred = self.oneover_sqrta[step] * (x_i - eps_pred * self.mab_over_sqrtmab[step])
            z = torch.randn(n_sample, *size).to(device) if step > 1 else 0
            x_im = (
                x_0_pred + self.sqrt_beta_t[step] * z
            )

            if step==self.n_T and is_log:# phase = 0
                dataset.sample_logger_for_first_step(writer,epoch_rate,eps_pred,x_0_pred,x_im,x_i)
            phase=self.n_T-step+1 # 1,2,...,self.n_T
            if phase%(max(1,self.n_T//10)) ==0 and is_log: # phase = 10,20,...,n_T
                dataset.sample_logger_by_step(writer,epoch_rate,eps_pred,x_0_pred,x_im,phase)
            x_i=x_im
        x_0=x_i
        return x_0

def train(dataset_name:str,data_dir:str,eps_model_name:str,result_dir:str,
            n_epoch:int=100,n_T=1000,batch_size=128,lr=2e-4,sample_num=16,
            device=torch.cuda.device("cpu") , use_tensorboard=True) -> None:
    if os.path.isdir(result_dir): 
        raise Exception(f"すでに結果ディレクトリ `{result_dir}`が存在します．学習を再開したい場合に備え，本機能は変更される可能性があります．") # TODO
    writer=SummaryWriter(result_dir) if use_tensorboard else None

    try:
        eps_model:ModelBase=model_map[eps_model_name](1)
    except Exception as e:
        raise Exception(f"Modelの読みこみに失敗しました．\n{e}")
    try:
        dataset:DatasetWithLogger=dataset_map[dataset_name](save_dir=data_dir)
    except Exception as e:
        raise Exception(f"datasetの読みこみに失敗しました．\n{e}")
    
    
    ddpm = DDPM(eps_model=eps_model, betas=(1e-4, 0.02), n_T=n_T)
    ddpm.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    if not os.path.isdir(f"{result_dir}"):
        os.makedirs(f"{result_dir}")

    for epoch in range(n_epoch):
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        losses=[]
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss= ddpm(x)

            loss.backward()
            losses.append(loss.item())
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"epoch:{epoch:5} loss: {loss_ema:.4f}")
            optim.step()
        writer.add_scalar("Loss/train",torch.tensor(losses).mean(),epoch)


        # TODO Diffusionは推論のステップが大きく，逐次的という意味で並列化が難しい
        # 今回の場合，特に低次元に置いて，GPUが余っているのに推論がボトルネックになっている．
        # そのため以下のように実装する
        # モデルを保存して，プロセスを分離して，余っったGPUメモリを使ってサンプリングを行う．
        # サンプリングが遅いのでほっとくと溢れてしまうので，プロセスプールの管理をするようにする．

        data_shape=x[0].shape
        ddpm.eval()
        if result_dir!=None:
            with torch.no_grad():
                is_log_epoch= (epoch%(n_epoch//10) ==0 or epoch==n_epoch-1) and use_tensorboard
                epoch_rate=(((epoch+1)*10)//n_epoch)/10

                x0_pred = ddpm.sample(sample_num, data_shape, device,is_log_epoch,epoch_rate,writer,dataset)
                # TODO sampleが大きくてGPUに乗らないとき，Batch処理を実装する

                x0_pred=x0_pred.cpu().detach().numpy()
                if is_log_epoch:
                    dataset.sample_logger_for_epoch(writer,x0_pred,epoch)
                if epoch==n_epoch-1 and use_tensorboard:
                    dataset.sample_logger_for_last_epoch(writer,x0_pred)
                np.savetxt(f"{result_dir}/psuede_sample_{epoch}.csv",x0_pred)

def get_parser():
    parser = argparse.ArgumentParser()
    dbs='\n\t'.join(dataset_map.keys())
    parser.add_argument("--dataset_name",help=f"You can use folloing strings.check dataset.py. {dbs}" )
    parser.add_argument("--data_dir")
    models='\n\t'.join(model_map.keys())
    parser.add_argument("--eps_model_name",help=f"You can use folloing strings.check model.py. {models}" )
    parser.add_argument("--result_dir")
    parser.add_argument("--n_epoch",default=100)
    parser.add_argument("--n_T",default=1000)
    parser.add_argument("--batch_size",default=128)
    parser.add_argument("--lr",default=2e-4)
    parser.add_argument("--sample_num",default=16)
    parser.add_argument("--device",default=0,help="put device id you want to use")
    parser.add_argument("--use_tensorboard", action="store_true")
    return parser

if __name__ == "__main__":
    model="FFNModel"
    size=100000
    n_T=100
    batch_size=8192
    lr=2e-4 * 80
    data_dir,dataset_name=f"data/psudedata/{size}size-1dim-1gmm-origin05","Psude1dimDataset"
    result_dir=f"./runs/debug"

    train(n_epoch=100,use_tensorboard=True,dataset_name=dataset_name,data_dir=data_dir,eps_model_name=model,result_dir=result_dir,n_T=n_T,batch_size=batch_size,lr=lr)

