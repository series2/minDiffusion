"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import json

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


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),# 各chnannelのサイズは変化しない
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1), # 各channelのサイズは変化しない(ごく近傍のみでcnn)
        )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)

class FFNModel(nn.Module):
    def __init__(self,dim) -> None:
        super(FFNModel, self).__init__()
        # self.ffn = nn.Sequential(
        #     nn.Linear(dim,5),
        #     nn.Sigmoid(),
        #     nn.Linear(5,dim)
        # )
        self.ffn1=nn.Linear(dim,5)
        self.sig=nn.Sigmoid()
        self.ffn2=nn.Linear(5,dim)

    def forward(self, x, t) -> torch.Tensor:
        # print("first",x.shape)
        x=self.ffn1(x)
        # print(x.shape)
        x=self.sig(x)
        # print(x.shape)
        x=self.ffn2(x)
        # print(x.shape)
        return x
        # return self.ffn(x)

class DeepFFNModel(nn.Module):
    def __init__(self,dim) -> None:
        super(DeepFFNModel, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim,5),
            nn.Sigmoid(),
            nn.Linear(5,10),
            nn.Sigmoid(),
            nn.Linear(10,20),
            nn.Sigmoid(),
            nn.Linear(20,10),
            nn.Sigmoid(),
            nn.Linear(10,5),
            nn.Sigmoid(),
            nn.Linear(5,dim),
        )

    def forward(self, x, t) -> torch.Tensor:
        return self.ffn(x)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1) , deviceも合わせる
        
        x_dim_num=len(x.shape) # batch 次元 1 + データ次元 ex) 1(batch) + 1(channel dim) + 1(width) + 1 (height)
        shape=[len(x)]+[1]*(x_dim_num-1) # (len(x),1,1,....) 1はx_dim_num-1だけほしい
        x_t = (
            # self.sqrtab[_ts, None, None, None] * x # Noneは次元を増やす．sqrtab ... shqpe([T+1]) -> shqpe([1,1,1])( ex [[[0.001]]] ) .各データ毎に同じだけの作用を施すため，1pixcel単位で同じようにする．
            # + self.sqrtmab[_ts, None, None, None] * eps
            self.sqrtab[_ts].reshape(shape)* x
            + self.sqrtmab[_ts].reshape(shape) * eps
        )
        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        eps_pred=self.eps_model(x_t, _ts / self.n_T)

        eps_sum,eps_s_sum= eps.sum(),(eps*eps).sum()
        eps_pred_sum,eps_pred_s_sum= eps_pred.sum(),(eps_pred*eps_pred).sum()


        return self.criterion(eps, eps_pred) , (eps_sum,eps_s_sum) ,( eps_pred_sum,eps_pred_s_sum)

    def sample(self, n_sample: int, size, device,r=None,writer:SummaryWriter=None) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1): # i = self.N_T , self.n_T-1 , ... , 1 
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_0_pred = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
            x_i = (
                x_0_pred + self.sqrt_beta_t[i] * z
            )
            if r!=None and writer!=None and  i%(max(1,self.n_T//10)) ==0:
                writer.add_histogram(f'{r}epochrate_x0pred_distribution-by-step', x_0_pred, self.n_T-i+1)
                writer.add_histogram(f'{r}epochrate_xt_distribution-by-step', x_i, self.n_T-i+1)

        return x_i
class Psude1dimDataset(Dataset):
    """
    現状json + ハードコーディングだが，yamlにしておきたい．
    """
    def __init__(self,save_dir=None,size=1000,
                    mus=None,
                    sigmas=None,
                    ws=None,
                    ):
        sampled_data=None
        if save_dir==None:
            save_dir=f"data/psudedata/{datetime.datetime.now()}"
        if not os.path.isdir(save_dir):
            # 3つの峰を持つ混合ガウス分布
            self.mus    = mus
            self.sigmas = sigmas
            self.ws      = ws
            k=len(self.mus)
            assert self.mus.shape==self.sigmas.shape ==self.ws.shape
            # assert self.ws.sum()==1 厳密にはむずいので

            os.makedirs(save_dir)
            # 混合ガウス分布からのサンプリングを p(w)p(x|w)により行う．
            k_s=np.random.choice(np.arange(k),size=size,p=self.ws)
            sampled_data=np.random.normal(self.mus[k_s],self.sigmas[k_s],len(k_s))
            np.savetxt(f"{save_dir}/train.csv",sampled_data)
            with open(f"{save_dir}/overview.json","w") as f:
                json.dump({"w":self.ws.tolist(),"mus":self.mus.tolist(),"sigmas":self.sigmas.tolist(),"size":size,},f)
        else:
            with open(f"{save_dir}/overview.json","r") as f:
                overview=json.load(f)
            self.mus,self.sigmas,self.ws=overview["mus"],overview["sigmas"],overview["w"]
            sampled_data=np.loadtxt(f"{save_dir}/train.csv")
            # raise NotImplementedError()
        self.x=torch.tensor(sampled_data[:,None],dtype=torch.float) # shape(size,1)

    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],0 # 通常は x とそのラベルが返される
    
    def show_distribution(self,compare_tos=[],show_orig=True,cut=False):
        # cut_20 ... 片方に付き10%非表示にする 
        if show_orig:
            x=np.linspace(-10.0,10.0,1000)
            y=np.zeros_like(x)
            for w,mus,sigmas in zip(self.ws,self.mus,self.sigmas):
                y+=w * np.exp(-0.5 * ((x - mus)/sigmas) ** 2 ) / (sigmas*np.sqrt(2 * np.pi))
            plt.plot(x,y)
            plt.hist(self.x[:,0],density=True,bins=100,color="red")
        
        # if compare_to!=None:
        for c in compare_tos:
            sampled=np.loadtxt(f"{c}")
            if cut:
                sampled.sort()
                sampled=sampled[int(len(sampled)*4.9)//10:int(len(sampled)*5.1)//10]
            plt.hist(sampled,density=True,bins=100)
        
        plt.title(f"sample num : {len(self.x)}")
        plt.show()
    
class Psude1dimClassedDataset(Dataset):
    """
    現状json + ハードコーディングだが，yamlにしておきたい．
    """
    def __init__(self,save_dir=None,size=1000,
                    mus=None,
                    sigmas=None,
                    ws=None,
                    class_assign=None, # ws のどれがどのクラスに割り当てられるか. ws_iに対して class_assign[ws_i]で取得．
                    ):
        sampled_data=None
        if save_dir==None:
            save_dir=f"data/psudedata/{datetime.datetime.now()}"
        if not os.path.isdir(save_dir):
            # 3つの峰を持つ混合ガウス分布
            self.mus    = mus
            self.sigmas = sigmas
            self.ws      = ws
            self.class_assign=class_assign
            k=len(self.mus)
            assert self.mus.shape==self.sigmas.shape ==self.ws.shape == self.class_assign
            # assert self.ws.sum()==1 厳密にはむずいので

            os.makedirs(save_dir)
            # 混合ガウス分布からのサンプリングを p(w)p(x|w)により行う．
            k_s=np.random.choice(np.arange(k),size=size,p=self.ws)
            class_label=self.class_assign[k_s]
            sampled_data=np.random.normal(self.mus[k_s],self.sigmas[k_s],len(k_s))
            data=np.stack([sampled_data,class_label],-1) # TODO concat して (n,2) のnumpyに
            np.savetxt(f"{save_dir}/train.csv",data)
            with open(f"{save_dir}/overview.json","w") as f:
                json.dump({"w":self.ws.tolist(),"mus":self.mus.tolist(),"sigmas":self.sigmas.tolist(),"size":size,"class_assign":self.class_assign.tolist()},f)
        else:
            with open(f"{save_dir}/overview.json","r") as f:
                overview=json.load(f)
            self.mus,self.sigmas,self.ws,self.class_assign=overview["mus"],overview["sigmas"],overview["w"],overview["class_assign"]
            data=np.loadtxt(f"{save_dir}/train.csv")
            # raise NotImplementedError()
        self.x=torch.tensor(sampled_data[:,0,None],dtype=torch.float) # shape(size,1)
        self.y=torch.tensor(sampled_data[:,1],dtype=torch.float) # shape (size,)

    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx] 
    
    def show_distribution(self,compare_tos=[],show_orig=True,cut=False):
        # cut_20 ... 片方に付き10%非表示にする 
        if show_orig:
            x=np.linspace(-10.0,10.0,1000)
            p=np.zeros_like(x)
            for w,mus,sigmas in zip(self.ws,self.mus,self.sigmas):
                p+=w * np.exp(-0.5 * ((x - mus)/sigmas) ** 2 ) / (sigmas*np.sqrt(2 * np.pi))
            plt.plot(x,p)
            plt.hist(self.x[:,0],density=True,bins=100,color="red")
        
        # if compare_to!=None:
        for c in compare_tos:
            sampled=np.loadtxt(f"{c}")
            if cut:
                sampled.sort()
                sampled=sampled[int(len(sampled)*4.9)//10:int(len(sampled)*5.1)//10]
            plt.hist(sampled,density=True,bins=100)
        
        plt.title(f"sample num : {len(self.x)}")
        plt.show()
    


def train(n_epoch: int = 100, device="cuda:0" , writer:SummaryWriter=None,dataset_name=None,data_dir=None,eps_model=None,result_dir=None,n_T=1000) -> None:
    # dataset パラメタは一時的なもの．もうちょっとちゃんとしたい
    if eps_model==None:
        ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=n_T)
    elif eps_model=="FFNModel":
        ddpm = DDPM(eps_model=FFNModel(1), betas=(1e-4, 0.02), n_T=n_T)
    elif eps_model=="DeepFFNModel":
        ddpm = DDPM(eps_model=DeepFFNModel(1), betas=(1e-4, 0.02), n_T=n_T)
    else:
        raise Exception(f"model {eps_model} is not exist")
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    # if dataset==None:
    #     raise Exception("指 定が必要")
    if dataset_name==None:
        # download to ./data
        dataset = MNIST(
            "./data",
            train=True,
            download=True,
            transform=tf,
        )
    elif dataset_name=="Psude1dimDataset":
        dataset=Psude1dimDataset(save_dir=data_dir)
    else:
        raise Exception("存在しない")
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(dataloader)#,desc=f"epoch:{i}"
        loss_ema = None
        losses=[]
        (eps_sum,eps_s_sum) ,( eps_pred_sum,eps_pred_s_sum) = (0,0),(0,0)

        for x, _ in pbar:
        # for data in pbar:
            # print("check",data,data[0].shape)
            # exit()
            optim.zero_grad()
            x = x.to(device)
            loss,(eps_sum_,eps_s_sum_) ,( eps_pred_sum_,eps_pred_s_sum_) = ddpm(x)

            loss.backward()
            losses.append(loss.item())
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"epoch:{i:5} loss: {loss_ema:.4f}")
            optim.step()
        writer.add_scalar("Loss/train",torch.tensor(losses).mean(),i)
        data_shape=x[0].shape
        ddpm.eval()
        if result_dir!=None:
            with torch.no_grad():
                if dataset_name==None: # MNNIST
                    xh = ddpm.sample(16, data_shape, device)
                    grid = make_grid(xh, nrow=4)
                    save_image(grid, f"{result_dir}/ddpm_sample_{i}.png")
                elif dataset_name=="Psude1dimDataset":
                    if not os.path.isdir(f"{result_dir}"):
                        os.makedirs(f"{result_dir}")
                    if i%(n_epoch//10) ==0 or i==n_epoch-1:
                        xh = ddpm.sample(100000, data_shape, device,i/n_epoch,writer) # log reverse diffusion process
                    else:
                        xh = ddpm.sample(100000, data_shape, device)
                    xh=xh.cpu().detach().numpy()
                    if i%(n_epoch//10) ==0 or i==n_epoch-1:
                        writer.add_histogram('finalstep_distribution-by-epoch', xh, i)
                    np.savetxt(f"{result_dir}/psuede_sample_{i}.csv",xh)
                else:
                    raise Exception(f"Invalid. dataset_name ... {dataset_name}")
                

            # save model
            # if i%10==0 or i==n_epoch-1:
                # torch.save(ddpm.state_dict(), f"{result_dir}/ddpm_mnist_{i}.pth")

# def psude_dataset_test():
#     d=Psude1dimDataset(save_dir="data/psudedata/2023-07-17 21:23:27.281479")
#     d.show_distribution(compare_to="./contents/psude/psuede_sample_99.csv")
def do_many_train():
    # models=["FFNModel","DeepFFNModel"]
    # models=["FFNModel"]
    # sizes=[1000,10000,100000]
    # n_Ts=[2,10,100,500,1000,10000]
    # sizes=[100000]
    # n_Ts=[50]
    # for size in sizes:
    #     data_dir=f"data/psudedata/{size}size-1dim-1gmm"
    #     d=Psude1dimDataset(save_dir=data_dir,size=size,
    #     mus=np.array([5.0]),sigmas=np.array([1.0]),ws=np.array([1.0]))
    #     for n_T in n_Ts:
    #         for eps_model in models:
    #             result_dir=f"./runs/PsudeExperiments/{size}size-1dim-1gmm-{n_T}step-{eps_model}"
    #             if os.path.isdir(f"{result_dir}"):continue
    #             writer=SummaryWriter(result_dir)
    #             train(writer=writer,dataset_name="Psude1dimDataset",data_dir=data_dir,eps_model=eps_model,result_dir=result_dir,n_T=n_T)
    
    # models=["FFNModel","DeepFFNModel"]
    # models=["FFNModel"]
    # sizes=[1000,10000,100000]
    # n_Ts=[2,10,100,500,1000,10000] # 1 は　errorr となる．理由を特定すべき．
    # sizes=[100000]
    # n_Ts=[50]
    # for size in sizes:
    #     data_dir=f"data/psudedata/{size}size-1dim-3gmm"
    #     d=Psude1dimDataset(save_dir=data_dir,size=size,
    #     mus=np.array([ 0.5,-2.0, 6.0]),sigmas=np.array([ 1.0, 0.3, 3.0]),ws=np.array([  0.2 ,  0.5 ,  0.3 ]))
    #     for n_T in n_Ts:
    #         for eps_model in models:
    #             result_dir=f"./runs/PsudeExperiments/{size}size-1dim-3gmm-{n_T}step-{eps_model}"
    #             if os.path.isdir(f"{result_dir}"):continue
    #             writer=SummaryWriter(result_dir)
    #             train(writer=writer,dataset_name="Psude1dimDataset",data_dir=data_dir,eps_model=eps_model,result_dir=result_dir,n_T=n_T)
    
    models=["FFNModel"]
    sizes=[100000]
    n_Ts=[2,10,100,500,1000,10000]
    for size in sizes:
        data_dir=f"data/psudedata/{size}size-1dim-3gmm2"
        d=Psude1dimDataset(save_dir=data_dir,size=size,
        mus=np.array([ -50,10.0, 50]),sigmas=np.array([ 1.0, 1.0, 3.0]),ws=np.array([  0.3 ,  0.3 ,  0.4 ]))
        for n_T in n_Ts:
            for eps_model in models:
                result_dir=f"./runs/PsudeExperiments/{size}size-1dim-3gmm2-{n_T}step-{eps_model}"
                if os.path.isdir(f"{result_dir}"):continue
                writer=SummaryWriter(result_dir)
                train(writer=writer,dataset_name="Psude1dimDataset",data_dir=data_dir,eps_model=eps_model,result_dir=result_dir,n_T=n_T)

def show_distribution():
    # data_dir="data/psudedata/100000size-1dim-1gmm"
    # data_dir="data/psudedata/10000size-1dim-1gmm"
    # data_dir="data/psudedata/1000size-1dim-1gmm"
    # data_dir="data/psudedata/1000size-1dim-3gmm"
    # data_dir="data/psudedata/10000size-1dim-3gmm"
    data_dir="data/psudedata/100000size-1dim-3gmm"
    d=Psude1dimDataset(save_dir=data_dir)
    d.show_distribution()
    # result_dir="./runs/PsudeExperiments/1000size-1dim-3gmm-10000step-FFNModel"
    print(data_dir)
    # result_dir="./runs/PsudeExperiments/1000size-1dim-3gmm-10000step-FFNModel"
    # d.show_distribution(compare_tos=
    # [
    #     f"{result_dir}/psuede_sample_99.csv",
    # ],show_orig=False,cut=True)



if __name__ == "__main__":
    # writer=SummaryWriter("./runs/NNIST")
    # train(writer=writer)

    do_many_train()

    # show_distribution()

    # result_dir="./runs/PsudeExperiments/100000size-1000step-1dim-1gmm"
    # data_dir="data/psudedata/100000size-1dim-1gmm"
    # d=Psude1dimDataset(save_dir=data_dir)
    # d.show_distribution(compare_tos=[
        # f"{result_dir}/psuede_sample_0.csv",
        # f"{result_dir}/psuede_sample_2.csv",
        # f"{result_dir}/psuede_sample_5.csv",
        # f"{result_dir}/psuede_sample_10.csv",
        # f"{result_dir}/psuede_sample_20.csv",
        # f"{result_dir}/psuede_sample_30.csv",
        # f"{result_dir}/psuede_sample_99.csv",
        # ])

    # psude_dataset_test()
