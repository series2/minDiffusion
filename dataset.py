import torch
from torch.utils.data import Dataset
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import os
import json
import numpy as np


"""
使い方
DatasetWithLoggerを継承してinit_class()で定義することによりdiffusion.pyで使用できます．
"""

def init_class():
    dataset_map={
        "Psude1dimDataset":Psude1dimDataset
    }
    return dataset_map

class DatasetWithLogger(Dataset):
    def __init__(self,save_dir:str):
        raise NotImplementedError
    def init_data(self,**args):
        raise NotImplementedError
    def sample_logger_for_first_step(self,writer:SummaryWriter,epoch_rate:float,
        eps_pred:Tensor,x_0_pred:Tensor,x_i_pred:Tensor,x_T:Tensor):
        raise NotImplementedError
    def sample_logger_by_step(self,writer:SummaryWriter,epoch_rate:float,
        eps_pred:Tensor,x_0_pred:Tensor,x_i_pred:Tensor,phase:int):
        raise NotImplementedError
    def sample_logger_for_epoch(self,writer:SummaryWriter,x_0_pred:Tensor,epoch:int):
        raise NotImplementedError
    def sample_logger_for_last_epoch(self,writer:SummaryWriter,x_0_pred:Tensor):
        raise NotImplementedError

class Psude1dimDataset(DatasetWithLogger):
    """
    現状json + ハードコーディングだが，yamlにしておきたい．
    """
    def __init__(self,save_dir=None,**args):
        if not os.path.isdir(save_dir): 
            self.init_data(save_dir,**args)
        else:
            with open(f"{save_dir}/overview.json","r") as f:
                overview=json.load(f)
            self.mus,self.sigmas,self.ws=overview["mus"],overview["sigmas"],overview["w"]
            sampled_data=np.loadtxt(f"{save_dir}/train.csv")
            self.x=torch.tensor(sampled_data[:,None],dtype=torch.float) # shape(size,1)

    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],0 # 通常は x とそのラベルが返される
    
    def init_data(self,save_dir,size, mus, sigmas, ws):
        if os.path.isdir(save_dir):raise Exception("save_dir is already exists.")
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
        self.x=torch.tensor(sampled_data[:,None],dtype=torch.float) # shape(size,1)

    def sample_logger_for_first_step(self,writer:SummaryWriter,epoch_rate:float,
        eps_pred:Tensor,x_0_pred:Tensor,x_i_pred:Tensor,x_T:Tensor):
        phase=0
        prefix=f"{epoch_rate:.1f}epochrate"
        writer.add_histogram(f'{prefix}_epspred-distribution_by-step', eps_pred, phase)
        writer.add_histogram(f'{prefix}_x0pred_distribution_by-step', x_0_pred, phase)
        writer.add_histogram(f'{prefix}_xt-distribution_by-step', x_T,phase) # ここだけ違う
        writer.add_scalar(f"{prefix}_epspred-var_by-step",eps_pred.std(), phase)
        writer.add_scalar(f"{prefix}_x0pred-var_by-step",x_0_pred.std(), phase)

    def sample_logger_by_step(self,writer:SummaryWriter,epoch_rate:float,eps_pred:Tensor,x_0_pred:Tensor,x_i_pred:Tensor,phase:int):
        prefix=f"{epoch_rate:.1f}epochrate"
        
        writer.add_histogram(f'{prefix}_epspred-distribution_by-step', eps_pred, phase)
        writer.add_histogram(f'{prefix}_x0pred_distribution_by-step', x_0_pred, phase)
        writer.add_histogram(f'{prefix}_xt-distribution_by-step', x_i_pred,phase)
        writer.add_scalar(f"{prefix}_epspred-var_by-step",eps_pred.std(), phase)
        writer.add_scalar(f"{prefix}_x0pred-var_by-step",x_0_pred.std(), phase)

    def sample_logger_for_epoch(self,writer:SummaryWriter,x_0_pred:Tensor,epoch:int):
        writer.add_histogram('finalstep_distribution-by-epoch', x_0_pred, epoch)
    
    def sample_logger_for_last_epoch(self,writer:SummaryWriter,x_0_pred:Tensor):
        writer.add_embedding(x_0_pred)
    
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
    

    
# class Psude1dimClassedDataset(Dataset):
#     """
#     現状json + ハードコーディングだが，yamlにしておきたい．
#     """
#     def __init__(self,save_dir=None,size=1000,
#                     mus=None,
#                     sigmas=None,
#                     ws=None,
#                     class_assign=None, # ws のどれがどのクラスに割り当てられるか. ws_iに対して class_assign[ws_i]で取得．
#                     ):
#         sampled_data=None
#         if save_dir==None:
#             save_dir=f"data/psudedata/{datetime.datetime.now()}"
#         if not os.path.isdir(save_dir): 
#             # 3つの峰を持つ混合ガウス分布
#             self.mus    = mus
#             self.sigmas = sigmas
#             self.ws      = ws
#             self.class_assign=class_assign
#             k=len(self.mus)
#             assert self.mus.shape==self.sigmas.shape ==self.ws.shape == self.class_assign
#             # assert self.ws.sum()==1 厳密にはむずいので

#             os.makedirs(save_dir)
#             # 混合ガウス分布からのサンプリングを p(w)p(x|w)により行う．
#             k_s=np.random.choice(np.arange(k),size=size,p=self.ws)
#             class_label=self.class_assign[k_s]
#             sampled_data=np.random.normal(self.mus[k_s],self.sigmas[k_s],len(k_s))
#             data=np.stack([sampled_data,class_label],-1) # TODO concat して (n,2) のnumpyに
#             np.savetxt(f"{save_dir}/train.csv",data)
#             with open(f"{save_dir}/overview.json","w") as f:
#                 json.dump({"w":self.ws.tolist(),"mus":self.mus.tolist(),"sigmas":self.sigmas.tolist(),"size":size,"class_assign":self.class_assign.tolist()},f)
#         else:
#             with open(f"{save_dir}/overview.json","r") as f:
#                 overview=json.load(f)
#             self.mus,self.sigmas,self.ws,self.class_assign=overview["mus"],overview["sigmas"],overview["w"],overview["class_assign"]
#             data=np.loadtxt(f"{save_dir}/train.csv")
#             # raise NotImplementedError()
#         self.x=torch.tensor(sampled_data[:,0,None],dtype=torch.float) # shape(size,1)
#         self.y=torch.tensor(sampled_data[:,1],dtype=torch.float) # shape (size,)

#     def __len__(self):
#         return len(self.x)
#     def __getitem__(self,idx):
#         return self.x[idx],self.y[idx] 
    
#     def show_distribution(self,compare_tos=[],show_orig=True,cut=False):
#         # cut_20 ... 片方に付き10%非表示にする 
#         if show_orig:
#             x=np.linspace(-10.0,10.0,1000)
#             p=np.zeros_like(x)
#             for w,mus,sigmas in zip(self.ws,self.mus,self.sigmas):
#                 p+=w * np.exp(-0.5 * ((x - mus)/sigmas) ** 2 ) / (sigmas*np.sqrt(2 * np.pi))
#             plt.plot(x,p)
#             plt.hist(self.x[:,0],density=True,bins=100,color="red")
        
#         # if compare_to!=None:
#         for c in compare_tos:
#             sampled=np.loadtxt(f"{c}")
#             if cut:
#                 sampled.sort()
#                 sampled=sampled[int(len(sampled)*4.9)//10:int(len(sampled)*5.1)//10]
#             plt.hist(sampled,density=True,bins=100)
        
#         plt.title(f"sample num : {len(self.x)}")
#         plt.show()
    
dataset_map=init_class()


if __name__=="__main__":
    pass
    # sizes=[1000,10000,100000]
    # for size in sizes:
    #     data_dir=f"data/psudedata/{size}size-1dim-1gmm"
    #     d=Psude1dimDataset(data_dir,size=size,
    #         mus=np.array([5.0]),sigmas=np.array([1.0]),ws=np.array([1.0])
    #     )
    # for size in sizes:
    #     data_dir=f"data/psudedata/{size}size-1dim-3gmm"
    #     d=Psude1dimDataset(save_dir=data_dir,size=size,
    #         mus=np.array([ 0.5,-2.0, 6.0]),
    #         sigmas=np.array([ 1.0, 0.3, 3.0]),
    #         ws=np.array([  0.2 ,  0.5 ,  0.3 ])
    #     )

    