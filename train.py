#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import time
import datetime
import pytorch_lightning as pl
import os
import numpy as np
import pickle
from preprocess.extra import EXTRA

def compute_correlations(labels, preds, return_detail = False):
    device = labels.device
    
    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    corr = np.nan_to_num([np.corrcoef(labels[:,i], preds[:,i])[0,1] for i in range(labels.shape[1])], nan = -1).tolist()
    if return_detail:
        return corr
    corr = np.mean(corr)
    return torch.FloatTensor([corr]).to(device)

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / (r_den + 1e-8)
    r_val = torch.nan_to_num(r_val,nan=-1)
    return r_val

def load_pickle(name):
    with open(name,'rb') as f:
        file = pickle.load(f)
    return file

def decompose(x):
    if x.startswith("__ambiguous"):
        x = x.replace("__ambiguous[","").replace("]","").split("+")
    return x

def is_symbol_in(name,gene2name):
    if isinstance(name, str):
        return "symbol" in gene2name[name]
    else:
        flag = True
        if len(name) > 3:
            return False
        for i in name:
            flag = flag and "symbol" in gene2name[i]
        return flag
    
class Description:
    def __init__(self, root, name_feature_path):
        super().__init__()
        
        self.gene = np.array(load_pickle(f"{root}/gene.pkl"))
        self.gene2name = load_pickle(f"{root}/gene2name.pkl")
        self.gene2name.update(EXTRA)
        self.mean = np.load(f"{root}/mean_expression.npy")
        self.name_feature = name_feature_path        

        self.generate_mask()
        
    def generate_mask(self):
        
        keep = set(list(zip(*sorted(zip(self.mean, range(self.mean.shape[0])))[::-1][:250]))[1])
        self.filter_name = [j for i,j in  enumerate(self.gene) if i in keep]
        self.test_mask = np.array([i in keep for i in range(len(self.gene))],dtype=bool)
        self.train_mask = np.logical_not(self.test_mask)
        
        for i,j in enumerate(self.gene):
            j = decompose(j)
            if j not in self.gene2name or "symbol" not in self.gene2name[j] or not os.path.exists(os.path.join(self.name_feature,f"{j}.pkl")):
                self.train_mask[i] = False
                self.test_mask[i] = False 
                
        assert self.test_mask.sum()==250, str(self.test_mask.sum())

        self.train_gene_indices = np.array(range(len(self.gene)))[self.train_mask]
        self.test_gene_indices = np.array(range(len(self.gene)))[self.test_mask]
        print(f"Train: {self.train_mask.sum()}; Test: {self.test_mask.sum()}")
        print(f"Train: {len(self.train_gene_indices)}; Test: {len(self.test_gene_indices)}")
   
    def load_test(self):
        if hasattr(self,"test_emb"):
            return self.test_emb, self.test_mask, self.size 
        test_emb = []
        size = []
        for idx in self.test_gene_indices:
            i = self.gene[idx]
            i = decompose(i)  if i.startswith("__ambiguous") else [i]
            current_emb = []
            for j in i:
                symbol = self.gene2name[j]['symbol']
                current_emb.append(load_pickle(os.path.join(self.name_feature,f"{symbol}.pkl"))[1][0].cpu())
                
            size.append([i.size(1) for i in current_emb])
            test_emb.append(torch.cat(current_emb,1))
            
        self.test_emb = test_emb
        self.test_mask = torch.from_numpy(self.test_mask)
        self.size = size
        return self.test_emb, self.test_mask, self.size 
    
    def sample_train(self):

        sample_gene = []
        sample_emb = []
        size = []
        sample_index = sorted(np.random.choice(self.train_gene_indices,64,replace=False))
        for idx in sample_index:
            sample_gene.append(idx)
            i = self.gene[idx]
            i = decompose(i)  if i.startswith("__ambiguous") else [i]
            current_emb = []
            for j in i:
                symbol = self.gene2name[j]['symbol']
                current_emb.append(load_pickle(os.path.join(self.name_feature,f"{symbol}.pkl"))[1][0].cpu())
                
            size.append([i.size(1) for i in current_emb])
            sample_emb.append(torch.cat(current_emb,1))
            
        return  sample_emb, sample_gene, size       

    
class TrainerModel(pl.LightningModule):
    
    def __init__(self, config,  model, meta, name_feature_path):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.automatic_optimization = False
        self.min_loss  = float("inf")
        self.max_corr  = float("-inf")
        self.max_eval_corr = float("-inf")
        self.min_eval_loss = float("inf")
        self.start_time  = None
        self.last_saved = None
        self.d = Description(meta, name_feature_path)
        
    @property
    def num_training_steps(self) -> int:
        dataset =  self.trainer._data_connector._train_dataloader_source.dataloader() #self.train_dataloader()
        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes) * self.trainer.num_nodes
        return len(dataset) // num_devices
    
    def correlationMetric(self,x, y):
      corr = 0
      for idx in range(x.size(1)):
          corr += pearsonr(x[:,idx], y[:,idx])
      corr /= (idx + 1)
      return (1 - corr).mean()
    def training_step(self,data,idx):
        
        if self.current_epoch == 0 and idx == 0:
            self.start_time  = time.time()
        
        optimizer = self.optimizers()
        
        emb, mask, size = self.d.sample_train()
        emb = [i.to(data["window"]["y"]) for i in emb]
        
        pred_count = self.model(data.x_dict,data.edge_index_dict, emb, size)
        loss   = self.criterion(pred_count,data["window"]["y"][:,mask])
        corrloss = self.correlationMetric(pred_count,data["window"]["y"])
        
        optimizer.zero_grad()
        self.manual_backward(loss + corrloss * 0.5)
        optimizer.step()
        
        self.produce_log(loss.detach(), 1 - corrloss.detach(),idx)
        
        
    def produce_log(self,loss,corr,idx):
        
        train_loss = self.all_gather(loss).mean().item()
        train_corr = self.all_gather(corr).mean().item()
        
        self.min_loss   = min(self.min_loss, train_loss)
        
        if self.trainer.is_global_zero and loss.device.index == 0 and idx % self.config.verbose_step == 0:
            
            current_lr = self.optimizers().param_groups[0]['lr']
            
            len_loader = self.num_training_steps
            
            batches_done = self.current_epoch  * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            time_left    = datetime.timedelta(seconds = batches_left * (time.time() - self.start_time) / batches_done)
                    
            self.config.logfun(
                        "[Epoch %d/%d] [Batch %d/%d] [Loss: %f, Corr: %f, lr: %f] [Min Loss: %f] ETA: %s" % 
                        (self.current_epoch,
                         self.trainer.max_epochs,
                         idx,
                         len_loader,
                         train_loss,
                         train_corr,
                         current_lr,
                         self.min_loss,
                         time_left
                            )
                        
                        )
            
    def validation_step(self,data,idx):
        emb, mask, size = self.d.load_test()
        emb = [i.to(data["window"]["y"]) for i in emb]
        mask = mask.to(data["window"]["y"].device) 
        pred_count = self.model(data.x_dict,data.edge_index_dict, emb, size)
        return pred_count,data["window"]["y"][:,mask]
        
    def validation_epoch_end(self,outputs):
        
        logfun = self.config.logfun
        
        pred_count = torch.cat([i[0] for i in outputs])
        count = torch.cat([i[1] for i in outputs])
        pred_count = self.all_gather(pred_count).view(-1,pred_count.size(-1))
        count = self.all_gather(count).view(-1,pred_count.size(-1))
        
        total_loss = self.criterion(pred_count,count).item()
        gene_corr = compute_correlations(count, pred_count, True)
        
        corr = np.mean(gene_corr)
        
        if self.trainer.is_global_zero and self.trainer.num_gpus != 0:
            if corr > self.max_eval_corr:
                 self.save(self.current_epoch, total_loss,corr)                    
            self.max_eval_corr = max(self.max_eval_corr,corr)
            self.min_eval_loss = min(self.min_eval_loss, total_loss)
                            
            logfun("==" * 25)
            logfun(
                "[Corr :%f, Loss: %f] [Min Loss :%f, Max Corr: %f]" %
                (corr,
                 total_loss,
                 self.min_eval_loss,
                 self.max_eval_corr,
                 )
                )            
            logfun("==" * 25)
        
    def save(self, epoch,loss, acc):
        
        self.config.logfun(self.last_saved)
        output_path = os.path.join(self.config.store_dir, "best.pt") 
        self.last_saved = output_path
        torch.save(self.model.state_dict(), output_path)
        self.config.logfun("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
                      
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
                            self.parameters(),
                            lr = self.config.lr,
                            betas = (0.9, 0.999),
                            weight_decay = self.config.weight_decay,
            )
        
        return optimizer                      
                      
                        
                        
                        
        
     
