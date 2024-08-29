#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import tifffile
import numpy as np
from tqdm import tqdm
import argparse
from scanpy import read_visium
import scanpy as sc
import pandas as pd
import timm
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="cuda:0", type=str)
parser.add_argument("--model",    default="resnet18", type=str)
parser.add_argument("--file_path",default="../../../10xgenomics/",type=str)
parser.add_argument("--save_dir", default="../extracted_feature/resnet18/", type=str)


args = parser.parse_args() 

device = args.device
encoder = timm.create_model(args.model, pretrained=True,num_classes=0)
for p in encoder.parameters():
    p.requires_grad = False
encoder=encoder.cuda()
encoder.eval()

st_files = sorted([file.replace("/spatial/","") for file in glob.glob(args.file_path + "/**/spatial/", recursive=True)])
print(st_files)

batch_size = 64
save_dir = args.save_dir
window = 256      

data = []
for path in st_files:
    h5 = read_visium(path,count_file=f"{path.split(os.sep)[-1]}_filtered_feature_bc_matrix.h5")
    img = tifffile.imread(path + f"/{path.split(os.sep)[-1]}_image.tif")
    h5.var_names_make_unique()
    h5.var["mt"] = h5.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(h5, qc_vars=["mt"], inplace=True)
    coord = pd.DataFrame(h5.obsm['spatial'], columns=['x_coord', 'y_coord'], index=h5.obs_names)
    data.append([img,coord.values.astype(int)])
    print(coord.values.astype(int).shape)
    
mapping = []
for i,(img,coord) in enumerate(data):
    mappings = []
    for j in range(len(coord)):
        mappings.append(j)
    mapping.append(mappings)

data_config = {'input_size': (3, 256, 256),
             'interpolation': 'bicubic',
             'mean': (0.485, 0.456, 0.406),
             'std': (0.229, 0.224, 0.225),
             'crop_pct': 1.0,
             'crop_mode': 'center'}
transforms = timm.data.create_transform(**data_config, is_training=False)
del transforms.transforms[-2]
            
os.makedirs(save_dir,exist_ok=True)
def generate():
                    
    def get_slide_gene(idx,):
        
        img,coord = data[idx[0]]
        coord = coord[idx[1]]
        
        x,y = coord
        img = img[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]
        
        
        code = idx[1]
        img = torch.as_tensor(img,dtype=torch.float, device = device).permute(2,0,1) / 255   
               
        return transforms(img), code
    
    def extract(imgs):
        img = torch.stack(imgs)
        return encoder(img).view(-1,512)
    
    for i, k in tqdm(enumerate(mapping)):
        batch_img, codes = [], []
        img_embedding = []
        for j in k:
            img, code = get_slide_gene([i,j])
            batch_img.append(img)
            codes.append(code)
            
            if len(batch_img) == batch_size:
                img_embedding += [extract(batch_img)]
                batch_img =  []
            
        if len(batch_img) != 0:
            img_embedding += [extract(batch_img)]
        
        img_embedding = torch.cat(img_embedding).contiguous()
        assert (np.array(codes) == np.sort(codes)).all()
        assert  img_embedding.size(0) == len(codes)
        print(img_embedding.size())
        torch.save(img_embedding, f"{save_dir}/{i}.pt")

generate()      
          
                    
                    
        
        
                