from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle
from scanpy import read_visium
import scanpy as sc
import pandas as pd
import tifffile
import argparse
from collections import namedtuple
import torch_geometric
import glob
from tqdm import tqdm
        

parser = argparse.ArgumentParser()
parser.add_argument("--size",default=256,type=int)
parser.add_argument("--savename",default="../extracted_feature/resnet18/",type=str)
parser.add_argument("--file_path",default="../../../10xgenomics/",type=str)
parser.add_argument("--num_edge",default=5,type=int)

args = parser.parse_args()  
   
class TxPDataset(Dataset):
    def __init__(self, breast_cancers,index_filter, transform, args, train = None):
        self.args = args
        self.train = train
        self.index_filter = index_filter
        self.data = self.load_raw(args.data)
        print(f"loaded {len(self.data)}")
        self.meta_info(args.data)
        self.transform = transform

        self.max = torch.log10(torch.as_tensor(self.max,dtype=torch.float) + 1)
        self.min = torch.log10(torch.as_tensor(self.min,dtype=torch.float) + 1)
        
        del self.data

        mapping = []
        for i in breast_cancers:
            img,counts,coord,emb = self.all_data[i]
            for j in range(len(counts)):
                mapping.append([i,j])
        self.map = mapping


    def load_raw(self, data_root):

        data = []
        for idx, file in enumerate(st_files):
            path = os.path.join(data_root,file)
            h5 = read_visium(path,count_file=f"{path.split(os.sep)[-1]}_filtered_feature_bc_matrix.h5")
            img = tifffile.imread(path + f"/{path.split(os.sep)[-1]}_image.tif")
            h5.var_names_make_unique()
            h5.var["mt"] = h5.var_names.str.startswith("MT-")
            sc.pp.calculate_qc_metrics(h5, qc_vars=["mt"], inplace=True)
            data.append([img,h5,idx])

        return data

    def meta_info(self, root):

        gene_names = set()
        mapping_dict = dict()
        for _, p, _  in tqdm(self.data):
            
            mapping_dict.update(dict(zip(p.var.index.values, p.var.gene_ids.values)))
            counts = pd.DataFrame(p.X.todense(), columns=p.var_names, index=p.obs_names)
            coord = pd.DataFrame(p.obsm['spatial'], columns=['x_coord', 'y_coord'], index=p.obs_names)
            gene_names = gene_names.union(
                set(counts.columns.values)
                )

        gene_names = list(gene_names)
        gene_names.sort()

        with open("gene.pkl", "wb") as f:
            pickle.dump(gene_names, f)
            print(len(gene_names))
            print("write gene name")

        with open("mapping.pkl", "wb") as f:
            pickle.dump(mapping_dict, f)
            print(len(mapping_dict))
            print("write mapping_dict")

        all_data = {}
        all_gene = []
        for img,p,idx in tqdm(self.data):
            counts = pd.DataFrame(p.X.todense(), columns=p.var_names, index=p.obs_names)
            coord = pd.DataFrame(p.obsm['spatial'], columns=['x_coord', 'y_coord'], index=p.obs_names)

            missing = list(set(gene_names) - set(counts.columns.values))
            c = counts.values.astype(float)
            pad = np.zeros((c.shape[0], len(missing)))
            c = np.concatenate((c, pad), axis=1)

            names = np.concatenate((counts.columns.values, np.array(missing)))
            c = c[:, np.argsort(names)]

            emb = torch.load(f"{self.args.emb_path}/{idx}.pt",map_location=torch.device("cpu"))

            assert emb.size(0) == c.shape[0]

            all_data[idx] = [img,c,coord.values.astype(int),emb]

            for i in c:
                all_gene.append(i)

        all_gene = np.array(all_gene)

        print(all_gene.shape)

        np.save("mean_expression.npy", np.mean(all_gene, 0))
        np.save("median_expression.npy", np.median(all_gene, 0))
        np.save("max_expression.npy", np.max(all_gene, 0))
        np.save("min_expression.npy", np.min(all_gene, 0))

        self.mean = np.mean(all_gene, 0)
        self.max  = np.max(all_gene,0)
        self.min  = np.min(all_gene,0)
        self.gene_names = gene_names
        self.all_data = all_data

        with open("gene.pkl", "wb") as f:
            pickle.dump(gene_names, f)

    
    def generate(self, idx):

        idx = self.map[idx]
        img,counts,coord,emb,  = self.all_data[idx[0]]
        counts, coord, emb = counts[idx[1]],coord[idx[1]],emb[idx[1]]
        
        x,y = coord
        window = self.args.size
        pos = [x//window, y//window]

        counts = torch.log10(torch.as_tensor(counts,dtype = torch.float) + 1)
        counts = (counts - self.min) / (self.max - self.min + 1e-8)
        
        return {
             "count" : counts,
             "p_feature": emb,
             "pos": torch.LongTensor(pos),
            }

    def  __getitem__(self, index):
        return self.generate(index)

    def __len__(self):
        return len(self.map)
  
st_files = sorted([file.replace(args.file_path, "").replace("/spatial/","") for file in glob.glob(args.file_path + "/**/spatial/", recursive=True)])
print(st_files)

def get_edge(x):
    edge_index = torch_geometric.nn.radius_graph(
            x,
            np.sqrt(2),
            None,
            False,
            max_num_neighbors=5,
            flow="source_to_target",
            num_workers=1,
        )
    return edge_index

def get_knn_edge(x,num_edge=5):
    edge_index = torch_geometric.nn.knn_graph(
        x,
        k=num_edge,
        )
    return edge_index


os.makedirs(os.path.join(args.savename, "graph"),exist_ok=True)

arg = namedtuple("arg",["size","emb_path","data"])
arg = arg(args.size, args.savename,args.file_path)  

for iid in tqdm(range(len(st_files))):
    dataset = TxPDataset([iid], None, None, arg, False)
    loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    drop_last=False,
    )
    img_data = []
    for x in loader:
        pos, p, py = x["pos"], x["p_feature"], x["count"]
        img_data.append([pos, p, py])
    
    window_edge = get_edge(torch.cat(([i[0] for i in img_data])).clone())
    
    knn_edge = get_knn_edge(torch.cat(([i[1] for i in img_data])).clone(),args.num_edge)
    
    print(window_edge.size(), knn_edge.size())
    
    
    data = torch_geometric.data.HeteroData()
    
    data["window"].pos = torch.cat(([i[0] for i in img_data])).clone()
    data["window"].x = torch.cat(([i[1] for i in img_data])).clone()
    data["window"].y = torch.cat(([i[2] for i in img_data])).clone()
       
    assert len(data["window"]["pos"]) == len(data["window"]["x"]) == len(data["window"]["y"])
    
    
    data['window', 'near', 'window'].edge_index = window_edge
    data['window', 'knn', 'window'].edge_index = knn_edge
    
    torch.save(data, os.path.join(args.savename, "graph", f"{iid}.pt"))
    print()
