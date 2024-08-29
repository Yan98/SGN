import argparse
import os
from model import GNN
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from train import TrainerModel
from sklearn.model_selection import KFold
import glob
import torch_geometric


def load_dataset(pts,file_path):
    all_files = sorted(glob.glob(f"{file_path}/*.pt"))
    print(all_files)
    selected_files = []
    for i in all_files:
        for j in pts:
            if i.endswith(str(j) + ".pt"):
                graph = torch.load(i)
                print(graph)
                selected_files.append(graph)
    return selected_files

def main(args,idx):


    XFOLD = glob.glob(f"{args.file_path}/*.pt")
    skf = KFold(n_splits=3,shuffle= True, random_state = 12345)
    KFOLD = []
    for x in skf.split(XFOLD):
        KFOLD.append(x)    


    cwd = os.getcwd()
    
    def write(director,name,*string):
        string = [str(i) for i in string]
        string = " ".join(string)
        with open(os.path.join(director,name),"a") as f:
            f.write(string + "\n")
            
    args.folder_name = "log" + "/"  + str(idx)
    store_dir = args.folder_name + "/" + "checkpoints_" + str(args.fold) + "/" 
    print = partial(write,cwd, args.folder_name + "/" +"log_f" + str(args.fold))
      
    os.makedirs(store_dir, exist_ok= True)
    
    print(args)
    

    train_patient, test_patient = KFOLD[args.fold]
    
    train_dataset = load_dataset(train_patient,args.file_path)
    test_dataset = load_dataset(test_patient,args.file_path)
    
    train_loader = torch_geometric.loader.DataLoader(
        train_dataset,
        batch_size=1,
        )
    
    test_loader = torch_geometric.loader.DataLoader(
        test_dataset,
        batch_size=1,
        )
    
    print(len(train_loader), len(test_loader))
    
    model = GNN(args.hidden_channels, args.embed_dim, args.out_channels, args.gnn_layer,args.feature_dim,args.name_dim)
    CONFIG = collections.namedtuple('CONFIG', ['lr', 'logfun', 'verbose_step', 'weight_decay', 'store_dir'])
    config = CONFIG(args.lr, print, args.verbose_step, args.weight_decay,store_dir)
        
    if args.checkpoints != None:
        model.load_state_dict(torch.load(args.checkpoints,map_location = torch.device("cpu")))
    
    model = TrainerModel(config, model,args.meta, args.name_feature)
    
    plt = pl.Trainer(max_epochs = args.epoch,num_nodes=1, gpus=args.gpus, val_check_interval = args.val_interval,checkpoint_callback = False, logger = False)
    plt.fit(model,train_dataloaders=train_loader,val_dataloaders=test_loader)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--epoch", default = 300, type = int)
    parser.add_argument("--fold", default = 0, type = int)
    parser.add_argument("--gpus", default = 1, type = int)
    parser.add_argument("--acce", default = "ddp", type = str)
    parser.add_argument("--val_interval", default = 0.8, type = float)
    parser.add_argument("--lr", default = 1e-4*5, type = float)
    parser.add_argument("--verbose_step", default = 10, type = int)
    parser.add_argument("--weight_decay", default = 1e-4, type = float)
    parser.add_argument("--checkpoints", default = None, type = str)
    parser.add_argument("--output", default = None, type = str)
    parser.add_argument("--folder_name", default = "log", type = str)
    parser.add_argument("--runs", default = 1, type = int)
    parser.add_argument("--file_path", default="extracted_feature/resnet18/graph", type = str)
    parser.add_argument("--name_feature", default="name_feature/Intel/neural-chat-7b-v3-1", type = str)
    parser.add_argument("--meta", default="preprocess/", type = str)
    parser.add_argument("--feature_dim", default=512, type = int)
    parser.add_argument("--name_dim", default=4096, type = int)
    parser.add_argument("--hidden_channels", default=512, type = int)
    parser.add_argument("--embed_dim", default=256, type = int)
    parser.add_argument("--out_channels", default=256, type = int)
    parser.add_argument("--gnn_layer", default=4, type = int)
    
    
    args = parser.parse_args()
    for idx in range(args.runs):
        for fold in range(3):
            args.fold = fold
            main(args,idx)
    
