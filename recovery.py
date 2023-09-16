from utils import *
from models.classify import *
from models.generator import *
from models.discri import *
import torch
import os
import numpy as np
from attack import inversion, dist_inversion
from argparse import  ArgumentParser
import model
from generator import *
from discri import *
import torch, os, time, random, generator, discri
torch.manual_seed(9)

parser = ArgumentParser(description='Inversion')
parser.add_argument('--configs', type=str, default='./config/celeba/attacking/celeba.json')    

args = parser.parse_args()



def init_attack_args(cfg):
    if cfg["attack"]["method"] =='kedmi':
        args.improved_flag = True
        args.clipz = True
        args.num_seeds = 1
    else:
        args.improved_flag = False
        args.clipz = False
        args.num_seeds = 5

    if cfg["attack"]["variant"] == 'L_logit' or cfg["attack"]["variant"] == 'ours':
        args.loss = 'logit_loss'
    else:
        args.loss = 'cel'

    if cfg["attack"]["variant"] == 'L_aug' or cfg["attack"]["variant"] == 'ours':
        args.classid = '0,1,2,3'
    else:
        args.classid = '0'



if __name__ == "__main__":
    # global args, logger

    cfg = load_json(json_file=args.configs)
    init_attack_args(cfg=cfg)
    
    # Save dir
    if args.improved_flag == True:
        prefix = os.path.join(cfg["root_path"], "kedmi_300ids") 
    else:
        prefix = os.path.join(cfg["root_path"], "gmi_300ids") 
    save_folder = os.path.join("{}_{}".format(cfg["dataset"]["name"], cfg["dataset"]["model_name"]), cfg["attack"]["variant"])
    prefix = os.path.join(prefix, save_folder)
    save_dir = os.path.join(prefix, "latent")
    save_img_dir = os.path.join(prefix, "imgs_{}".format(cfg["attack"]["variant"]))
    args.log_path = os.path.join(prefix, "invertion_logs")

    os.makedirs(prefix, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)


    E = model.VGG16_V(1000)
    path_E = '/workspace/KDDMI/final_tars/VGG16_eval.tar'
    E = nn.DataParallel(E).cuda()
    checkpoint = torch.load(path_E)
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'])

    g_path = "KED_G.tar"
    G = generator.Generator()
    G = nn.DataParallel(G).cuda()
    ckp_G = torch.load(g_path)
    G.load_state_dict(ckp_G['state_dict'], strict=False)

    d_path = "KED_D.tar"
    D = discri.MinibatchDiscriminator()
    D = nn.DataParallel(D).cuda()
    ckp_D = torch.load(d_path)
    D.load_state_dict(ckp_D['state_dict'], strict=False)
    # Load models

    T = model.VGG16_V(2000)
    path_T = '/workspace/KDDMI/final_tars/MCKD.tar'
    T = nn.DataParallel(T).cuda()
    checkpoint = torch.load(path_T)
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'])
    targetnets = T
    fea_mean, fea_logvar = 0,0
    n_classes =1000
    N = 5
    bs = 60
    

    # Begin attacking
    for i in range(1):
        iden = torch.from_numpy(np.arange(bs))

        # evaluate on the first 300 identities only
        target_cosines = 0
        eval_cosines = 0
        for idx in range(5):
            iden = iden %n_classes
            print("--------------------- Attack batch [%s]------------------------------" % idx)
            print('Iden:{}'.format(iden))
            save_dir_z = '{}/{}_{}'.format(save_dir,i,idx)
            
            if args.improved_flag == True:
                #KEDMI
                print('kedmi')

                dist_inversion(G, D, targetnets, E, iden,  
                                        lr=cfg["attack"]["lr"], iter_times=cfg["attack"]["iters_mi"],
                                        momentum=0.9, lamda=100,  
                                        clip_range=1, improved=args.improved_flag, 
                                        num_seeds=args.num_seeds, 
                                        used_loss=args.loss,
                                        prefix=save_dir_z,
                                        save_img_dir=os.path.join(save_img_dir, '{}_'.format(idx)),
                                        fea_mean=fea_mean,
                                        fea_logvar=fea_logvar,
                                        lam=cfg["attack"]["lam"],
                                        clipz=args.clipz)
            else:
                #GMI
                print('gmi')
                if cfg["attack"]["same_z"] =='':
                    inversion(G, D, targetnets, E, iden,  
                                            lr=cfg["attack"]["lr"], iter_times=cfg["attack"]["iters_mi"], 
                                            momentum=0.9, lamda=100, 
                                            clip_range=1, improved=args.improved_flag,
                                            used_loss=args.loss,
                                            prefix=save_dir_z,
                                            save_img_dir=save_img_dir,
                                            num_seeds=args.num_seeds,                                        
                                            fea_mean=fea_mean,
                                            fea_logvar=fea_logvar,lam=cfg["attack"]["lam"],
                                            istart=args.istart)
                else:
                    inversion(G, D, targetnets, E, iden,  
                                            lr=args.lr, iter_times=args.iters_mi, 
                                            momentum=0.9, lamda=100, 
                                            clip_range=1, improved=args.improved_flag,
                                            used_loss=args.loss,
                                            prefix=save_dir_z,
                                            save_img_dir=save_img_dir,
                                            num_seeds=args.num_seeds,                                        
                                            fea_mean=fea_mean,
                                            fea_logvar=fea_logvar,lam=cfg["attack"]["lam"],
                                            istart=args.istart,
                                            same_z='{}/{}_{}'.format(args.same_z,i,idx))
            iden = iden + bs 

