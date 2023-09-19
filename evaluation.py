from argparse import  ArgumentParser
from metrics.eval_accuracy import eval_accuracy, eval_acc_class
from utils import load_json, get_attack_model
import os
import csv 
import model
import generator 
from discri import *
parser = ArgumentParser(description='Evaluation')
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


if __name__ == '__main__':
    # Load Data
    cfg = load_json(json_file=args.configs)
    init_attack_args(cfg=cfg)

    # Save dir
    if args.improved_flag == True:
        prefix = os.path.join(cfg["root_path"], "kedmi_300ids") 
    else:
        prefix = os.path.join(cfg["root_path"], "gmi_300ids") 
    save_folder = os.path.join("{}_{}".format(cfg["dataset"]["name"], cfg["dataset"]["model_name"]), cfg["attack"]["variant"])
    prefix = os.path.join(prefix, save_folder)
    save_dir = 'attack_results/kedmi_300ids/celeba_VGG16/L_logit/latent/'
    save_img_dir = os.path.join(prefix, "imgs_{}".format(cfg["attack"]["variant"]))

    # Load models
    Es = []
    E_VGG = model.VGG16_V(1000)
    path_E = '/workspace/KDDMI/final_tars/eval/VGG16_80.09.tar'
    E_VGG = nn.DataParallel(E_VGG).cuda()
    checkpoint = torch.load(path_E)
    ckp_E = torch.load(path_E)
    E_VGG.load_state_dict(ckp_E['state_dict'])
    Es.append(E_VGG)

    E_VIB = model.VGG16_vib(1000)
    path_E = '/workspace/KDDMI/final_tars/eval/VIB_teacher_0.010_62.25.tar'
    E_VIB = nn.DataParallel(E_VIB).cuda()
    checkpoint = torch.load(path_E)
    ckp_E = torch.load(path_E)
    E_VIB.load_state_dict(ckp_E['state_dict'])
    Es.append(E_VIB)
    
    E_HSIC = model.VGG16(1000,True)
    path_E = '/workspace/KDDMI/final_tars/teacher/BiDO_teacher_71.78_0.1_0.1.tar'
    E_HSIC = nn.DataParallel(E_HSIC).cuda()
    checkpoint = torch.load(path_E)
    ckp_E = torch.load(path_E)
    E_HSIC.load_state_dict(ckp_E['state_dict'])
    Es.append(E_HSIC)

    E_KD = model.VGG16_V(2000)
    path_E = '/workspace/KDDMI/final_tars/MCKD.tar'
    E_KD = nn.DataParallel(E_KD).cuda()
    checkpoint = torch.load(path_E)
    ckp_E = torch.load(path_E)
    E_KD.load_state_dict(ckp_E['state_dict'])
    Es.append(E_KD)

    g_path = "KED_G.tar"
    G = generator.Generator()
    G = nn.DataParallel(G).cuda()
    ckp_G = torch.load(g_path)
    G.load_state_dict(ckp_G['state_dict'], strict=False)

    # Metrics



    
    for e in Es:
        metric = cfg["attack"]["eval_metric"].split(',')
        fid = 0
        aver_acc, aver_acc5, aver_std, aver_std5 = 0, 0, 0, 0
        knn = 0, 0
        nsamples = 0 
        dataset, model_types = '', ''
        aver_acc, aver_acc5, aver_std, aver_std5 = eval_accuracy(G=G, E=e, save_dir=save_dir, args=args)
        aver_acc *= 5
        aver_acc5 *= 5
        aver_std *=5
        aver_std5 *=5

        
        csv_file = os.path.join(prefix, 'Eval_results.csv') 
        if  os.path.exists(csv_file):
            header = ['Save_dir', 'Method', 'Succesful_samples',                    
                        'acc','std','acc5','std5',
                        'fid','knn']
            with open(csv_file, 'w') as f:                
                writer = csv.writer(f)
                writer.writerow(header)
        
        fields=['{}'.format(save_dir), 
                '{}'.format(cfg["attack"]["method"]),
                '{}'.format(cfg["attack"]["variant"]),
                '{:.2f}'.format(aver_acc),
                '{:.2f}'.format(aver_std),
                '{:.2f}'.format(aver_acc5),
                '{:.2f}'.format(aver_std5),
                ]
        
        print("---------------Evaluation---------------")
        print('Method: {} '.format(cfg["attack"]["method"]))

        print('Variant: {}'.format(cfg["attack"]["variant"]))
        print('Top 1 attack accuracy:{:.2f} +/- {:.2f} '.format(aver_acc, aver_std))
        print('Top 5 attack accuracy:{:.2f} +/- {:.2f} '.format(aver_acc5, aver_std5))     
        
        print("----------------------------------------")  
        print()
        print()
        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
