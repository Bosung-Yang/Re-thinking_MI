from utils import *
from models.classify import *
from models.generator import *
from models.discri import *
import torch
import numpy as np

from attack import attack_acc
import statistics 

from metrics.fid import concatenate_list, gen_samples


device = torch.torch.cuda.is_available()

def accuracy(fake_dir, E):
    
    aver_acc, aver_acc5, aver_std, aver_std5 = 0, 0, 0, 0

    N = 5
    E.eval()
    for i in range(1):      
        all_fake = np.load(fake_dir+'full.npy',allow_pickle=True)  
        all_imgs = all_fake.item().get('imgs')
        all_label = all_fake.item().get('label')

        # calculate attack accuracy
        with torch.no_grad():
            N_succesful = 0
            N_failure = 0

            for random_seed in range(len(all_imgs)):
                if random_seed % N == 0:
                    res, res5 = [], []
                    
                #################### attack accuracy #################
                fake = all_imgs[random_seed]
                label = all_label[random_seed]

                label = torch.from_numpy(label)
                fake = torch.from_numpy(fake)

                acc,acc5 = attack_acc(fake,label,E)

                
                print("Seed:{} Top1/Top5:{:.3f}/{:.3f}\t".format(random_seed, acc,acc5))
                res.append(acc)
                res5.append(acc5)
                

                if (random_seed+1)%5 == 0:      
                    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
                    std = statistics.stdev(res)
                    std5 = statistics.stdev(res5)

                    print("Top1/Top5:{:.3f}/{:.3f}, std top1/top5:{:.3f}/{:.3f}".format(acc, acc_5, std, std5))

                    aver_acc += acc / N
                    aver_acc5 += acc5 / N
                    aver_std += std / N
                    aver_std5 +=  std5 / N
            print('N_succesful',N_succesful,N_failure)


    return aver_acc, aver_acc5, aver_std, aver_std5



def eval_accuracy(G, E, save_dir, args):
    
    successful_imgs, _ = gen_samples(G, E, save_dir, args.improved_flag)
    
    aver_acc, aver_acc5, \
    aver_std, aver_std5 = accuracy(successful_imgs, E)
    
    
    return aver_acc, aver_acc5, aver_std, aver_std5

def acc_class(filename,fake_dir,E):
    
    E.eval()

    sucessful_fake = np.load(fake_dir + 'success.npy',allow_pickle=True)  
    sucessful_imgs = sucessful_fake.item().get('sucessful_imgs')
    sucessful_label = sucessful_fake.item().get('label')
    sucessful_imgs = concatenate_list(sucessful_imgs)
    sucessful_label = concatenate_list(sucessful_label)

    N_img = 5
    N_id = 300
    with torch.no_grad():
        acc = np.zeros(N_id)
        for id in range(N_id):                
            index = sucessful_label == id
            acc[id] = sum(index)
            
    acc=acc*100.0/N_img 
    print('acc',acc)
    csv_file = '{}acc_class.csv'.format(filename)
    print('csv_file',csv_file)
    import csv
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        for i in range(N_id):
            # writer.writerow(['{}'.format(i),'{}'.format(acc[i])])
            writer.writerow([i,acc[i]])

def eval_acc_class(G, E, save_dir, prefix, args):
    
    successful_imgs, _ = gen_samples(G, E, save_dir, args.improved_flag)
    
    filename = "{}/{}_".format(prefix, args.loss)
    
    acc_class(filename,successful_imgs,E)

def gen_samples(G, E, save_dir, improved_gan, n_iden=5, n_img=5):
    total_gen = 0
    seed = 9
    torch.manual_seed(seed)
    img_ids_path = os.path.join(save_dir, 'attack{}_'.format(seed))

    all_sucessful_imgs = []
    all_failure_imgs = []
    
    all_imgs = []                            
    all_fea = []
    all_id = []
    all_sucessful_imgs = []
    all_sucessful_id =[]
    all_sucessful_fea=[]

    all_failure_imgs = []                            
    all_failure_fea = []
    all_failure_id = [] 
    
    E.eval()
    G.eval()
    if not os.path.exists(img_ids_path + 'full.npy'):
        for loop in range(1):
            for i in range(n_iden): #300 ides 
                for j in range(n_img): #5 images/iden
                    z, iden = get_z(improved_gan, save_dir, loop, i, j)
                    z = torch.clamp(z, -1.0, 1.0).float()
                    total_gen = total_gen + z.shape[0]
                    # calculate attack accuracy
                    with torch.no_grad():
                        fake = G(z.to(device))
                        save_tensor_images(fake, os.path.join(save_dir, "gen_{}_{}.png".format(i,j)), nrow = 60)

                        eval_fea, eval_prob = E(utils.low2high(fake))
                        
                        ### successfully attacked samples       
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        sucessful_iden = []
                        failure_iden = []
                        for id in range(iden.shape[0]):
                            if eval_iden[id]==iden[id]:
                                sucessful_iden.append(id)
                            else:
                                failure_iden.append(id)
                        
                        
                        fake = fake.detach().cpu().numpy()
                        eval_fea = eval_fea.detach().cpu().numpy()  
                    
                        all_imgs.append(fake)
                        all_fea.append(eval_fea)
                        all_id.append(iden)
                          
                        if len(sucessful_iden)>0:                              
                            sucessful_iden = np.array(sucessful_iden)                            
                            sucessful_fake = fake[sucessful_iden,:,:,:]                    
                            sucessful_eval_fea = eval_fea[sucessful_iden,:]
                            sucessful_iden = iden[sucessful_iden]
                        else:
                            sucessful_fake = []
                            sucessful_iden = []
                            sucessful_eval_fea = []
                        
                        all_sucessful_imgs.append(sucessful_fake)
                        all_sucessful_id.append(sucessful_iden)
                        all_sucessful_fea.append(sucessful_eval_fea)

                        if len(failure_iden)>0: 
                            failure_iden = np.array(failure_iden)
                            failure_fake = fake[failure_iden,:,:,:]                    
                            failure_eval_fea = eval_fea[failure_iden,:]
                            failure_iden = iden[failure_iden]
                        else:
                            failure_fake = []
                            failure_iden = []
                            failure_eval_fea = []
              
                        all_failure_imgs.append(failure_fake)
                        all_failure_id.append(failure_iden)
                        all_failure_fea.append(failure_eval_fea)     
        np.save(img_ids_path+'full',{'imgs':all_imgs,'label':all_id,'fea':all_fea})
        np.save(img_ids_path+'success',{'sucessful_imgs':all_sucessful_imgs,'label':all_sucessful_id,'sucessful_fea':all_sucessful_fea})
        np.save(img_ids_path+'failure',{'failure_imgs':all_failure_imgs,'label':all_failure_id,'failure_fea':all_failure_fea})
        
    return img_ids_path, total_gen