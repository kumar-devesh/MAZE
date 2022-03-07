import argparse
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import random
import numpy as np

seed = 2021
random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.multiprocessing.set_sharing_strategy("file_system")


from utils.simutils.timer import timer
from utils.config import parser
from models import get_model
from datasets import get_dataset
from utils.helpers import test
from attacks import (
    knockoff,
    noise,
    jbda,
    maze,
)

def attack():
    savedir = "{}/{}/{}/".format(args.logdir, args.dataset, args.model_victim)

    #to be removed
    train_loader, test_loader = get_dataset(args.dataset, args.batch_size, train_and_test=True)

    T = get_model(args, args.model_victim, args.n_classes, args.dataset)  # Target (Teacher)
    S = get_model(args, args.model_clone, args.n_classes, args.dataset)  # Clone  (Student)
    S = S.to(args.device)
    T = T.to(args.device)

    #check the target model accuracy on test_dataset

    ############################dummy####################
    #_, tar_acc = test(T, args.device, test_loader)
    #print("* Loaded Target Model *")
    #print("Target Model Accuracy: {:.2f}\n".format(tar_acc))
    tar_acc=100
    #######################################################

    #perform the attack
    if args.attack == "noise":
        noise(args, T, S, test_loader, tar_acc)
    elif args.attack == "knockoff":
        knockoff(args, T, S, test_loader, tar_acc)
    elif args.attack == "jbda":
        jbda(args, T, S, train_loader, test_loader, tar_acc)
    elif args.attack == "maze":
        maze(args, T, S, train_loader, test_loader, tar_acc)
    else:
        sys.exit("Unknown Attack {}".format(args.attack))

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    torch.save(S.state_dict(), savedir + "{}.pt".format(args.attack))
    print("* Saved Sur model * ")


def main():
    pid = os.getpid()
    print("pid: {}".format(pid))
    timer(attack) #calls the attack function with start, end (time.time())
    exit(0)


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='MAZE')

    parser.add_argument('--wandb_project', type=str, default="trial", help='wandb project name')
    parser.add_argument('--dataset', type=str, default="randomvideoslikekinetics400", help='eval dataset')
    parser.add_argument('--n_classes', type=int, default=400, help='number of classes in the dataset')
    #parser.add_argument('--budget', type=int, default=5e8, help='query budget')


    parser.add_argument('--attack', type=str, default="maze", help='attack type')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    #parser.add_argument('--model_victim', type=str, default="ResNet3d_T", help='victim model to be used')     
    #parser.add_argument('--model_clone', type=str, default="ResNet3d_S", help='clone attacker model')
    #parser.add_argument('--model_gen', type=str, default="Generator_cgen", help='clone attacker model')
    #parser.add_argument('--latent_dim', type=int, default=7, help='latent dim for generator ((16x)*(16x)) generated image resolution')

    parser.add_argument('--device', type=str, default="gpu", help='`gpu`/`cpu` device')
    parser.add_argument('--opt', type=str, default="adam", help='sgd for sgd, otherwize adam is used')
    parser.add_argument('--logdir', type=str, default="checkpoints", help='checkpoints directory')
    parser.add_argument('--white_box', type=bool, default=False, help='True if whitebox training (backprop through the model)')

    parser.add_argument('--alpha_gan', type=float, default=0.0, help='positive weight for PD setting')
    #parser.add_argument('--in_dim', type=int, default=120, help='generator input dimension for embedding')
    parser.add_argument('--lr_gen', type=float, default=1e-3, help='lr for generator model')  
    parser.add_argument('--lr_clone', type=float, default=0.1, help='lr for clone model') 
    parser.add_argument('--ndirs', type=int, default=1, help='number of directions for gradient estimation') 
    parser.add_argument('--mu', type=float, default=1e-3, help='epsilon value for normalized noise') 

    parser.add_argument('--iter_clone', type=int, default=5, help='iter_clone for clone model')
    parser.add_argument('--iter_gen', type=int, default=5, help='iter_gen for generator model')
    parser.add_argument('--iter_exp' ,type=int, default=10, help='iter_exp gives the number of experience replay iterations')
    parser.add_argument('--log_iter', type=int, default=1e4, help='log iterations')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1') #for adam optimizer
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2')

    parser.add_argument('--resume_training', type=bool, default=False, help='bool value: resume training from checkpoint')
    parser.add_argument('--PATH', type=str, default=" ", help='checkpoint path "./checkpoints/model.pth"')

    ################################demo add################################
    parser.add_argument('--model_gen', type=str, default="simple_gen", help='clone attacker model')
    parser.add_argument('--latent_dim', type=int, default=40, help='toy model')
    parser.add_argument('--model_victim', type=str, default="simple_dis", help='victim model to be used')     
    parser.add_argument('--model_clone', type=str, default="simple_dis", help='clone attacker model')
    parser.add_argument('--in_dim', type=int, default=40, help='generator input dimension for embedding')
    parser.add_argument('--budget', type=int, default=30, help='query budget')
    ########################################################################
    args = parser.parse_args()
    
    if args.device=="cpu":
        args.device = torch.device(args.device) 
    else:
        args.device = torch.device("cuda") 

    wandb.init(project=args.wandb_project)
    run_name = "{}_{}".format(args.dataset, args.attack)
    if args.attack == "maze":
        if args.alpha_gan > 0:
            run_name = "{}_{}".format(args.dataset, "pdmaze")
        budget_M = args.budget / 1e6
    
        if args.white_box:
            grad_est = "wb"
        else:
            grad_est = "nd{}".format(args.ndirs)
    
        if args.iter_exp > 0:
            run_name += "_{:.2f}M_{}".format(budget_M, grad_est)
        else:
            run_name += "_{:.2f}M_{}_noexp".format(budget_M, grad_est)
    
    wandb.run.name = run_name
    wandb.run.save()
    
    # Select hardware
    
    if args.device == "gpu":
        import torch.backends.cudnn as cudnn
    
        cudnn.enabled = True
        cudnn.benchmark = True
        #args.device = "cuda"
    #else:
    #    args.device = "cpu"
    main()
