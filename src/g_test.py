import argparse
from email.quoprimime import header_check
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
from attacks.maze import train_generator, train_student
from models.Generator import Generator
from attacker import parse_arguments
from models.models import get_model
from attacks.attack_utils import generate_images

def test(args, n_samples = 25):
    loaddir = "{}/{}/{}/".format(args.logdir, args.dataset, args.model_victim)

    #load final weights
    path = loaddir+"{}_final.pt".format(args.attack)
    G = get_model(args, modelname = args.model_gen, n_classes=args.n_classes, dataset = args.dataset, latent_dim=args.latent_dim)
    G = G.to(args.device)
    G.eval()

    T = get_model(args, args.model_victim, args.n_classes, args.dataset)  # Target (Teacher)
    T = T.to(args.device)

    S = get_model(args, args.model_clone, args.n_classes, args.dataset)  # Target (Teacher)
    S = S.to(args.device)

    with torch.no_grad():
        gen_samples = []
        for i in range(n_samples//args.batch_size):
            z = torch.randn(args.batch_size, args.in_dim).to(args.device)
            class_label = torch.randint(low=0, high=args.n_classes, size=(args.batch_size,)).to(args.device)
            samples, _ = G(z, class_label)
            #print(torch.argmax(T(samples), dim=-1))
            print("Teacher Logits:", torch.max(T(samples, print_outputs=True), dim=-1))
            #print("Student Logits", S(samples))
            gen_samples.append(samples)
            x = generate_images(args, G, z, class_label, "G")

        gen_samples = torch.cat(tensors=gen_samples, dim=0)
        print(gen_samples.size())
    


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = parse_arguments()
    args = parser.parse_args()

    wandb.init(project=args.wandb_project)

    if args.device=="cpu":
        args.device = torch.device(args.device) 
    else:
        args.device = torch.device("cuda") 
    
    if args.device == "gpu":
        import torch.backends.cudnn as cudnn
    
        cudnn.enabled = True
        cudnn.benchmark = True
        #args.device = "cuda"
    #else:
    #    args.device = "cpu"
    test(args)