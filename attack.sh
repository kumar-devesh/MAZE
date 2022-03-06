nvidia-smi
python -c 'import torch; print(torch.cuda.is_available()); print(torch.version.cuda)'
python src/attacker.py --dataset=fashionmnist --model_tgt=lenet --model_clone=wres22 --attack=maze --budget=5e6 --log_iter=1e5 --lr_clone=0.1 --lr_gen=1e-3 --iter_clone=5 --iter_exp=10