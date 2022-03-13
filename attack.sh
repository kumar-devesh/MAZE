nvidia-smi
python -c 'import torch; print(torch.cuda.is_available()); print(torch.version.cuda)'
#cd ./src/models
#bash GD.sh
#cd ../..

#python3 src/attacker.py --batch_size=4 --budget_gen=300000 --budget_clone=50000 --std_wt=0.0
python3 src/attacker.py --batch_size=1 --budget_gen=100 --budget_clone=4 --std_wt=0.0