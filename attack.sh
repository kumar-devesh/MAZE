nvidia-smi
python -c 'import torch; print(torch.cuda.is_available()); print(torch.version.cuda)'
#cd ./src/models
#bash GD.sh
#cd ../..
python3 src/attacker.py --batch_size=4 --budget_gen=50000 --budget_clone=50000