#nvidia-smi
python -c 'import torch; print(torch.cuda.is_available()); print(torch.version.cuda)'
#cd ./src/models
#bash GD.sh
#cd ../..
python src/attacker.py