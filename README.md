# HDDAE
The code for performing dereverberation in terms of HDDAE.

# Extracting log power spectrum: python build_data.py (Reverber_Inp) (LPS_Output)
python build_data.py ~/IDEA/train/3 ~/IDEA/Data/train/noisy_3
python build_data.py ~/IDEA/train/6 ~/IDEA/Data/train/noisy_6
python build_data.py ~/IDEA/train/9 ~/IDEA/Data/train/noisy_9
python build_data.py ~/IDEA/train/clean ~/IDEA/Data/train/clean

# Training process: 
python main.py --mode base --path base6/ --gpus 0

# Testing process:
python main.py --mode test --path base6/ --data 4 --gpus 0
python main.py --mode test --path base6/ --data 7 --gpus 0
python main.py --mode test --path base6/ --data 10 --gpus 0
python main.py --mode test --path base6/ --data 3 --gpus 0
python main.py --mode test --path base6/ --data 6 --gpus 0
python main.py --mode test --path base6/ --data 9 --gpus 0

# Reconstruction (from LPS to waveform)
python convert.py ~/IDEA/base6/3/
python convert.py ~/IDEA/base6/6/
python convert.py ~/IDEA/base6/9/
python convert.py ~/IDEA/base6/4/
python convert.py ~/IDEA/base6/7/
python convert.py ~/IDEA/base6/10/

# Please cite the paper:
W. Lee, S. Wang, F. Chen, X. Lu, S. Chien and Y. Tsao, "Speech Dereverberation Based on Integrated Deep and Ensemble Learning Algorithm," in Proc. ICASSP, pp. 5454-5458, 2018
