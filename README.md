
Code for the following paper on [biorxiv](https://www.biorxiv.org/content/10.1101/2021.04.26.441401v1):

Title: Construction of a neural network energy function for protein physics

Authors: Huan Yang, Zhaoping Xiong, Francesco Zonta

Abstract:
Classical potentials are widely used to describe protein physics, due to their simplicity and accuracy, but they are continuously challenged as real applications become more demanding with time. Deep neural networks could help generating alternative ways of describing protein physics. Here we propose an unsupervised learning method to derive a neural network energy function for proteins. The energy function is a probability density model learned from plenty of 3D local structures which have been extensively explored by evolution. We tested this model on a few applications (assessment of protein structures, protein dynamics and protein sequence design), showing that the neural network can correctly recognize patterns in protein structures. In other words, the neural network learned some aspects of protein physics from experimental data.

### data
_need to upload the training and test data to dropbox._ 

nnef/dataset/data_chimeric.py is the Pytorch dataset. 

Local structure = central segment + nearest k (k=10) residues
Currently, the local structure is represented as ordered segments.

Local structure features include 
1) coordinates (r, theta, phi)
2) sequences (or evolutionary profiles / other sequence embedding features)
3) an indicator of the segments connectivity 

### ML model
model/local_ss.py is the latest model. 

The model components: 
1) linear embedding layers
2) a few trasformer layers with masked attention
3) linear output layers
4) calculate energy loss based on Gaussian mixtures model

### protein physics
nnef/physics/protein_os.py has a few components:
1) calculate energy of a full protein chain 
2) Monte Carlo search of the energy function
3) calculate gradients of the energy function and run dynamics


### training 
train_chimeric.py is the script for training. 

>python nnef/train_chimeric.py --data_flag hhsuite_CB_cullpdb_cath-alpha-beta_train.csv --seq_len 14 --seq_type residue --chimeric --residue_type_num 20 --epochs 1000 --val_data_flag hhsuite_CB_cullpdb_cath-alpha-beta_val.csv --test_data_flag hhsuite_CB_cullpdb_cath-ab.csv --batch_size 1000 --total_num_samples 1000000 --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 --mixture_r 2 --mixture_angle 3 --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 --lr 5e-5 --n_warmup_steps 5000 --steps_decay_scale 30000 --coords_angle_loss_lamda 1 --profile_loss_lamda 10 --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 --save_path runs/exp1 --log_interval 30 --save_interval 100 --num_workers=4 

### test

_need to upload pretrained model to dropbox_

1. scripts for scoring protein structures: 
* decoy_nm_score.py is for normal modes decoys;
* decoy_score.py is for a few decoy sets; 
* decoy_seq.py is for the sequence decoys; 
* score_md_trj.py is for MD trajectories.
   
for examples:
>python nnef/decoy_score.py --seq_len 14 --mode CB --seq_type residue     --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4     --mixture_r 2 --mixture_angle 3 --mixture_seq 1 --smooth_gaussian --smooth_r 0.3 --smooth_angle 45  --coords_angle_loss_lamda 1 --profile_loss_lamda 10 --radius_loss_lamda 1 --start_id_loss_lamda 1     --res_counts_loss_lamda 1  --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 --load_exp runs/exp1 --decoy_set 3DRobot_set --decoy_loss_dir decoy_loss_exp1

>python nnef/score_md_trj.py --seq_len 14 --mode CB --seq_type residue --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 --mixture_r 2 --mixture_angle 3 --mixture_seq 1 --smooth_gaussian --smooth_r 0.3 --smooth_angle 45  --coords_angle_loss_lamda 1 --profile_loss_lamda 10 --radius_loss_lamda 1 --start_id_loss_lamda 1 --res_counts_loss_lamda 1 --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 --load_exp runs/exp1 

   
2. fold_os.py and fold_os_deep.py are the scripts for sampling protein conformations. 

>python nnef/fold_os.py  --seq_len 14 --mode CB --seq_type residue     --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4     --mixture_r 2 --mixture_angle 3 --mixture_seq 1 --smooth_gaussian --smooth_r 0.3 --smooth_angle 45  --coords_angle_loss_lamda 1 --profile_loss_lamda 10 --radius_loss_lamda 1 --start_id_loss_lamda 1     --res_counts_loss_lamda 1  --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 --load_exp runs/exp1 --fold_engine dynamics --x_type cart --L 30000 --T_max 0.05 --lr 0.01 --save_dir exp1dynamics_val_deep

3. design_sample.py and design_sample_deep.py are the scripts for protein design. 

>python nnef/design_sample_deep.py --seq_len 14 --mode CB --seq_type residue     --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4     --mixture_r 2 --mixture_angle 3 --mixture_seq 1 --smooth_gaussian --smooth_r 0.3 --smooth_angle 45  --coords_angle_loss_lamda 1 --profile_loss_lamda 10 --radius_loss_lamda 1 --start_id_loss_lamda 1     --res_counts_loss_lamda 1  --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 --load_exp runs/exp1 --fold_engine anneal --random_init --L 1000 --T_max 4.2 --T_min 1.4 --save_dir exp1anneal_val_deep




