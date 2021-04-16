#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "

OPTS+="--list_train data/MUSIC_train.csv "
OPTS+="--list_val data/MUSIC_val.csv "
#OPTS+="--list_val data/MUSIC_test.csv "

# Models
#OPTS+="--arch_sound unet7 "                            #U-Net
OPTS+="--arch_sound deeplabV3Plus_mobilenetv2 "         #MV2
#OPTS+="--arch_frame resnet18dilated "                  #Res-18
OPTS+="--arch_frame resnet18dilated_50 "                #Res-50
OPTS+="--arch_avol AVOL "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 11 "
OPTS+="--num_class 11 "

# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "

# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 1 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "
OPTS+="--stft_frame 1022 "
OPTS+="--stft_hop 256 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--workers 12 "
OPTS+="--batch_size_per_gpu 10 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_avol 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 40 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

OPTS+="--ckpt ./ckpt_train_sep "
#OPTS+="--ckpt ./ckpt_train_locSep "
#OPTS+="--ckpt ./ckpt_train_CatEmb "

OPTS+="--dup_trainset 100 "
OPTS+="--lamda 1.0 "
OPTS+="--dataset MUSIC "

CUDA_VISIBLE_DEVICES="0" python -u main_Appearance_Sound.py $OPTS
#CUDA_VISIBLE_DEVICES="0" python -u main_Appearance_att_Sound.py $OPTS
#CUDA_VISIBLE_DEVICES="0" python -u main_CatEmb_Sound.py $OPTS
