#!/bin/bash

OPTS=""
OPTS+="--mode eval "

### A(Res-50) + S(DV3P)
#OPTS+="--ckpt ./ckpt_res50_DV3P_MUSIC_N2_f1_binary_bs10_TrainS335_D65_ValValS100_ValTestS130_dup100_f8fps_11k "
#OPTS+="--id MUSIC-2mix-LogFreq-resnet18dilated_50-deeplabV3Plus_mobilenetv2-frames1stride24-maxpool-binary-weightedLoss-channels11-epoch100-step40_80 "

### A(Res-50, att) + S(DV3P)
OPTS+="--ckpt ./ckpt_res50_att_DV3P_MUSIC_N2_f1_binary_bs10_TrainS335_D65_ValValS100_ValTestS130_dup100_f8fps_11k "
OPTS+="--id MUSIC-2mix-LogFreq-resnet18dilated_50-deeplabV3Plus_mobilenetv2-AVOL-frames1stride24-maxpool-binary-weightedLoss-channels11-epoch100-step40_80 "

### A(Ground Category Emb) + S(DV3P)
#OPTS+="--ckpt ./ckpt_GCEmb_DV3P_MUSIC_N2_f1_binary_bs10_TrainS335_D65_ValValS100_ValTestS130_dup100_f8fps_11k "
#OPTS+="--id MUSIC-2mix-LogFreq-deeplabV3Plus_mobilenetv2-frames1stride24-maxpool-binary-weightedLoss-channels11-epoch100-step40_80 "


# Data Lists
OPTS+="--list_train data/musicall_TrainS335_D65_f8fps_11k.csv "
#OPTS+="--list_val data/musicall_ValValS100_f8fps_11k.csv "
OPTS+="--list_val data/musicall_ValTestS130_f8fps_11k.csv "

# Models
#OPTS+="--arch_sound unet7 "				#U-Net
OPTS+="--arch_sound deeplabV3Plus_mobilenetv2 "		#DV3P
#OPTS+="--arch_frame resnet18dilated " 			#Res-18
OPTS+="--arch_frame resnet18dilated_50 "		#Res-50
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

OPTS+="--num_gpus 1 "
OPTS+="--workers 12 "
OPTS+="--batch_size_per_gpu 10 "
OPTS+="--num_vis 200 "


#CUDA_VISIBLE_DEVICES="0" python -u main_Appearance_Sound.py $OPTS
CUDA_VISIBLE_DEVICES="0" python -u main_Appearance_att_Sound.py $OPTS
#CUDA_VISIBLE_DEVICES="0" python -u main_GCEmb_Sound.py $OPTS
