# Separating-Sounds-from-a-Single-Image

[Paper](https://arxiv.org/pdf/2007.07984.pdf) | [project](https://ly-zhu.github.io/separating-sounds-from-single-image)

A PyTorch implementation of "Separating Sounds from a Single Image". Authors: [Lingyu Zhu](https://ly-zhu.github.io) and [Esa Rahtu](http://esa.rahtu.fi). Tampere University, Finland.

<img src="figures/locSep3_MUSIC.png" width="800"/>

<!-- ## Examples of Sound Source Separation
<img src="separating-sounds-from-single-image/figures/locSep_vis_MUSIC.png" width="800"/>

## Examples of Sound Source Localization
<img src="separating-sounds-from-single-image/figures/loc_vis_MUSIC_res50_dv3p.png" width="800"/>
-->

# Environment
	Python>=3.5, PyTorch>=0.4.0

# Preparing the data (not released yet, you can train the model on your own dataset for now by setting the following info)
	# Place the csv file lists under the folder data, the csv file has the format as below: 
		audio_path, frames_path, frames count
	# Edit the dataset path at line 163 of file dataset/music.py
		

# Model selection
Replace the --arch_frame and --arch_sound in scripts/train_locSep.sh to switch to diffeent appearance and sound networks.

# Training
	# Training the A(Res-50) + S(DV3P) model
	./scripts/train_locSep.sh

	# Training the A(Res-50, att) + S(DV3P) model
	-The network A(Res-50, att) + S(DV3P) is trained based on A(Res-50) + S(DV3P). 
	-Uncomment the line of "CUDA_VISIBLE_DEVICES="0" python -u main_Appearance_att_Sound.py $OPTS" in scripts/train_locSep.sh to start the training.

	# Training the A(Ground Category Emb) + S(DV3P) model
	-Uncomment the line of "CUDA_VISIBLE_DEVICES="0" python -u main_GCEmb_Sound.py $OPTS" in scripts/train_locSep.sh to start the training.


# Evaluation
	# Ajust accordingly based on the selected model
	./scripts/eval_locSep.sh


# Reference

[1] Zhao, Hang, et al. "The sound of pixels." Proceedings of the European conference on computer vision (ECCV). 2018.

[2] Arandjelovic, Relja, and Andrew Zisserman. "Objects that sound." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

## Citation
```bibtex   
    @misc{zhu2020separating,
    title={Separating Sounds from a Single Image},
    author={Lingyu Zhu and Esa Rahtu},
    year={2020},
    eprint={2007.07984},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

# Acknowledgement
This repo is developed based on [Sound-of-Pixels](https://github.com/hangzhaomit/Sound-of-Pixels).
