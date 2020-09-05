import os
import random
from .base import BaseDataset
import numpy as np
import sys
import pdb

label = {}
label['accordion'] = 1
label['acoustic_guitar'] = 2
label['cello'] = 3
label['clarinet'] = 4
label['erhu'] = 5
label['flute'] = 6
label['saxophone'] = 7
label['trumpet'] = 8
label['tuba'] = 9
label['violin'] = 10
label['xylophone'] = 11


label['cello-acoustic_guitar'] = 12
label['xylophone-flute'] = 13
label['acoustic_guitar-violin'] = 14
label['clarinet-acoustic_guitar'] = 15
label['saxophone-acoustic_guitar'] = 16
label['flute-violin'] = 17
label['xylophone-acoustic_guitar'] = 18
label['trumpet-tuba'] = 19
label['flute-trumpet'] = 20


vemb = {}
vemb['accordion'] = [1,0,0,0,0,0,0,0,0,0,0]
vemb['acoustic_guitar'] = [0,1,0,0,0,0,0,0,0,0,0]
vemb['cello'] = [0,0,1,0,0,0,0,0,0,0,0]
vemb['clarinet'] = [0,0,0,1,0,0,0,0,0,0,0]
vemb['erhu'] = [0,0,0,0,1,0,0,0,0,0,0]
vemb['flute'] = [0,0,0,0,0,1,0,0,0,0,0]
vemb['saxophone'] = [0,0,0,0,0,0,1,0,0,0,0]
vemb['trumpet'] = [0,0,0,0,0,0,0,1,0,0,0]
vemb['tuba'] = [0,0,0,0,0,0,0,0,1,0,0]
vemb['violin'] = [0,0,0,0,0,0,0,0,0,1,0]
vemb['xylophone'] = [0,0,0,0,0,0,0,0,0,0,1]

vemb['cello-acoustic_guitar'] = [0,1,1,0,0,0,0,0,0,0,0]
vemb['xylophone-flute'] = [0,0,0,0,0,1,0,0,0,0,1]
vemb['acoustic_guitar-violin'] = [0,1,0,0,0,0,0,0,0,1,0]
vemb['clarinet-acoustic_guitar'] = [0,1,0,1,0,0,0,0,0,0,0]
vemb['saxophone-acoustic_guitar'] = [0,1,0,0,0,0,1,0,0,0,0]
vemb['flute-violin'] = [0,0,0,0,0,1,0,0,0,1,0]
vemb['xylophone-acoustic_guitar'] = [0,1,0,0,0,0,0,0,0,0,1]
vemb['trumpet-tuba'] = [0,0,0,0,0,0,0,1,1,0,0]
vemb['flute-trumpet'] = [0,0,0,0,0,1,0,1,0,0,0]


class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.dataset = opt.dataset

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]
        frame_emb = [None for n in range(N)]
        classes = [None for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index]
        path_audioN = infos[0][0]
        music_category = path_audioN.split('/')[1]
        frame_emb[0] = np.array(vemb[music_category])
        classes[0] = label[music_category]
        music_lib = []
        type_1st = 0
        type_2nd = 0
        if '-' in music_category: # if it is duet video
            type_1st = 1
            music_a = music_category.split('-')[0]
            music_b = music_category.split('-')[1]
            music_lib.append(music_a)
            music_lib.append(music_b)
        else:
            type_1st = 0
            music_lib.append(music_category)

        second_music_category = music_category
        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            second_video_idx = random.randint(0, len(self.list_sample)-1)
            second_music_category = self.list_sample[second_video_idx][0].split('/')[1]
            if '-' in second_music_category:
                type_2st = 1
                music_2a = second_music_category.split('-')[0]
                music_2b = second_music_category.split('-')[1]
                # to load videos from different categories
                while second_music_category in music_lib or music_2a in music_lib or music_2b in music_lib:
                    music_2a = ''
                    music_2b = ''
                    second_video_idx = random.randint(0, len(self.list_sample)-1)
                    second_music_category = self.list_sample[second_video_idx][0].split('/')[1]
                    if '-' in second_music_category:
                        type_2st = 1
                        music_2a = second_music_category.split('-')[0]
                        music_2b = second_music_category.split('-')[1]
                    else:
                        type_2st = 0
            else:
                type_2st = 0
                music_2a = ''#second_music_category.split('-')[0]
                music_2b = ''#second_music_category.split('-')[1]
                while second_music_category in music_lib or music_2a in music_lib or music_2b in music_lib:
                    music_2a = ''
                    music_2b = ''
                    second_video_idx = random.randint(0, len(self.list_sample)-1)
                    second_music_category = self.list_sample[second_video_idx][0].split('/')[1]
                    if '-' in second_music_category:
                        type_2st = 1
                        music_2a = second_music_category.split('-')[0]
                        music_2b = second_music_category.split('-')[1]
                    else:
                        type_2st = 0

            if type_2st==0:
                music_lib.append(second_music_category)
            else:
                music_lib.append(music_2a)
                music_lib.append(music_2b)
            frame_emb[n] = np.array(vemb[second_music_category])
            classes[n] = label[second_music_category]
            infos[n] = self.list_sample[second_video_idx]
        
        if music_category == second_music_category:
            match = 1
        else: 
            match = 0
            
        # select frames
        idx_margin = max(
            int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                center_frameN = random.randint(
                    idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            data_path = '../../dataset/' + self.dataset + '/'
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(data_path + path_frameN+'/{:06d}.jpg'.format(center_frameN + idx_offset))
            path_audios[n] = data_path + path_audioN
        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)
        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags, 'frame_emb': frame_emb, 'classes': classes, 'match': match}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos
        return ret_dict
