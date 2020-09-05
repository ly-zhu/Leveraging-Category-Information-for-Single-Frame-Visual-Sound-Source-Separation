import os
import random
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
from mir_eval.separation import bss_eval_sources

from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs
from viz import plot_loss_loc_sep_acc_metrics
import matplotlib.pyplot as plt
import soundfile
import cv2


# Network wrapper, defines forward pass
class NetWrapper1(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper1, self).__init__()
        self.net_sound = nets

    def forward(self, mags, mag_mix, args):
        mag_mix = mag_mix + 1e-10

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)

        # calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                # for simplicity, mag_N > 0.5 * mag_mix
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)

        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # forward net_sound
        feat_sound = self.net_sound(log_mag_mix)
        feat_sound = activate(feat_sound, args.sound_activation)

        return feat_sound, \
            {'gt_masks': gt_masks, 'mag_mix': mag_mix, 'mags': mags, 'weight': weight}


class NetWrapper2(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper2, self).__init__()
        self.net_frame = nets

    def forward(self, frame, args):
        
        N = args.num_mix

        # return appearance features and appearance embedding
        feat_frames = [None for n in range(N)]
        emb_frames = [None for n in range(N)]
        for n in range(N):
            feat_frames[n], emb_frames[n] = self.net_frame.forward_multiframe_feat_emb(frame[n], pool=True)
            emb_frames[n] = activate(emb_frames[n], args.img_activation)
        
        return feat_frames, emb_frames


class NetWrapper3(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper3, self).__init__()
        self.net_avol = nets

    def forward(self, feat_frame, feat_sound, args):
        N = args.num_mix

        pred_mask = [None for n in range(N)]
        # appearance attention
        for n in range(N):
            pred_mask[n] = self.net_avol(feat_frame[n], feat_sound)
            pred_mask[n] = activate(pred_mask[n], args.output_activation)

        return pred_mask


# Calculate metrics
def calc_metrics(batch_data, pred_masks_, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()
    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']
    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]
    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)
    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)
        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]


# Visualize predictions
def output_visuals_PosNeg(vis_rows, batch_data, masks_pos,  masks_neg, idx_pos, idx_neg, pred_masks_, gt_masks_, mag_mix_, weight_, args):
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    frames = batch_data['frames']
    infos = batch_data['infos']

    # masks to cpu, numpy
    masks_pos = torch.squeeze(masks_pos, dim=1)
    masks_pos = masks_pos.cpu().float().numpy()
    masks_neg = torch.squeeze(masks_neg, dim=1)
    masks_neg = masks_neg.cpu().float().numpy()

    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    gt_masks_linear = [None for n in range(N)]

    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, gt_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]
            gt_masks_linear[n] = gt_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    idx_pos = int(idx_pos.detach().cpu().numpy())
    idx_neg = int(idx_neg.detach().cpu().numpy())
    for n in range(N):
        pred_masks_[n] = pred_masks_[n].detach().cpu().numpy()
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()
        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()
        # threshold if binary mask
        if args.binary_mask:
            pred_masks_[n] = (pred_masks_[n] > args.mask_thres).astype(np.float32)
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    threshold = 0.5
    # loop over each sample
    for j in range(B):
        row_elements = []
        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        matplotlib.image.imsave(os.path.join(args.vis, filename_mixmag), mix_amp[::-1, :, :])
        matplotlib.image.imsave(os.path.join(args.vis, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(args.vis, filename_mixwav), args.audRate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # GT and predicted audio recovery
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]
            gt_mag_ = mag_mix_[j, 0] * gt_masks_[n][j, 0]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            pred_mag_ = mag_mix_[j, 0] * pred_masks_[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

            # output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n+1))
            filename_predmask = os.path.join(prefix, 'predmask{}.jpg'.format(n+1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask = (np.clip(pred_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)

            matplotlib.image.imsave(os.path.join(args.vis, filename_gtmask), gt_mask[::-1, :])
            matplotlib.image.imsave(os.path.join(args.vis, filename_predmask), pred_mask[::-1, :])

            # ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag_)
            pred_mag = magnitude2heatmap(pred_mag_)

            matplotlib.image.imsave(os.path.join(args.vis, filename_gtmag), gt_mag[::-1, :, :])
            matplotlib.image.imsave(os.path.join(args.vis, filename_predmag), pred_mag[::-1, :, :])

            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n+1))
            wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_wav)
            wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, preds_wav[n])

        # save frame
        frames_tensor = recover_rgb(frames[idx_pos][j,:,int(args.num_frames//2)])
        frames_tensor = np.asarray(frames_tensor)
        filename_frame = os.path.join(prefix, 'frame{}.png'.format(idx_pos+1))
        matplotlib.image.imsave(os.path.join(args.vis, filename_frame), frames_tensor)
        frame = frames_tensor.copy()
        # get heatmap and overlay for postive pair
        height, width = masks_pos.shape[-2:]
        heatmap = np.zeros((height*16, width*16))
        for i in range(height):
            for k in range(width):
                mask_pos = masks_pos[j]
                value = mask_pos[i,k]
                value = 0 if value < threshold else value
                ii = i * 16
                jj = k * 16
                heatmap[ii:ii + 16, jj:jj + 16] = value
        heatmap = (heatmap * 255).astype(np.uint8)
        filename_heatmap = os.path.join(prefix, 'heatmap_{}_{}.jpg'.format(idx_pos+1, idx_pos+1))
        plt.imsave(os.path.join(args.vis, filename_heatmap), heatmap, cmap='hot')
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap, 0.5, frame, 0.5, 0, dtype = cv2.CV_32F)
        path_overlay = os.path.join(args.vis, prefix, 'overlay_{}_{}.jpg'.format(idx_pos+1, idx_pos+1))
        cv2.imwrite(path_overlay, fin)

        # save frame
        frames_tensor = recover_rgb(frames[idx_neg][j,:,int(args.num_frames//2)])
        frames_tensor = np.asarray(frames_tensor)
        filename_frame = os.path.join(prefix, 'frame{}.png'.format(idx_neg+1))
        matplotlib.image.imsave(os.path.join(args.vis, filename_frame), frames_tensor)
        frame = frames_tensor.copy()
        # get heatmap and overlay for postive pair
        height, width = masks_neg.shape[-2:]
        heatmap = np.zeros((height*16, width*16))
        for i in range(height):
            for k in range(width):
                mask_neg = masks_neg[j]
                value = mask_neg[i,k]
                value = 0 if value < threshold else value
                ii = i * 16
                jj = k * 16
                heatmap[ii:ii + 16, jj:jj + 16] = value
        heatmap = (heatmap * 255).astype(np.uint8)
        filename_heatmap = os.path.join(prefix, 'heatmap_{}_{}.jpg'.format(idx_pos+1, idx_neg+1))
        plt.imsave(os.path.join(args.vis, filename_heatmap), heatmap, cmap='hot')
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap, 0.5, frame, 0.5, 0, dtype = cv2.CV_32F)
        path_overlay = os.path.join(args.vis, prefix, 'overlay_{}_{}.jpg'.format(idx_pos+1, idx_neg+1))
        cv2.imwrite(path_overlay, fin)

        vis_rows.append(row_elements)


def evaluate(crit_loc, crit_sep, netWrapper1, netWrapper2, netWrapper3, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=False)

    # switch to eval mode
    netWrapper1.eval()
    netWrapper2.eval()
    netWrapper3.eval()

    # initialize meters
    loss_meter = AverageMeter()
    loss_acc_meter = AverageMeter()
    loss_sep_meter = AverageMeter()
    loss_loc_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()
    
    vis_rows = []
    for i, batch_data in enumerate(loader):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']

        N = args.num_mix
        B = mag_mix.shape[0]
        
        for n in range(N):
            frames[n] = torch.autograd.Variable(frames[n]).to(args.device)
            mags[n] = torch.autograd.Variable(mags[n]).to(args.device)
        mag_mix = torch.autograd.Variable(mag_mix).to(args.device)
            
        # forward pass
        # return feat_sound
        feat_sound, outputs = netWrapper1.forward(mags, mag_mix, args)
        gt_masks = outputs['gt_masks']
        mag_mix_ = outputs['mag_mix']
        weight_ = outputs['weight']
        
        # return feat_frame, and emb_frame
        feat_frame, emb_frame = netWrapper2.forward(frames, args)

        # random select positive/negative pairs
        idx_pos = torch.randint(0,N, (1,))
        idx_neg = N -1 -idx_pos

        # appearance attention
        masks = netWrapper3.forward(feat_frame, emb_frame[idx_pos], args)
        mask_pos = masks[idx_pos]
        mask_neg = masks[idx_neg]

        # max pooling
        pred_pos = F.adaptive_max_pool2d(mask_pos, 1)
        pred_pos = pred_pos.view(mask_pos.shape[0])
        pred_neg = F.adaptive_max_pool2d(mask_neg, 1)
        pred_neg = pred_neg.view(mask_neg.shape[0])

        # ground truth for the positive/negative pairs
        y1 = torch.ones(B,device=args.device).detach()
        y0 = torch.zeros(B, device=args.device).detach()

        # localization loss
        loss_loc_pos = crit_loc(pred_pos, y1).reshape(1)
        loss_loc_neg = crit_loc(pred_neg, y0).reshape(1)
        loss_loc = args.lamda * (loss_loc_pos + loss_loc_neg)/N

        # Calculate val accuracy
        pred_pos = (pred_pos > args.mask_thres)
        pred_neg = (pred_neg > args.mask_thres)
        valacc = 0
        for j in range(B):
            if pred_pos[j].item() == y1[j].item():
                valacc += 1.0
            if pred_neg[j].item() == y0[j].item():
                valacc += 1.0
        valacc = valacc/N/B

        # sepatate sounds
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]

        pred_masks = [None for n in range(N)]
        for n in range(N):
            feat_img = emb_frame[n]
            feat_img = feat_img.view(B, 1, C)
            pred_masks[n] = torch.bmm(feat_img, feat_sound.view(B, C, -1)) \
                .view(B, 1, *sound_size[2:])
            pred_masks[n] = activate(pred_masks[n], args.output_activation)

        # separatioon loss
        loss_sep = crit_sep(pred_masks, gt_masks, weight_).reshape(1)

        # total loss
        loss = loss_loc + loss_sep

        loss_meter.update(loss.item())
        loss_acc_meter.update(valacc)
        loss_sep_meter.update(loss_sep.item())
        loss_loc_meter.update(loss_loc.item())

        print('[Eval] iter {}, loss: {:.4f}, loss_loc: {:.4f}, loss_sep: {:.4f}, acc: {:.4f} '.format(i, loss.item(), loss_loc.item(), loss_sep.item(),  valacc))

        # calculate metrics
        sdr_mix, sdr, sir, sar = calc_metrics(batch_data, pred_masks, args)
        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)

        # output visualization
        if len(vis_rows) < args.num_vis:
            output_visuals_PosNeg(vis_rows, batch_data, mask_pos, mask_neg, idx_pos, idx_neg, pred_masks, gt_masks, mag_mix_, weight_, args)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, Loss_loc: {:.4f},  Loss_sep: {:.4f},  acc: {:.4f}, sdr_mix: {:.4f}, sdr: {:.4f}, sir: {:.4f}, sar: {:.4f}, '
           .format(epoch, loss_meter.average(), loss_loc_meter.average(), loss_sep_meter.average(), loss_acc_meter.average(), sdr_mix_meter.average(), sdr_meter.average(), sir_meter.average(), sar_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['err_loc'].append(loss_loc_meter.average())
    history['val']['err_sep'].append(loss_sep_meter.average())
    history['val']['acc'].append(loss_acc_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_loc_sep_acc_metrics(args.ckpt, history)
    print('this evaluation round is done!')


# train one epoch
def train(crit_loc, crit_sep, netWrapper1, netWrapper2, netWrapper3, loader, optimizer, history, epoch, args):
    print('Training at {} epochs...'.format(epoch))
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    netWrapper1.train()
    netWrapper2.train()
    netWrapper3.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']

        N = args.num_mix
        B = mag_mix.shape[0]
        for n in range(N):
            frames[n] = torch.autograd.Variable(frames[n]).to(args.device)
            mags[n] = torch.autograd.Variable(mags[n]).to(args.device)
        mag_mix = torch.autograd.Variable(mag_mix).to(args.device)

        # forward pass
        optimizer.zero_grad()
        # return feat_sound
        feat_sound, outputs = netWrapper1.forward(mags, mag_mix, args)
        gt_masks = outputs['gt_masks']
        mag_mix_ = outputs['mag_mix']
        weight_ = outputs['weight']

        # return feat_frame, and emb_frame
        feat_frame, emb_frame = netWrapper2.forward(frames, args)
        
        # random select positive/negative pairs
        idx_pos = torch.randint(0,N, (1,))
        idx_neg = N -1 -idx_pos
        # appearance attention
        masks = netWrapper3.forward(feat_frame, emb_frame[idx_pos], args)
        mask_pos = masks[idx_pos]
        mask_neg = masks[idx_neg]

        # max pooling
        pred_pos = F.adaptive_max_pool2d(mask_pos, 1)
        pred_pos = pred_pos.view(mask_pos.shape[0])
        pred_neg = F.adaptive_max_pool2d(mask_neg, 1)
        pred_neg = pred_neg.view(mask_neg.shape[0])

        # ground truth for the positive/negative pairs
        y1 = torch.ones(B,device=args.device).detach()
        y0 = torch.zeros(B, device=args.device).detach()

        # localization loss and acc
        loss_loc_pos = crit_loc(pred_pos, y1).reshape(1)
        loss_loc_neg = crit_loc(pred_neg, y0).reshape(1)
        loss_loc = args.lamda * (loss_loc_pos + loss_loc_neg)/N
        pred_pos = (pred_pos > args.mask_thres)
        pred_neg = (pred_neg > args.mask_thres)
        valacc = 0
        for j in range(B):
            if pred_pos[j].item() == y1[j].item():
                valacc += 1.0
            if pred_neg[j].item() == y0[j].item():
                valacc += 1.0
        valacc = valacc/N/B

        # sepatate sounds (for simplicity, we don't use the alpha and beta)
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]
        pred_masks = [None for n in range(N)]
        for n in range(N):
            feat_img = emb_frame[n]
            feat_img = feat_img.view(B, 1, C)
            pred_masks[n] = torch.bmm(feat_img, feat_sound.view(B, C, -1)) \
                .view(B, 1, *sound_size[2:])
            pred_masks[n] = activate(pred_masks[n], args.output_activation)

        # separation loss
        loss_sep = crit_sep(pred_masks, gt_masks, weight_).reshape(1)
        
        # total loss
        loss = loss_loc + loss_sep
        loss.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, lr_avol: {}, '
                  'loss: {:.5f}, loss_loc: {:.5f}, loss_sep: {:.5f}, acc: {:.5f} '
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, args.lr_avol,
                          loss.item(), loss_loc.item(), loss_sep.item(), 
                          valacc))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(loss.item())
            history['train']['err_loc'].append(loss_loc.item())
            history['train']['err_sep'].append(loss_sep.item())
            history['train']['acc'].append(valacc)


def checkpoint(net_sound, net_frame, net_avol, optimizer, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    state = {'epoch': epoch, \
             'state_dict_net_sound': net_sound.state_dict(), \
             'state_dict_net_frame': net_frame.state_dict(),\
             'state_dict_net_avol': net_avol.state_dict(),\
             'optimizer': optimizer.state_dict(), \
             'history': history, }

    torch.save(state, '{}/checkpoint_{}'.format(args.ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    if cur_err <= args.best_err:
        args.best_err = cur_err
        torch.save(state, '{}/checkpoint_{}'.format(args.ckpt, suffix_best))


def load_checkpoint(net_sound, net_frame, net_avol, optimizer, history, filename):
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        net_sound.load_state_dict(checkpoint['state_dict_net_sound'])
        net_frame.load_state_dict(checkpoint['state_dict_net_frame'])
        net_avol.load_state_dict(checkpoint['state_dict_net_avol'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
                    
        history = checkpoint['history']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return net_sound, net_frame, net_avol, optimizer, start_epoch, history


def load_checkpoint_from_train(net_sound, net_frame, net_avol, filename):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        print('epoch: ', checkpoint['epoch'])
        net_sound.load_state_dict(checkpoint['state_dict_net_sound'])
        net_frame.load_state_dict(checkpoint['state_dict_net_frame'])
        net_avol.load_state_dict(checkpoint['state_dict_net_avol'])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return net_sound, net_frame, net_avol


def load_sep(net_sound, net_frame, filename):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        print('epoch: ', checkpoint['epoch'])
        net_sound.load_state_dict(checkpoint['state_dict_net_sound'])
        net_frame.load_state_dict(checkpoint['state_dict_net_frame'])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return net_sound, net_frame


def create_optimizer(net_sound, net_frame, net_avol, args):
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_frame},
                    {'params': net_frame.parameters(), 'lr': args.lr_sound},
                    {'params': net_avol.parameters(), 'lr': args.lr_avol}]
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_avol *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        input_channel=1,
        output_channel=args.num_channels,
        fc_dim=args.num_channels,
        weights=args.weights_sound)
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame)
    net_avol = builder.build_avol(
        arch=args.arch_avol,
        fc_dim=args.num_channels,
        weights=args.weights_frame)

    crit_loc = nn.BCELoss()
    crit_sep = builder.build_criterion(arch=args.loss)

    # Dataset and Loader
    dataset_train = MUSICMixDataset(
        args.list_train, args, split='train')
    dataset_val = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split='val')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False)
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # Set up optimizer
    optimizer = create_optimizer(net_sound, net_frame, net_avol, args)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': [], 'err_loc': [], 'err_sep': [], 'acc': []},
        'val': {'epoch': [], 'err': [],  'err_loc': [], 'err_sep': [], 'acc': [], 'sdr': [], 'sir': [], 'sar': []}}


    # Training loop
    # Load from pretrained models
    start_epoch = 1
    model_name = args.ckpt + '/checkpoint.pth'
    if os.path.exists(model_name):
        if args.mode == 'eval':
            net_sound, net_frame, net_avol = load_checkpoint_from_train(net_sound, net_frame, net_avol, model_name)
        elif args.mode == 'train':
            model_name = args.ckpt + '/checkpoint_latest.pth'
            net_sound, net_frame, net_avol, optimizer, start_epoch, history = load_checkpoint(net_sound, net_frame, net_avol, optimizer, history, model_name)
            print("Loading from previous checkpoint.")
    
    else:
        if args.mode == 'train' and start_epoch==1 and os.path.exists(args.weights_model):
            net_sound, net_frame = load_sep(net_sound, net_frame, args.weights_model)
            print("Loading from appearance + sound checkpoint.")
    
    # Wrap networks
    netWrapper1 = NetWrapper1(net_sound)
    netWrapper1 = torch.nn.DataParallel(netWrapper1, device_ids=range(args.num_gpus)).cuda()
    netWrapper1.to(args.device)

    netWrapper2 = NetWrapper2(net_frame)
    netWrapper2 = torch.nn.DataParallel(netWrapper2, device_ids=range(args.num_gpus)).cuda()
    netWrapper2.to(args.device)

    netWrapper3 = NetWrapper3(net_avol)
    netWrapper3 = torch.nn.DataParallel(netWrapper3, device_ids=range(args.num_gpus)).cuda()
    netWrapper3.to(args.device)


    # Eval mode
    #evaluate(crit_loc, crit_sep, netWrapper1, netWrapper2, netWrapper3, loader_val, history, 0, args)
    if args.mode == 'eval':
        evaluate(crit_loc, crit_sep, netWrapper1, netWrapper2, netWrapper3, loader_val, history, 0, args)
        print('Evaluation Done!')
        return

        
    for epoch in range(start_epoch, args.num_epoch + 1):    
        train(crit_loc, crit_sep, netWrapper1, netWrapper2, netWrapper3, loader_train, optimizer, history, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

        ## Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(crit_loc, crit_sep, netWrapper1, netWrapper2, netWrapper3, loader_val, history, epoch, args)
            # checkpointing
            checkpoint(net_sound, net_frame, net_avol, optimizer, history, epoch, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}-{}'.format(
            args.arch_frame, args.arch_sound, args.arch_avol)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        if args.binary_mask:
            assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    if args.mode == 'train':
        args.weights_model = 'ckpt_res50_DV3P_MUSIC_N2_f1_binary_bs10_TrainS335_D65_ValValS100_ValTestS130_dup100_f8fps_11k/MUSIC-2mix-LogFreq-resnet18dilated_50-deeplabV3Plus_mobilenetv2-frames1stride24-maxpool-binary-weightedLoss-channels11-epoch100-step40_80/checkpoint.pth'
        args.vis = os.path.join(args.ckpt, 'visualization_train/')
        makedirs(args.ckpt, remove=False)
    elif args.mode == 'eval':
        args.vis = os.path.join(args.ckpt, 'visualization_val/')
    elif args.mode == 'test':
        args.vis = os.path.join(args.ckpt, 'visualization_test/')

    # initialize best error with a big number
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
