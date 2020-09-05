import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss_loc_sep_acc_metrics(path, history):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['err'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['err'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')


    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['err_loc'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['err_loc'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss_loc.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['err_sep'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['err_sep'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss_sep.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['acc'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['acc'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'acc.png'), dpi=200)
    plt.close('all')


    fig = plt.figure()
    plt.plot(history['val']['epoch'], history['val']['sdr'],
             color='r', label='SDR')
    plt.plot(history['val']['epoch'], history['val']['sir'],
             color='g', label='SIR')
    plt.plot(history['val']['epoch'], history['val']['sar'],
             color='b', label='SAR')
    plt.legend()
    fig.savefig(os.path.join(path, 'metrics.png'), dpi=200)
    plt.close('all')


def plot_loss_metrics(path, history):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['err'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['err'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['val']['epoch'], history['val']['sdr'],
             color='r', label='SDR')
    plt.plot(history['val']['epoch'], history['val']['sir'],
             color='g', label='SIR')
    plt.plot(history['val']['epoch'], history['val']['sar'],
             color='b', label='SAR')
    plt.legend()
    fig.savefig(os.path.join(path, 'metrics.png'), dpi=200)
    plt.close('all')
