# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import h5py
import tensorflow as tf
import numpy as np
import itertools
import pretty_midi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import moviepy.editor as mpy
import pypianoroll
import glob
import music21
from IPython.display import HTML, Video
from IPython import display
import seaborn as sns
plt.ion()

""" Each dataset has different programs for its tracks. The Bach dataset has 
4 tracks of pianos, so programs=[0, 0, 0, 0]. When there is a drum track, set
its program to 0 and is_drum to True. """

# --- local samples------------------------------------------------------------------

def load_or_create_sample(filename):    
    """Load the samples used for evaluation."""
    
    # Here we select n_sample = 10 for the evaluation
    n_sample = 10
    
    data = np.load(filename)
    data = np.asarray(data, dtype=np.float32) # {-1, 1}

    random_idx = np.random.choice(len(data), n_sample, replace=False)
    
    sample_x = data[random_idx]

    sample_z = tf.random.truncated_normal((n_sample, 2, 8, 512))

    return sample_x, sample_z


    
# --- Metrics ------------------------------------------------------------------

def compute_metrics(pianoroll) :
    pianoroll = pianoroll.reshape((-1,2,16,128,4))
    def empty_bar_rate(tensor):
        """Return the ratio of empty bars to the total number of bars."""
        if len(tensor.shape) != 5:
            raise ValueError("Input tensor must have 5 dimensions.")
        return np.mean(
            np.any(tensor > 0.5, (2, 3)).astype(np.float32), (0, 1))

    def n_pitches_used(tensor):
        """Return the number of unique pitches used per bar."""
        if len(tensor.shape) != 5:
            raise ValueError("Input tensor must have 5 dimensions.")
        pitch_hist = np.mean(np.sum(tensor, 2), (0, 1))
        return np.linalg.norm(np.ones(pitch_hist.shape)-pitch_hist, axis=0) #Sums across each timestep in bar
    def polyphonic_rate(tensor, threshold=2):
        """Return the ratio of the number of time steps where the number of pitches
        being played is larger than `threshold` to the total number of time steps"""
        if len(tensor.shape) != 5:
            raise ValueError("Input tensor must have 5 dimensions.")
        n_poly = np.count_nonzero((np.count_nonzero(tensor, 3) > threshold), 2)
        return np.mean((n_poly / tensor.shape[2]), (0, 1))
    def in_scale_rate(tensor):
        """Return the in_scale_rate metric value."""
        if len(tensor.shape) != 5:
            raise ValueError("Input tensor must have 5 dimensions.")
        if tensor.shape[3] != 12:
            raise ValueError("Input tensor must be a chroma tensor.")

        def _scale_mask(key=3):
            """Return a scale mask for the given key. Default to C major scale."""
            a_scale_mask = np.array([[[1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]]], bool)
            return np.expand_dims(np.roll(a_scale_mask, -key, 2), -1)

        scale_mask = _scale_mask().astype(np.float32)
        in_scale = np.sum(scale_mask * np.sum(tensor, 2), (0, 1, 2))
        return in_scale / np.sum(tensor, (0, 1, 2, 3))
    
    def to_chroma(pianoroll):
        """Return the chroma features (not normalized)."""
        if len(pianoroll.shape) != 5:
            raise ValueError("Input tensor must have 5 dimensions.")
        remainder = pianoroll.shape[3] % 12
        if remainder:
            pianoroll = np.pad(
                pianoroll, ((0, 0), (0, 0), (0, 0), (0, 12 - remainder), (0, 0)))
        reshaped = np.reshape(
            pianoroll, (-1, pianoroll.shape[1], pianoroll.shape[2], 12,
                        pianoroll.shape[3] // 12 + int(remainder > 0),
                        pianoroll.shape[4]))
        return np.sum(reshaped, 4)
    pianoroll = pianoroll[:,:,:,24:108,:]
    chroma = to_chroma(pianoroll)
    metrics = {'empty_bar_rate':empty_bar_rate(pianoroll), 
               'pitch_histogram_distance' : n_pitches_used(pianoroll), 
               'n_pitch_classes_used': n_pitches_used(chroma),
               'polyphonic_rate':polyphonic_rate(pianoroll),
               'in_scale_ratio': in_scale_rate(chroma)
                }
    return metrics
    


def plot_metrics(metrics,train_vals,num_instr, Metrics = ['empty_bar_rate', 'pitch_histogram_distance','in_scale_ratio'] ):
    
    sns.set()
    fig, axs = plt.subplots(3,num_instr, sharex=True, figsize=(60, 30))
    plt.xscale('log')
    [(axs[metric_num][i].plot(metrics['iters'], np.ones(len(metrics[metric_name]))*train_vals[metric_name][i],'r',linewidth=10,alpha=0.7),
       axs[metric_num][i].tick_params(axis='both', which='major', labelsize=30),
        axs[metric_num][i].tick_params(axis='both', which='minor', labelsize=30))     
     for i in range(num_instr) 
     for metric_num,metric_name in enumerate(Metrics)]
    
    [ (axs[metric_num][i].scatter(metrics['iters'], [temp[i] for temp in metrics[metric_name]],linewidth=10),
      axs[metric_num][i].tick_params(axis='both', which='major', labelsize=30)
        ,axs[metric_num][i].tick_params(axis='both', which='minor', labelsize=30))
     for i in range(num_instr) 
     for metric_num,metric_name in enumerate(Metrics) ]
    
    
    
    [ axs[i][0].set_ylabel(ylabel=Metrics[i],fontsize=40) for i in range(len(Metrics))] 
    [ axs[2][i].set_xlabel(xlabel= "Iterations"+"("+pretty_midi.program_to_instrument_class(i)+")",fontsize=40) for i in range(num_instr)] 
    
    #fig.suptitle('Training History', fontsize=40)
    
#     fig.text(0.5,-0.02, "Iterations", ha="center", va="center",fontsize=70)
    fig.tight_layout()
    plt.show()   


    

# --- Training ------------------------------------------------------------------

def generate_midi(pianoroll,
                  programs=[0, 0, 0, 0],
                  is_drums=[False, False, False, False],
                  tempo=100,           # in bpm
                  beat_resolution=4,  # number of time steps
                  saveto_dir=None, iteration=-1
                  ):

    pianoroll = pianoroll > 0  # 4, 32, 128, 4
    
#     pianoroll = pianoroll[0]

    # TODO: MuseGan concatenates all samples in the batch along time axis and
    # then convert to a single midi file. But different samples may come
    # from different segments of different midi files. It does not make much
    # sense to concatenate them together. We may want to only convert each
    # sample in the batch to a midi file. In order to generate longer midi
    # files, we should increase the number of bars and/or the length of each
    # bar (e.g., from current 16 to 48 used in MuseGAN)

    # Reshape batched pianoroll array to a single pianoroll array
    pianoroll_ = pianoroll.reshape((-1, pianoroll.shape[2], pianoroll.shape[3]))  # 32*B, 128, 4

    # Create the tracks
    tracks = []
    for idx in range(pianoroll_.shape[2]):
        tracks.append(pypianoroll.Track(
            pianoroll_[..., idx], programs[idx], is_drums[idx]))

    multitrack = pypianoroll.Multitrack(
        tracks=tracks, tempo=tempo, beat_resolution=beat_resolution)

    if iteration>-1:
        file = os.path.join(saveto_dir, 'iteration-{}.mid'.format(iteration))
        multitrack.write(file)
    else: 
        mid_files = glob.glob(os.path.join(saveto_dir, 'test-*.mid'))
        file = os.path.join(saveto_dir, 'test-%d.mid' % len(mid_files))
        multitrack.write(file)
    print('save to ', file)
    return file


# --- Inference ------------------------------------------------------------------


def get_sample_c(midi=None, root_dir=None, phrase_length=32, beat_resolution=4):
    
    if not isinstance(midi, str):
        # ----------- Generation from preprocessed dataset ------------------
        sample_x = midi
        sample_c = np.expand_dims(sample_x[..., 0], -1)  # 4, 32, 128, 1
    else:
        # --------------- Generation from midi file -------------------------
        midi_file = midi
        parsed = pypianoroll.Multitrack(beat_resolution=beat_resolution)
        parsed.parse_midi(midi_file)

        programs = [track.program for track in parsed.tracks]
        programs = [0,0,0,0]
        is_drums = [track.is_drum for track in parsed.tracks]

        # Extract condition track
        assert programs[0] == 0, 'The first track should be the piano track !!'

        sample_c = parsed.tracks[0].pianoroll.astype(np.float32)
        # Remove initial steps that have no note-on events
        first_non_zero = np.nonzero(sample_c.sum(axis=1))[0][0]
        # Use the first 'phrase_length' steps as the primer
        sample_c = sample_c[first_non_zero: first_non_zero + phrase_length]

        # Binarize data (ignore velocity value)
        sample_c[sample_c > 0] = 1
        sample_c[sample_c <= 0] = -1

        sample_c = np.expand_dims(np.expand_dims(sample_c, 0), -1)  # 1, 32, 128, 1

    return sample_c
                
def evaluate_midi(generator, root_dir, eval_dir, midi_file='./Experiments/data/happy_birthday_easy.mid'):
        
    # Noise
    sample_c = get_sample_c(midi=midi_file, root_dir=root_dir)
    sample_c = tf.convert_to_tensor(sample_c, dtype=tf.float32)
    programs = [0, 0, 0, 0]
    is_drums = [False, False, False, False]
    sample_z = tf.random.truncated_normal((1, 2, 8, 512))
    fake_sample_x = generator((sample_c, sample_z), training=False)
    return generate_midi(fake_sample_x.numpy(), saveto_dir=eval_dir, programs=programs,
              is_drums=is_drums)
    
def evaluate_pianoroll(generator, root_dir, eval_dir, midi_file='./Experiments/data/happy_birthday_easy.mid', n_pr = 4):
     # Noise
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    sample_c = get_sample_c(midi=midi_file, root_dir=root_dir)
    sample_c = np.repeat(sample_c, n_pr, axis=0)
    sample_c = tf.convert_to_tensor(sample_c, dtype=tf.float32)  
    for i in range(n_pr):
        sample_z = tf.random.truncated_normal((n_pr, 2, 8, 512))
        fake_sample_x = generator((sample_c, sample_z), training=False)
        show_pianoroll(fake_sample_x)
        
# def evaluate_pianoroll(generator, root_dir, eval_dir, midi_file='./Experiments/data/happy_birthday_easy.mid'):
#  # Noise
#     sample_c = get_sample_c(midi=midi_file, root_dir=root_dir)
#     sample_c = tf.convert_to_tensor(sample_c, dtype=tf.float32)   
#     fig = plt.figure(figsize=(15, 8))
#     n_pr = 5 
#     fig, axs = plt.subplots(n_pr)
#     for i in range(n_pr):
#         sample_z = tf.random.truncated_normal((1, 2, 8, 512))
#         fake_sample_x = generator((sample_c, sample_z), training=False)
#         plotpianoroll(fake_sample_x.numpy(), ax=axs[i])
    


        
    
# --- plot------------------------------------------------------------------


def plot_loss_logs(G_loss, D_loss, figsize=(15, 5), smoothing=0.001):
    """Utility for plotting losses with smoothing."""
    #TODO
    #G_loss = vutils.smooth_data(G_loss, amount=smoothing)
    #D_loss = vutils.smooth_data(D_loss, amount=smoothing)
    sns.set()
    plt.figure(figsize=figsize)
    plt.plot(D_loss, label='C_loss')
    plt.plot(G_loss, label='G_loss')
    plt.legend(loc='lower right', fontsize='medium')
    plt.xlabel('Iteration', fontsize='x-large')
    plt.ylabel('Losses', fontsize='x-large')
    plt.title('Training History', fontsize='xx-large')
    
    
def playmidi(filename):
    mf = music21.midi.MidiFile()
    mf.open(filename)
    mf.read()
    mf.close()
    s = music21.midi.translate.midiFileToStream(mf)
    s.show('midi')
    
def show_pianoroll(xs, min_pitch=45, max_pitch=85,
                   programs = [0, 0, 0, 0], save_dir=None):
    """ Plot a MultiTrack PianoRoll

    :param x: Multi Instrument PianoRoll Tensor
    :param min_pitch: Min Pitch / Min Y across all instruments.
    :param max_pitch: Max Pitch / Max Y across all instruments.
    :param programs: Program Number of the Tracks.
    :param file_name: Optional. File Name to save the plot.
    :return:
    """

    # Convert fake_x to numpy and convert -1 to 0
    xs = xs > 0

    channel_last = lambda x: np.moveaxis(np.array(x), 2, 0)
    xs = [channel_last(x) for x in xs]

    assert len(xs[0].shape) == 3, 'Pianoroll shape must have 3 dims, Got %d' % len(xs[0].shape)
    n_tracks, time_step, _ = xs[0].shape

    fig = plt.figure(figsize=(15, 4))
    
    x = xs[0]
    
    for j in range(4):
        b = j+1
        ax = fig.add_subplot(1,4,b)
        nz = np.nonzero(x[b-1])

        if programs:
            ax.set_xlabel('Time('+pretty_midi.program_to_instrument_class(programs[j%4])+')')

        if (j+1)== 1:
            ax.set_ylabel('Pitch')
        else:
            ax.set_yticks([])

        ax.scatter(nz[0], nz[1], s=np.pi * 3, color='bgrcmk'[b-1])
        ax.set_ylim(45, 85)
        ax.set_xlim(0, time_step)
        fig.add_subplot(ax)
        

def plot_pianoroll(iteration, xs, fake_xs, min_pitch=45, max_pitch=85,
                   programs = [0, 0, 0, 0], save_dir=None):
    """ Plot a MultiTrack PianoRoll

    :param x: Multi Instrument PianoRoll Tensor
    :param min_pitch: Min Pitch / Min Y across all instruments.
    :param max_pitch: Max Pitch / Max Y across all instruments.
    :param programs: Program Number of the Tracks.
    :param file_name: Optional. File Name to save the plot.
    :return:
    """

    # Convert fake_x to numpy and convert -1 to 0
    xs = xs > 0
    fake_xs = fake_xs > 0

    channel_last = lambda x: np.moveaxis(np.array(x), 2, 0)
    xs = [channel_last(x) for x in xs]
    fake_xs = [channel_last(fake_x) for fake_x in fake_xs]

    assert len(xs[0].shape) == 3, 'Pianoroll shape must have 3 dims, Got %d' % len(xs[0].shape)
    n_tracks, time_step, _ = xs[0].shape

    fig = plt.figure(figsize=(15, 8))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.2)

    for i in range(4):
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, n_tracks,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)

        x, fake_x = xs[i], fake_xs[i]

        for j, (a, b) in enumerate(itertools.product([1, 2], [1, 2, 3, 4])):

            ax = fig.add_subplot(inner_grid[j])

            if a == 1:
                nz = np.nonzero(x[b-1])
            else:
                nz = np.nonzero(fake_x[b-1])

            if programs:
                ax.set_xlabel('Time('+pretty_midi.program_to_instrument_class(programs[j%4])+')')

            if b == 1:
                ax.set_ylabel('Pitch')
            else:
                ax.set_yticks([])

            ax.scatter(nz[0], nz[1], s=np.pi * 3, color='bgrcmk'[b-1])
            ax.set_ylim(45, 85)
            ax.set_xlim(0, time_step)
            fig.add_subplot(ax)

    if isinstance(iteration, int):
        plt.suptitle('iteration: {}'.format(iteration), fontsize=20)
        filename = os.path.join(save_dir, 'sample_iteration_%05d.png' % iteration)
    else:
        plt.suptitle('Inference', fontsize=20)
        filename = os.path.join(save_dir, 'sample_inference.png')
    plt.savefig(filename)
    plt.close(fig)

def display_loss(iteration, d_losses, g_losses):
    sns.set()
    display.clear_output(wait=True)
    fig = plt.figure(figsize=(15,5))
    line1, = plt.plot(range(iteration+1), d_losses, 'r')
    line2, = plt.plot(range(iteration+1), g_losses, 'k')
    plt.xlabel('Iterations')
    plt.ylabel('Losses')
    plt.legend((line1, line2), ('C-loss', 'G-loss'))
    
    return display.display(fig)


def make_training_video(folder_dir):
    files = sorted([os.path.join(folder_dir, f) for f in os.listdir(folder_dir) if f.endswith('.png')])
    frames = [mpy.ImageClip(f).set_duration(1) for f in files]  
    clip = mpy.concatenate_videoclips(frames, method="compose")
    clip.write_videofile("movie.mp4",fps=15) 
    return Video("movie.mp4")
    