#%matplotlib notebook
#import IPython.display as ipd

from datetime import datetime
import numpy as np
import scipy.signal
import math
import sys
from toposort import toposort, toposort_flatten


import matplotlib.pyplot as plt
import librosa.display
import sounddevice as sd


# just for debug purpose
np.set_printoptions(threshold=sys.maxsize)
np.seterr(all='raise')

#### helpers
# pan in (-60, 60)
# Based on DAFX chapter "SPATIAL EFFECTS", p144
# Assume loudspeaker is place in front of the listener, 60 fov.
def panning(x, pan):
    theta0 = math.pi / 6
    if len(x.shape) == 1:
        # mono -> stereo
        l, r = x, x
    else:
        l, r = x[0], x[1]
    p = pan / 180 * math.pi
    a = (math.cos(p)+math.sin(p)*math.tan(theta0)) / (math.cos(p)-math.sin(p)*math.tan(theta0))
    l_out = l * math.sqrt(1 / (1 + a*a))
    r_out = r * math.sqrt(1 / (1 + a*a)) * a
    return np.array([l_out, r_out])
        
#### Generators
def sine(A, pan):
    def _sine(sr, f, t):
        return panning(A * np.exp(-1j * 2 * np.pi * f * t), pan)
    return _sine

# General Saw wave
# width = 1 is rising sawtooth
# width = 0.5 is triangle
# width = 0 is falling sawtooth
# pan is in [-1, 1], -1 for left, 1 for right
def saw(A, width=1, pan=0):
    def _saw(sr, f, t):
        real  = scipy.signal.sawtooth(2 * np.pi * f * t, width=width)
        im  = scipy.signal.sawtooth(2 * np.pi * f * t + np.pi / 2, width=width)
        y = A * (real + 1j * im)
        return panning(y, pan)
    return _saw

def noise(A, pan):
    def _f(sr, f, t):
        a = math.ceil(sr / f)
        n = t.shape[-1]
        y = np.random.random(n + a)
        return panning(y[:n-a] + 1j * y[a:], pan)
    return _f

def sampler(A, file_path):
    rate, x = scipy.io.wavfile.read(file_path)
    x = x.T
    if x.dtype == np.int16:
        x = x.astype(float) / 2**15
    
    def _f(sr, f, t):
        assert(rate==sr)
        n = t.shape[-1]
        if n > x.shape[-1]:
            return np.append(x, np.zeros((2, n - x.shape[-1])))
        else:
            return x[:, :n]
    return _f


#### Filters

def pass_thru():
    return (lambda sr, x: x)

# Simple delay line.
# y[n] = x[n] + decay * y[n-d]
# d is in seconds
def delay(d, decay):
    def _delay(sr, x):
        y = np.full_like(x, 0)
        delay_count = int(d * sr)
        for i in range(x.shape[1]):
            if i - delay_count < 0:
                delay_y = 0
            else:
                delay_y = y[:, i-delay_count]
            y[:, i] = x[:, i] + decay * delay_y
        return y
    return _delay

# Variable-delay-value delay line.
# @delay_func: delay_func(i) gives the delay value at sample point `i`
def vdelay(delay_func, decay_func):
    def _f(sr, x):
        y = np.full_like(x, 0)
        for i in range(x.shape[1]):
            delay_count = int(delay_func(i)*sr)
            y[:, i] = x[:, i] + decay_func(i) * y[:, i-delay_count] if (i-delay_count) >= 0 else 0
        return y
    return _f

# IIR Filter
# @btype: one of ['lowpass', 'highpass', 'bandpass', 'bandstop']
# @Wn: 
# @bw: bandwidth, unit in sr/2 = 1
def iirfilter(btype, wpass, wstop, gpass=3, gstop=35):
    N, Wn = scipy.signal.buttord(wpass, wstop, gpass, gstop, analog=False)
    def _f(sr, x):
        b, a = scipy.signal.butter(N, Wn, btype, analog=False)
        ret = scipy.signal.filtfilt(b, a, x).astype('complex128')
        return ret
    return _f

## Modulators
def ring_modulator(f_c, carrier_func=np.sin, phi0=0):
    def _f(sr, x):
        n = x.shape[-1]
        return carrier_func(2*np.pi * f_c/sr * np.arange(n) + phi0) * x.real + \
            1j * carrier_func(2*np.pi * f_c/sr * np.arange(n) + phi0 + np.pi/2) * x.imag
    return _f

def amplitude_modulator(f_c, alpha, carrier_func=np.sin, phi0=0):
    def _f(sr, x):
        n = x.shape[-1]
        return (1 + alpha * carrier_func(2*np.pi * f_c/sr * np.arange(n) + phi0)) * x.real + \
            (1 + alpha * carrier_func(2*np.pi * f_c/sr * np.arange(n) + phi0 + np.pi/2)) * x.imag
    return _f

def phase_modulator(f_c, A=1, k=1):
    f = lambda sr, n, x: A * np.cos(2*np.pi* f_c/sr * np.arange(n) + k * x.real)
    def _f(sr, x):
        n = x.shape[-1]
        return f(sr, n, x.real) + 1j * f(sr, n, x.imag)
    return _f

def frequency_modulator(f_c, A=1, k=1):
    def _f(sr, x):
        n = x.shape[-1]
        sum_x = np.full_like(x, 0)
        for i in range(n):
            sum_x[:, i] = np.sum(x[:, i])
        f = lambda data: A * np.cos(2*np.pi* f_c/sr * np.arange(n) + 2*np.pi * k * data)
        return f(sum_x.real) + 1j*f(sum_x.imag) 
    return _f

def ssb_modulator(f_c, carrier_func=np.cos):
    def _f(sr, x):
        n = x.shape[-1]
        return carrier_func(2*np.pi * f_c/sr * np.arange(n)) * x.real - \
             np.sign(f_c) * carrier_func(2*np.pi * f_c/sr * np.arange(n) + np.pi/2) * x.imag
    return _f

#### Dynamic Range Control


# DAFX: p110
def limiter(threshold_db, attack_time, release_time, delay_time, plot=False):
    def _f(sr, x):
        threshold = 10 ** (threshold_db/10)
        at = 1 - math.exp(-2.2/(attack_time*sr))
        rt = 1 - math.exp(-2.2/(release_time*sr))
        n = x.shape[-1]
        delay_n = round(delay_time*sr)
        def calculate(x_in):
            xpeak = np.array([0, 0])
            gain = np.array([1, 1])
            y = np.full_like(x_in, 0)
            abs_xn = np.abs(x_in)
            gains = np.full_like(x_in, 0)
            for i in range(n):
                k = np.where(abs_xn[:, i] > xpeak, at, rt)
                xpeak = (1-k)*xpeak + k*abs_xn[:, i]
                # Do not replace this with min(1, threshold/xpeak) for DivisionByZero error
                f = np.full_like(xpeak, 0)
                for j in range(len(xpeak)):
                    f[j] = threshold/xpeak[j] if xpeak[j] > threshold else 1
                k = np.where(f < gain, at, rt)
                gain = (1-k)*gain + k*f
                gains[:, i] = gain
                y[:, i] = gain * x_in[:, i-delay_n] if i-delay_n >= 0 else 0
            return y, gains
        y_real, gain_real = calculate(x.real)
        y_imag, _ = calculate(x.imag)
        if plot:
            plt.plot(np.arange(x.shape[-1])/sr, 10*np.log10(gain_real[0, :]))
        return y_real + 1j*y_imag
    return _f


# DAFX: p112
def compressor(compressor_threshold_db, 
               compressor_scale, 
               expander_threshold_db, 
               expander_scale, 
               attack_time, 
               release_time, 
               delay_time, 
               average_time, 
               plot=False):
    def _f(sr, x):
        at = 1 - math.exp(-2.2/(attack_time*sr))
        rt = 1 - math.exp(-2.2/(release_time*sr))
        tav = 1 - math.exp(-2.2/(average_time*sr))
        n = x.shape[-1]
        delay_n = round(delay_time*sr)
        def calculate(x_in):
            xrms = np.array([0, 0])
            gain = np.array([1, 1])
            y = np.full_like(x_in, 0)
            gains = np.full_like(x_in, 0)
            for i in range(n):
                xrms = (1-tav)*xrms + tav*x_in[:, i]*x_in[:, i]
                gdb = np.full_like(xrms, 0)
                for j in range(len(xrms)):
                    if xrms[j] == 0:
                        gdb[j] = 0
                    else:
                        xdb = 10 * np.log10(xrms[j])
                        #print('xdb', xdb)
                        gdb[j] = min(
                            0, 
                            compressor_scale*(compressor_threshold_db-xdb), 
                            expander_scale*(expander_threshold_db-xdb))
                f = 10**(gdb/20)
                k = np.where(f < gain, at, rt)
                gain = (1-k)*gain + k*f
                
                gains[:, i] = gain
                y[:, i] = gain * x_in[:, i-delay_n] if i-delay_n >= 0 else 0
            return y, gains
        y_real, gain_real = calculate(x.real)
        y_imag, _ = calculate(x.imag)
        if plot:
            plt.plot(np.arange(x.shape[-1])/sr, 10*np.log10(gain_real[0, :]))
        return y_real + 1j*y_imag
    return _f


#### A simple player and mixer
def mix(sr, freq, time_points, generators, filters, connections, output_channels=('0',), profile=True):
    deps = {}
    for f, t in connections:
        if t in deps:
            deps[t].add(f)
        else:
            deps[t] = set([f])

    channel_outs = {}
    
    sort_result = toposort(deps)
    
    profile_generator = {}
    
    processed_channels = set()
    
    all_channels = set([x for x in generators] + [x for x in filters])

    
    def process_own_channel(channel, channel_in=np.zeros([2, len(time_points)], dtype='complex128')):
        channel_out = channel_in
        if channel in generators:
            for i, gen in enumerate(generators[channel]):
                t1 = datetime.now()
                channel_out += gen(sr, freq, time_points)
                t2 = datetime.now()
                if profile:
                    print('channel "%s", id=%d, generator "%s", time=%s' % (channel, i, gen, t2-t1))

        # If not filters, assume passing through
        if channel in filters:
            for i, filt in enumerate(filters[channel]):
                t1 = datetime.now()
                channel_out = filt(sr, channel_out)
                t2 = datetime.now()
                if profile:
                    print('channel "%s", id=%d, filter "%s", time=%s' % (channel, i, filt, t2-t1))
                    
        return channel_out
    
    for channels in sort_result:
        for channel in channels:
            channel_in = np.zeros([2, len(time_points)], dtype='complex128')
            if channel in deps:
                for dep_channel in deps[channel]:
                    channel_in += channel_outs[dep_channel]
            channel_outs[channel] = process_own_channel(channel, channel_in)
            processed_channels.add(channel)

    for channel in all_channels - processed_channels:
        channel_outs[channel] = process_own_channel(channel)

    ret = []
    for c in output_channels:
        ret.append(channel_outs[c])
    return ret


def plot_dft(sr, y, title='', ylim=None):
    z = np.fft.fft(y)
    mag = np.abs(np.real(z)) / (len(y)/2)
    db = np.log10(np.where(mag > 1e-10, mag, 1e-10)) * 10
    #phi = np.angle(z) / np.pi * 180
    
    fs = np.fft.fftfreq(y.shape[-1]) * sr
    valid_n = len(fs) // 2
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    p = ax.plot(fs[:valid_n], db[:valid_n])
    plt.xlabel('f(Hz)')
    plt.ylabel('mag(dB)')
    if ylim:
        plt.ylim(*ylim)
    plt.xlim(20, 20000)
    
    plt.title(title)
    ax.set_xscale('log')

def plot_filter_transfer_function(sr, f, stereo=True):
    x = np.zeros([2, sr])
    x[:, 0] = sr / 2
    y = f(sr, x)
    plot_dft(sr, y[0], title='Transfer Function(Magnitude), L')
    plot_dft(sr, y[1], title='Transfer Function(Magnitude), R')
    
def easy_visualize(sr, y):
    first_n = 1024
    
    # wave left
    plt.figure()
    plt.plot(np.arange(min(first_n, np.shape(y)[1])) / sr, y[0, :first_n])
    
    # wave right
    #plt.figure()
    #plt.plot(np.arange(min(first_n, np.shape(y)[1])) / sr, y[1, :first_n])
    
    # dft
    Yl, Yr = librosa.stft(y[0]), librosa.stft(y[1])
    Ydb_l, Ydb_r = librosa.amplitude_to_db(abs(Yl)), librosa.amplitude_to_db(abs(Yr))
    plt.figure()
    librosa.display.specshow(Ydb_l, sr=sr, x_axis='time', y_axis='log')

    plot_dft(sr, y[0], ylim=(-50, 3))
    #plot_dft(sr, y[1], ylim=(-50, 3))
    plt.show()

sr = 44100
T = 2
t = np.linspace(0, T, int(T*sr))
f = 220

generators = {
    'saw': [
        saw(0.5, 0.5, pan=30), 
        #noise(0.5, pan=0),
    ],
    'sine': [
        sine(A=0.5, pan=-30),
    ],
    'drums': [
        sampler(A=0.5, file_path='drums.wav'),
    ],
    'piano': [
        sampler(A=0.5, file_path='piano.wav'),
    ]
}

filters = {
    'vdelay': [
        delay(0.1, 0.5),
        vdelay(
            lambda i: 0.3*(math.sin(2*math.pi*0.5*i/sr)+1)/2, lambda i: 0.5),
    ],
    '2': [
        delay(0.8, 0.5),
    ],
    'iir': [
        iirfilter('lowpass', 1000/(sr/2), 1500/(sr/2)),
    ],
    'rm': [
        ring_modulator(f_c=50, carrier_func=np.sin),
    ],
    'am': [
        amplitude_modulator(f_c=2, alpha=0.5, carrier_func=np.sin),
    ],
    'pm': [
        phase_modulator(f_c=2),
    ],    
    'fm': [
        frequency_modulator(f_c=2, k=1),
    ],
    'ssb': [
        ssb_modulator(f_c=-2)
    ],
    'compressor': [
        compressor(compressor_threshold_db=-40,
                   compressor_scale=0.9,
                   expander_threshold_db=0,
                   expander_scale=1,
                   attack_time=0.01, 
                   release_time=0.01, 
                   delay_time=0.001,
                   average_time=0.05,
                   plot=True)
    ]
}
connections = [
    ('saw', 'iir'),
    ('saw', 'vdelay'),
    ('vdelay', 'master'),
    ('iir', 'master'),
    
    ('saw', 'rm'),
    ('saw', 'am'),
    ('saw', 'pm'),
    ('saw', 'fm'),
    ('saw', 'ssb'),
    ('piano', 'compressor'),
]

y_complex, = mix(sr, f, t, generators, filters, connections, output_channels=('compressor',))
y = y_complex.real

# scipy wants y to be (nsamples, nchannels)
scipy.io.wavfile.write('audio.wav', sr, y.T.astype('float32'))

# Or play it directly
#sd.default.samplerate = sr
#sd.play(qy.T, blocking=True)

# Also, you can visualize it
#easy_visualize(sr, y)
#plot_filter_transfer_function(sr, delay(1/100, 0.5), stereo=False)

# When in ipython play sound in this way
#ipd.Audio(y, rate=sr)


