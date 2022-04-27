# Standard python numerical analysis imports:
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import os
from os.path import exists

# LIGO-specific readligo.py 
from ligotools import readligo as rl
from ligotools import utils

eventname = 'GW150914'
fnjson = "ligotools/tests/ligo_data/BBH_events_v3.json"
events = json.load(open(fnjson,"r"))
event = events[eventname]
tevent = event['tevent']
fs = event['fs']

fn_H1 = 'ligotools/tests/ligo_data/' + event['fn_H1']    
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
time = time_H1

NFFT = 4*fs
dt = time[1] - time[0]

Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
psd_H1 = interp1d(freqs, Pxx_H1)

def test_whiten():
	strain_H1_whiten = utils.whiten(strain_H1,psd_H1,dt)

	assert (round(strain_H1_whiten[0]) == 648) & (len(strain_H1_whiten) == 131072)
	
def test_write_wavfile():
	deltat_sound = 2.
	strain_H1_whiten = utils.whiten(strain_H1,psd_H1,dt)
	
	indxd = np.where((time >= tevent-deltat_sound) & (time < tevent+deltat_sound))
	utils.write_wavfile(eventname+"_H1.wav",int(fs), strain_H1[indxd])
	assert exists(eventname+"_H1.wav")
	os.remove(eventname+"_H1.wav")

def test_reqshift():
	strain_H1_whiten = utils.whiten(strain_H1,psd_H1,dt)
	shifted = utils.reqshift(strain_H1_whiten)
	
	assert (round(shifted[0]) == 223) & (len(shifted) == 131072)
	
def test_plot_all():
	psd_window = np.blackman(NFFT)
	NOVL = NFFT/2
	fn_template = 'ligotools/tests/ligo_data/' + event['fn_template']
	f_template = h5py.File(fn_template, "r")
	template_p, template_c = f_template["template"][...]
	template = (template_p + template_c*1.j) 
	try:   
		dwindow = signal.tukey(template.size, alpha=1./8)
	except: 
		dwindow = signal.blackman(template.size)      
	det = 'H1'

	data = strain_H1.copy()
	
	datafreq = np.fft.fftfreq(template.size)*fs
	
	data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
	
	f = freqs.copy()
	df = f[2]-f[1]
	data_fft = np.fft.fft(data*dwindow) / fs
	template_fft = np.fft.fft(template*dwindow) / fs
	power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
	
	optimal = data_fft * template_fft.conjugate() / power_vec
	optimal_time = 2*np.fft.ifft(optimal)*fs

	sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
	sigma = np.sqrt(np.abs(sigmasq))
	SNR_complex = optimal_time/sigma

	peaksample = int(data.size / 2)  # location of peak in the template
	SNR_complex = np.roll(SNR_complex,peaksample)
	SNR = abs(SNR_complex)

	indmax = np.argmax(SNR)
	timemax = time[indmax]
	SNRmax = SNR[indmax]

	d_eff = sigma / SNRmax
	horizon = sigma/8

	phase = np.angle(SNR_complex[indmax])
	offset = (indmax-peaksample)

	template_phaseshifted = np.real(template*np.exp(1j*phase))
	template_rolled = np.roll(template_phaseshifted,offset) / d_eff

	fband = event['fband'] 
	bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
	normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
	strain_H1_whiten = utils.whiten(strain_H1,psd_H1,dt)
	strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization
	template_whitened = utils.whiten(template_rolled,interp1d(freqs, data_psd),dt)
	template_match = filtfilt(bb, ab, template_whitened) / normalization
	pcolor='r'
	strain_whitenbp = strain_H1_whitenbp
	template_H1 = template_match.copy()

	# -- Plot the result
	utils.plot_all(time, timemax, SNR, pcolor, det, eventname, 'png', tevent, strain_whitenbp, template_match, template_fft, datafreq, d_eff, freqs, data_psd, fs, 'ligotools/tests/')
	
	assert(exists('ligotools/tests/GW150914_H1_matchfreq.png'))
	assert(exists('ligotools/tests/GW150914_H1_matchtime.png'))
	assert(exists('ligotools/tests/GW150914_H1_SNR.png'))
	os.remove('ligotools/tests/GW150914_H1_matchfreq.png')
	os.remove('ligotools/tests/GW150914_H1_matchtime.png')
	os.remove('ligotools/tests/GW150914_H1_SNR.png')
	