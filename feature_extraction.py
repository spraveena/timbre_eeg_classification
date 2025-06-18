import numpy as np
import pandas as pd
from scipy.fft import fftn
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
from antropy import spectral_entropy
from scipy.signal import butter, lfilter


def offsets_features(data):


	offset_data = [[]]
	epoch_data = []
	
	for epoch in data:
		features_by_channel = []
		for channel in epoch:
			offset_data = channel[1024:1400]
			b,a = butter(4, [0.1, 10], fs=2048, btype='band')
			channel = lfilter(b, a, channel)
			#Metrics: Slope, area under curve, mean 
			offset_mean = np.mean(offset_data).astype(float)
			offset_gradient = np.gradient(offset_data).astype(float)
			
			offset_gradient_val = (offset_gradient[len(offset_gradient)-1]-offset_gradient[0])/len(offset_gradient)

			# features_by_channel=np.append(offset_mean,offset_gradient_val)
			features_by_channel.append(offset_gradient_val)
			features_by_channel.append(offset_mean)
		#total of 128 values in one channel
		epoch_data.append(features_by_channel)

	epoch_data = np.asarray(epoch_data)

	return epoch_data

def compute_psd(data):
	fs = 2048
	# freq_bands = [[0.5,3.5],[3.5,7],[8,13],[18,25],[30,70]]
	freq_bands = [[0.5,4],[4,8],[8,12]]

	epoch_psd = list()
	#band power spectral density
	for epoch in data:
		features_by_channel = list()
		for channel in epoch:
			psd,freqs = psd_array_multitaper(channel,fs,adaptive=True, normalization='full', verbose=0)
			dx = freqs[1]-freqs[0]
			for band in freq_bands:
				idx_band = np.logical_and(freqs >= band[0], freqs < band[1])
				signal_power = simps(psd[idx_band], dx=dx)
				features_by_channel.append(signal_power)
		epoch_psd.append(features_by_channel)
	epoch_psd = np.asarray(epoch_psd)
	
	return epoch_psd
	

def fundamental_freq(data,debug=False):

	#sub harmonic series: 110 Hz, 55 Hz, 27.5 Hz, 
	#harmonic series: 220Hz, 440Hz, 880 Hz
	#take freq bands +/- 1Hz

	# cz channel: 40
	fs = 2048
	# freq_points = [ 440, 880]
	freq_points = [27.5, 55, 110, 220, 440, 880]
	epoch_fundfreq_power = list()
	
	for epoch in data:
		features_by_channel = list()
		for channel in epoch:
			# b,a = butter(4, [0.1, 1024], fs=2048, btype='band')
			# channel = lfilter(b, a, channel)
			# Get frequency representation (use multitaper?)
			psd, freqs = psd_array_multitaper(channel, fs, adaptive = True, normalization='full', verbose = 0)
			for freq in freq_points:
				idx_band = np.logical_and(freqs >= freq-10, freqs <= freq + 10)
				signal_power = simps(psd[idx_band], dx=freqs[1]-freqs[0])
				features_by_channel.append(signal_power)
				
		epoch_fundfreq_power.append(features_by_channel)
	epoch_fundfreq_power = np.asarray(epoch_fundfreq_power)
	print(f"Shape of epoch_fundfreq_power : {np.shape(epoch_fundfreq_power)}") 

	return epoch_fundfreq_power

def erp_features(data,debug=False):

	#read erp wave file
	#identify zero-crossings
	#find peaks between zero-crossings
	
	fs=2048
	p1n1_epochfeatures = list()
	for epoch in data:
		features_by_channel = list()
		for channel in epoch:
			b,a = butter(4, [0.1, 40], fs=2048, btype='band')
			channel = lfilter(b, a, channel)
			p1_amp = max(channel[0:168])
			n1_amp = min(channel[168:307])
			p2_amp = max(channel[308:512])
			p1_lat = np.where(channel == p1_amp)[0][0]/fs
			n1_lat = np.where(channel == n1_amp)[0][0]/fs
			p2_lat = np.where(channel == p2_amp)[0][0]/fs
			p1_amp = p1_amp - channel[0]- channel[0]
			n1_amp = n1_amp - channel[0]- channel[0]
			p2_amp = p2_amp - channel[0]- channel[0]
			p1_mean = np.mean(channel[0:168])
			n1_mean = np.mean(channel[168:307])
			p2_mean = np.mean(channel[308:512])

			# features_by_channel.append(p1_amp)
			features_by_channel.append(n1_amp)
			features_by_channel.append(p2_amp)
			# features_by_channel.append(p1_lat)
			features_by_channel.append(n1_lat)
			features_by_channel.append(p2_lat)
			# features_by_channel.append(p1_mean)
			features_by_channel.append(n1_mean)
			features_by_channel.append(p2_mean)

		p1n1_epochfeatures.append(features_by_channel)
	p1n1_epochfeatures = np.asarray(p1n1_epochfeatures)
	if debug:
		print(f"Shape: {p1n1_epochfeatures.shape}")

	return p1n1_epochfeatures

def compute_spectral_entropy(data, debug=False):


	fs = 2048
	epoch_sp_data = list()
	# freq_bands = [[0.5,2],[2,4],[4,6.5],[6.5,8],[8,10.5],[10.5,12.5],[12.5,14],[14,19]]
	freq_bands = [[0.5,4],[4,8],[8,12]]


	for epoch in data:
		features_by_channel = list()
		for channel in epoch:
			psd, freqs = psd_array_multitaper(channel, fs, adaptive=True, normalization='full', verbose=0)
			for band in freq_bands:
				idx_band = np.logical_and(freqs >= band[0], freqs < band[1])
				band_psd = psd[idx_band]
				entropy = spectral_entropy(band_psd, fs, method="welch")
				features_by_channel.append(entropy)

		epoch_sp_data.append(features_by_channel)
	epoch_sp_data = np.asarray(epoch_sp_data)
	if debug:
		print(f"Shape of epoch_sp_data : {np.shape(epoch_sp_data)}") 

	return epoch_sp_data

def compute_periodicity(data, debug=False):
	

	periodicity_data= list()
	
	for epoch in data:
		features_by_channel = list()
		for channel in epoch:
			#play around with bandpass filter as appropriate
			# b,a = butter(4, [0.1, 50], fs=2048, btype='band')
			# channel = lfilter(b, a, channel)
			channel_fft = 2*np.abs(fftn(channel))
			channel_mag_spec = 2*channel_fft[1:int(np.ceil((len(channel)+1)/2))]
			correlation_array = np.correlate(channel_mag_spec, channel_mag_spec)
			correlation_score = np.sum(np.abs(correlation_array))/len(correlation_array)
			features_by_channel.append(correlation_score)
		periodicity_data.append(features_by_channel)
	periodicity_data = np.asarray(periodicity_data)
	if debug:
		print(f"Shape of periodicity data: {np.shape(periodicity_data)}")
	return periodicity_data

def peak_power(data, debug=False):
	#Extact peak power and corresponding frequency from oscillatory bands
	peak_power_data= list()
	# freq_bands = [[0.5,2],[2,4],[4,6.5],[6.5,8],[8,10.5],[10.5,12.5],[12.5,14],[14,21],[21,30],[30,40]]
	freq_bands = [[200,400],[400,600],[600,800],[800,1000]]
	fs = 2048
	for epoch in data:
		features_by_channel = list()
		for channel in epoch:
			psd, freqs = psd_array_multitaper(channel, fs, adaptive=True, normalization='full', verbose=0)
			for band in freq_bands:
				idx_band = np.logical_and(freqs >= band[0], freqs < band[1])
				band_psd = psd[idx_band]
				peak_power = max(band_psd)
				peak_power_freq = np.where(band_psd == peak_power)[0][0]+band[0]
				# print(peak_power_freq)

				features_by_channel.append(peak_power)
				features_by_channel.append(peak_power_freq)
		peak_power_data.append(features_by_channel)
	peak_power_data = np.asarray(peak_power_data)
	if debug:
		print(f"Shape of peak power data: {np.shape(peak_power_data)}")
	return peak_power_data


def read_coordinates(num_trials):
	coord = pd.read_csv('electrode_coord.csv',header=None,index_col=False)
	coord_data= list()
	
	
	electrode_coord = list()	
	for index, row in coord.iterrows():
		
		electrode_coord.extend([row[0],row[1]])
	coord_data = np.tile(np.asarray(electrode_coord),[num_trials,1])



	print(f"Shape of electrode coordinate data: {np.shape(coord_data)}")
	return coord_data
