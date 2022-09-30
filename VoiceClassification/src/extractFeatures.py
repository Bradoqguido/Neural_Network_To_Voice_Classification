import pandas as pd
import librosa
from soundfile import *
import numpy as np
import os
from multiprocessing import  Pool
import logging

class ExtractFeatures:
	def __init__(self, file_type, folder_path, dataFrame):
		self.file_type = file_type
		self.folder_path = folder_path
		self.extracted_features = self.parallelize_dataFrame(dataFrame, self.extract_features_caller)
		logging.info('Generating train features train...')
		self.features = []
		for i in range(0, len(self.extracted_features)):
			self.features.append(np.concatenate((
				self.extracted_features[i][0],
				self.extracted_features[i][1],
				self.extracted_features[i][2],
				self.extracted_features[i][3],
				self.extracted_features[i][4]),axis=0))
	
	def parallelize_dataFrame(self, df, func, n_cores=16):
		# Use parallelism processing to process faster.
		df_split = np.array_split(df, n_cores)
		pool = Pool(n_cores)
		df = pd.concat(pool.map(func, df_split))
		pool.close()
		pool.join()
		return df

	def extract_features_caller(self, files):
		tmpDataFrame = pd.DataFrame(files)
		return tmpDataFrame.apply(self.extract_features, axis=1)

	def extract_features(self, files):
		logging.info('Processing %s file: %s.', self.file_type, files.file)
		# Sets the name to be the path to where the file is in my computer
		file_name = os.path.join(os.path.abspath(self.folder_path) + '/' + str(files.file))

		# logging.info('Loading the audio file as a floating point time series and assigns the default sample rate...')
		# logging.info('Sample rate is set to 22050 by default...')
		X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

		# logging.info('Generating Mel-frequency cepstral coefficients (MFCCs) from a time series...')
		mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

		# logging.info('Generating a Short-time Fourier transform (STFT) to use in the chroma_stft...')
		stft = np.abs(librosa.stft(X))

		# logging.info('Computing a chromagram from a waveform or power spectrogram...')
		chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

		# logging.info('Computing a mel-scaled spectrogram...')
		mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

		# logging.info('Computing spectral contrast...')
		contrast = np.mean(librosa.feature.spectral_contrast(S=stft,sr=sample_rate).T,axis=0)

		# logging.info('Computing the tonal centroid features (tonnetz)...')
		tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)

		return mfccs, chroma, mel, contrast, tonnetz
