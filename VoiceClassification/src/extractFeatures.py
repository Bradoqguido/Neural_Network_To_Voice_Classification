import pandas as pd
import librosa
from soundfile import *
import numpy as np
import os
from multiprocessing import  Pool
import logging
from feature import Feature

class ExtractFeatures:
	def __init__(self, file_type, path_to_csv_file, folder_path, dataFrameToExtract):
		self.file_type = file_type
		self.folder_path = folder_path
		self.dataFrameToExtract = dataFrameToExtract
		self.path_to_csv_file = path_to_csv_file
		self.features_train = []
		self.extracted_features = []
		self.json_features = []
	
	def importFromFile(self):
		try:
			raw_extracted_features = pd.read_csv(self.path_to_csv_file, index_col=0)
			json_features = []
			for i in range(0, len(raw_extracted_features)):
				tmpFeature = Feature()
				tmpFeature.fromObject(raw_extracted_features['0'][i])
				json_features.append(tmpFeature)
			self.extracted_features = json_features
			return False
		except:
			return True

	def generateTrain(self):
		logging.info('Generating %s features train...', self.file_type)
		for i in range(0, len(self.extracted_features)):
			self.features_train.append(np.concatenate((
				self.extracted_features[i].mfccs,
				self.extracted_features[i].chroma,
				self.extracted_features[i].mel,
				self.extracted_features[i].contrast,
				self.extracted_features[i].tonnetz),axis=0))

	def extractFeatures(self):
		logging.info('Extracting %s features from files...', self.file_type)
		self.json_features = self._parallelize_dataFrame(self.dataFrameToExtract, self._extract_features_caller)
		json_features = []
		for i in range(0, len(self.json_features)):
			tmpFeature = Feature()
			tmpFeature.fromObject(self.json_features[i])
			json_features.append(tmpFeature)
		self.extracted_features = json_features

	def _parallelize_dataFrame(self, df, func, n_cores=16):
		# Use parallelism processing to process faster.
		df_split = np.array_split(df, n_cores)
		pool = Pool(n_cores)
		df = pd.concat(pool.map(func, df_split))
		pool.close()
		pool.join()
		return df

	def _extract_features_caller(self, files):
		tmpDataFrame = pd.DataFrame(files)
		return tmpDataFrame.apply(self._extract_features, axis=1)

	def _extract_features(self, files):
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

		tmp = Feature(str(files.file), mfccs.tolist(), chroma.tolist(), mel.tolist(), contrast.tolist(), tonnetz.tolist())
		return tmp.toObject()
