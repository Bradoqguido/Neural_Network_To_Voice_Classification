print('Importing libraries...')
from importSpeakers import ImportSpeakers

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import pandas as pd
import librosa
from soundfile import *
import numpy as np
import os
from multiprocessing import  Pool
import logging

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

importedSpeakers = ImportSpeakers('speakers_train', 'speakers_test', 'speakers_validation')				

trainDataFrame = importedSpeakers.trainDataFrame
testDataFrame = importedSpeakers.testDataFrame
validationDataFrame = importedSpeakers.validationDataFrame

logging.info('Generating dataframe from audio files...')

def extract_train_features_caller(files):
	tmpDataFrame = pd.DataFrame(files)
	return tmpDataFrame.apply(extract_train_features, axis=1)

def extract_train_features(files):
	logging.info('Processing train file: %s.', files.file)
	# Sets the name to be the path to where the file is in my computer
	file_name = os.path.join(os.path.abspath('speakers_train') + '/' + str(files.file))
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

logging.info('Extracting train features from files...')
extracted_train_features = parallelize_dataframe(trainDataFrame, extract_train_features_caller)
extracted_train_features.to_csv('../out/extracted_train_features.csv')

logging.info('Generating train features train...')
features_for_train = []
for i in range(0, len(extracted_train_features)):
	features_for_train.append(np.concatenate((
		extracted_train_features[i][0],
		extracted_train_features[i][1],
		extracted_train_features[i][2],
		extracted_train_features[i][3],
		extracted_train_features[i][4]),axis=0))

def extract_test_features_caller(files):
	tmpDataFrame = pd.DataFrame(files)
	return tmpDataFrame.apply(extract_test_features, axis=1)

def extract_test_features(files):
	logging.info('Processing test file: %s.', files.file)
	# Sets the name to be the path to where the file is in my computer
	file_name = os.path.join(os.path.abspath('speakers_test') + '/' + str(files.file))
	# logging.info('Loading the audio file as a floating point time series and assigns the default sample rate...')
	# logging.info('Sample rate is set to 22050 by default...')
	X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
	#logging.info('Generating Mel-frequency cepstral coefficients (MFCCs) from a time series...')
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

logging.info('Extracting test features from files...')
extracted_test_features = parallelize_dataframe(testDataFrame, extract_test_features_caller)
extracted_test_features.to_csv('../out/extracted_test_features.csv')

logging.info('Generating test features train...')
features_for_test = []
for i in range(0, len(extracted_test_features)):
	features_for_test.append(np.concatenate((
		extracted_test_features[i][0],
		extracted_test_features[i][1],
		extracted_test_features[i][2],
		extracted_test_features[i][3],
		extracted_test_features[i][4]),axis=0))

def extract_validation_features_caller(files):
	tmpDataFrame = pd.DataFrame(files)
	return tmpDataFrame.apply(extract_validation_features, axis=1)

def extract_validation_features(files):
	logging.info('Processing validation file: %s.', files.file)
	# Sets the name to be the path to where the file is in my computer
	file_name = os.path.join(os.path.abspath('speakers_validation') + '/' + str(files.file))
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

logging.info('Extracting validation features from files...')
extracted_validation_features = parallelize_dataframe(validationDataFrame, extract_validation_features_caller)
extracted_validation_features.to_csv('../out/extracted_validation_features.csv')

logging.info('Generating validation features train...')
features_for_validation = []
for i in range(0, len(extracted_validation_features)):
	features_for_validation.append(np.concatenate((
		extracted_validation_features[i][0],
		extracted_validation_features[i][1],
		extracted_validation_features[i][2],
		extracted_validation_features[i][3],
		extracted_validation_features[i][4]),axis=0))

logging.info('Generating x train data from features_for_train...')
X_trainData = np.array(features_for_train)

logging.info('Generating x test data from features_for_test...')
X_testData = np.array(features_for_test)

logging.info('Generating x validation data from features_for_validation...')
X_validationData = np.array(features_for_validation)

# logging.info('Generating y train data from speakers features array...')
# y_trainData = np.array(trainDataFrame['speakerId'])

# logging.info('Generating y validation data from speakers features array...')
# y_validationData = np.array(['speakerId'])

# logging.info('Encoding y_train files to be ready for the neural network...')
# lb = LabelEncoder()
# y_trainData = to_categorical(lb.fit_transform(y_trainData))
# y_validationData = to_categorical(lb.fit_transform(y_validationData))

# logging.info('Generating scaled data...')
# ss = StandardScaler()

# logging.info('Generating scaloned X_trainData...')
# X_trainData = ss.fit_transform(X_trainData)

# logging.info('Generatind scaloned X_validationData...')
# X_validationData = ss.transform(X_validationData)

# logging.info('Generating scaloned X_testData...')
# X_testData = ss.transform(X_testData)

logging.info('end.')
