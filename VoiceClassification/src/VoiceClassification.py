print('Importing libraries...')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import pandas as pd
import librosa
from soundfile import *
import numpy as np
import os

trainDataFrame = pd.DataFrame([])
validationDataFrame = pd.DataFrame([])
testDataFrame = pd.DataFrame([])

def generateTrainData():
	print('Importing trains files...')
	filelist = os.listdir('speakers_train')
	
	print('Reading them into pandas...')
	trainDataFrame = pd.DataFrame(filelist)

	print('Renaming the column name to file...')
	trainDataFrame = trainDataFrame.rename(columns={0: 'file'})

	# Code in case we have to drop the '.DS_Store' and reset the index
	trainDataFrame[trainDataFrame['file'] == '.DS_Store']
	trainDataFrame.drop(16, inplace=True)
	trainDataFrame = trainDataFrame.sample(frac=1).reset_index(drop=True)

	# We create an empty list where we will append all the speakers ids for each row of our dataframe by slicing the file name since we know the id is the first number before the hash
	print('Extracting speakers...')
	speaker = []
	for i in range(0, len(trainDataFrame)):
		speaker.append(trainDataFrame['file'][i].split('-')[0])

	print('Linking speakers...')
	# We now assign the speaker to a new column
	trainDataFrame['speakerId'] = speaker
	trainDataFrame.to_csv('../out/trainDataFrame.csv')

def generateValidationData():
	print('Importing validation files...')
	filelist = os.listdir('speakers_validation')
	
	print('Generating dataframe from audio files...')

	# Read them into pandas
	validationDataFrame = pd.DataFrame(filelist)

	print('Renaming the column name to file...')
	validationDataFrame = validationDataFrame.rename(columns={0: 'file'})

	# Code in case we have to drop the '.DS_Store' and reset the index
	validationDataFrame[validationDataFrame['file'] == '.DS_Store']
	validationDataFrame.drop(16, inplace=True)
	validationDataFrame = validationDataFrame.sample(frac=1).reset_index(drop=True)

	# We create an empty list where we will append all the speakers ids for each row of our dataframe by slicing the file name since we know the id is the first number before the hash
	print('Extracting speakers...')
	speaker = []
	for i in range(0, len(validationDataFrame)):
		speaker.append(validationDataFrame['file'][i].split('-')[0])

	print('Linking speakers...')
	# We now assign the speaker to a new column
	validationDataFrame['speakerId'] = speaker
	validationDataFrame.to_csv('../out/validationDataFrame.csv')

def generateTestData(): 
	print('Importing test files...')
	filelist = os.listdir('speakers_test')
	
	print('Generating dataframe from audio files...')

	# Read them into pandas
	testDataFrame = pd.DataFrame(filelist)

	print('Renaming the column name to file...')
	testDataFrame = testDataFrame.rename(columns={0: 'file'})

	# Code in case we have to drop the '.DS_Store' and reset the index
	testDataFrame[testDataFrame['file'] == '.DS_Store']
	testDataFrame.drop(16, inplace=True)
	testDataFrame = testDataFrame.sample(frac=1).reset_index(drop=True)

	# We create an empty list where we will append all the speakers ids for each row of our dataframe by slicing the file name since we know the id is the first number before the hash
	print('Extracting speakers...')
	speaker = []
	for i in range(0, len(testDataFrame)):
		speaker.append(testDataFrame['file'][i].split('-')[0])

	print('Linking speakers...')
	# We now assign the speaker to a new column
	testDataFrame['speakerId'] = speaker
	testDataFrame.to_csv('../out/testDataFrame.csv')

print('Generating dataframe from audio files...')
generateTrainData()
# generateValidationData()
# generateTestData()

def extract_train_features(files):
	# Sets the name to be the path to where the file is in my computer
	file_name = os.path.join(os.path.abspath('speakers_train') + '/' + str(files.file))

	print('Loading the audio file as a floating point time series and assigns the default sample rate...')
	print('Sample rate is set to 22050 by default...')
	X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

	print('Generating Mel-frequency cepstral coefficients (MFCCs) from a time series...')
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

	print('Generating a Short-time Fourier transform (STFT) to use in the chroma_stft...')
	stft = np.abs(librosa.stft(X))

	print('Computing a chromagram from a waveform or power spectrogram...')
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

	print('Computing a mel-scaled spectrogram...')
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

	print('Computing spectral contrast...')
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft,sr=sample_rate).T,axis=0)
	
	print('Computing the tonal centroid features (tonnetz)...')
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
	return mfccs, chroma, mel, contrast, tonnetz

print('Extracting train features from files...')
extracted_train_features = trainDataFrame.apply(extract_train_features, axis=1)
extracted_train_features.to_csv('../out/extracted_train_features.csv')

print('Generating train features train...')
features_for_train = []
for i in range(0, len(extracted_train_features)):
	features_for_train.append(np.concatenate((
		extracted_train_features[i][0],
		extracted_train_features[i][1],
		extracted_train_features[i][2],
		extracted_train_features[i][3],
		extracted_train_features[i][4]),axis=0))

def extract_test_features(files):
	# Sets the name to be the path to where the file is in my computer
	file_name = os.path.join(os.path.abspath('speakers_test') + '/' + str(files.file))
	print('Loading the audio file as a floating point time series and assigns the default sample rate...')
	print('Sample rate is set to 22050 by default...')
	X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
	print('Generating Mel-frequency cepstral coefficients (MFCCs) from a time series...')
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	print('Generating a Short-time Fourier transform (STFT) to use in the chroma_stft...')
	stft = np.abs(librosa.stft(X))
	print('Computing a chromagram from a waveform or power spectrogram...')
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	print('Computing a mel-scaled spectrogram...')
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
	print('Computing spectral contrast...')
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft,sr=sample_rate).T,axis=0)
	print('Computing the tonal centroid features (tonnetz)...')
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
	return mfccs, chroma, mel, contrast, tonnetz

# print('Extracting test features from files...')
# extracted_test_features = trainDataFrame.apply(extract_test_features, axis=1)
# extracted_test_features.to_csv('../out/extracted_test_features.csv')

# print('Generating test features train...')
# features_for_test = []
# for i in range(0, len(extracted_test_features)):
# 	features_for_test.append(np.concatenate((
# 		extracted_test_features[i][0],
# 		extracted_test_features[i][1],
# 		extracted_test_features[i][2],
# 		extracted_test_features[i][3],
# 		extracted_test_features[i][4]),axis=0))

def extract_validation_features(files):
	# Sets the name to be the path to where the file is in my computer
	file_name = os.path.join(os.path.abspath('speakers_validation') + '/' + str(files.file))
	print('Loading the audio file as a floating point time series and assigns the default sample rate...')
	print('Sample rate is set to 22050 by default...')
	X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
	print('Generating Mel-frequency cepstral coefficients (MFCCs) from a time series...')
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	print('Generating a Short-time Fourier transform (STFT) to use in the chroma_stft...')
	stft = np.abs(librosa.stft(X))
	print('Computing a chromagram from a waveform or power spectrogram...')
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	print('Computing a mel-scaled spectrogram...')
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
	print('Computing spectral contrast...')
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft,sr=sample_rate).T,axis=0)
	print('Computing the tonal centroid features (tonnetz)...')
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
	return mfccs, chroma, mel, contrast, tonnetz

# print('Extracting validation features from files...')
# extracted_validation_features = trainDataFrame.apply(extract_test_features, axis=1)
# extracted_validation_features.to_csv('../out/extracted_validation_features.csv')

# print('Generating validation features train...')
# features_for_validation = []
# for i in range(0, len(extracted_validation_features)):
# 	features_for_validation.append(np.concatenate((
# 		extracted_validation_features[i][0],
# 		extracted_validation_features[i][1],
# 		extracted_validation_features[i][2],
# 		extracted_validation_features[i][3],
# 		extracted_validation_features[i][4]),axis=0))

# print('Generating x train data from features_train...')
# X_trainData = np.array(features_train)

# print('Generating x validation data from features_train...')
# X_validationData = np.array(validationDataFrame)

# print('Generating x test data from features_train...')
# X_testData = np.array(testDataFrame)

# print('Generating y train data from speakers features array...')
# y_trainData = np.array(trainDataFrame['speakerId'])

# print('Generating y validation data from speakers features array...')
# y_validationData = np.array(['speakerId'])

# print('Encoding y_train files to be ready for the neural network...')
# lb = LabelEncoder()
# y_trainData = to_categorical(lb.fit_transform(y_trainData))
# y_validationData = to_categorical(lb.fit_transform(y_validationData))

# print('Generating scaled data...')
# ss = StandardScaler()

# print('Generating scaloned X_trainData...')
# X_trainData = ss.fit_transform(X_trainData)

# print('Generatind scaloned X_validationData...')
# X_validationData = ss.transform(X_validationData)

# print('Generating scaloned X_testData...')
# X_testData = ss.transform(X_testData)

print('end.')
