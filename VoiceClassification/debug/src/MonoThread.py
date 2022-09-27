print('Importing libraries...')
from cmath import log
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import pandas as pd
import librosa
from soundfile import *
import numpy as np
import os

print('Importing trains files...')
#list the files
filelist = os.listdir('speakers_train')

print('Generating dataframe from train files...')
#read them into pandas
originalDataFrame = pd.DataFrame(filelist)

# Renaming the column name to file
trainDataFrame = originalDataFrame.rename(columns={0: 'file'})

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

print('Generating trainDataFrame.csv for analysis...')
trainDataFrame.to_csv('../out/debugMonoThreadTrainDataFrame.csv')


def extract_features(files):
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
	mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
	print('Computing spectral contrast...')
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft,sr=sample_rate).T,axis=0)
	print('Computing the tonal centroid features (tonnetz)...')
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
	return mfccs, chroma, mel, contrast, tonnetz

print('Extracting features from files...')
train_features = trainDataFrame.apply(extract_features, axis=1)

train_features.to_csv('debugMonoThread..csv');

print('end.')