print('Importing libraries...')
from feature import Feature
from importSpeakers import ImportSpeakers
from extractFeatures import ExtractFeatures

from cmath import log
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import librosa
from soundfile import *
import os

import pandas as pd
import numpy as np
import logging

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

logging.info('Indexing and importing audio files from speakers...')

trainSpeakers = ImportSpeakers('train', '../out/debugMonoThreadTrainDataFrame.csv', 'speakers_train')
trainSpeakers.importedDataFrame.to_csv('../out/debugMonoThreadTrainDataFrame.csv')

logging.info('Generating dataFrame from audio files...')

logging.info('Generating train files...')
train_features_extractor = ExtractFeatures('train', '../out/debugMonoThreadTrainFeatures.csv', 'speakers_train', trainSpeakers.importedDataFrame)
if (train_features_extractor.importFromFile()):
    train_features_extractor.extractFeatures()
    pd.DataFrame(train_features_extractor.json_features).to_csv('../out/debugMonoThreadTrainFeatures.csv')
train_features_extractor.generateTrain()

# extracted_features = pd.read_csv('../out/debugMonoThreadTrainFeatures.csv', index_col=0)
# json_features = []
# for i in range(0, len(extracted_features)):
# 	tmpFeature = Feature()
# 	tmpFeature.fromObject(extracted_features['0'][i])
# 	json_features.append(tmpFeature)

# features_train = []
# print('Generating train features train...')
# for i in range(0, len(json_features)):
# 	features_train.append(np.concatenate((
# 		json_features[i].mfccs,
# 		json_features[i].chroma,
# 		json_features[i].mel,
# 		json_features[i].contrast,
# 		json_features[i].tonnetz),axis=0))

X_trainData = np.array(train_features_extractor.features_train)

print('end.')