print('Importing libraries...')
from importSpeakers import ImportSpeakers
from extractFeatures import ExtractFeatures

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

logging.info('Extracting train features from files...')
train_features_extractor = ExtractFeatures('train', 'speakers_train', trainDataFrame)
train_features = pd.DataFrame(train_features_extractor.extracted_features)
train_features.to_csv('../out/extracted_train_features.csv')

logging.info('Extracting test features from files...')
test_features_extractor = ExtractFeatures('test', 'speakers_test', testDataFrame)
test_features = pd.DataFrame(test_features_extractor.extracted_features)
test_features.to_csv('../out/extracted_test_features.csv')

logging.info('Extracting validation features from files...')
validation_features_extractor = ExtractFeatures('validation', 'speakers_validation', validationDataFrame)
validation_features = pd.DataFrame(validation_features_extractor.extracted_features)
validation_features.to_csv('../out/extracted_validation_features.csv')

logging.info('Generating x train data from features_for_train...')
X_trainData = np.array(test_features_extractor.features)

logging.info('Generating x test data from features_for_test...')
X_testData = np.array(test_features_extractor.features)

logging.info('Generating x validation data from features_for_validation...')
X_validationData = np.array(validation_features_extractor.features)

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
