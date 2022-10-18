print('Importing libraries...')
from importSpeakers import ImportSpeakers
from extractFeatures import ExtractFeatures

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import logging

from sqlalchemy import create_engine
engine = create_engine('postgresql://username:password@localhost:5432/mydatabase')
# df.to_sql('table_name', engine)

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

logging.info('Indexing and importing audio files from speakers...')

trainSpeakers = ImportSpeakers('train', '../out/trainDataFrame.csv', 'speakers_train')
trainSpeakers.importedDataFrame.to_csv('../out/trainDataFrame.csv')

testSpeakers = ImportSpeakers('test', '../out/testDataFrame.csv', 'speakers_test')
testSpeakers.importedDataFrame.to_csv('../out/testDataFrame.csv')

validationSpeakers = ImportSpeakers('validation', '../out/validationDataFrame.csv', 'speakers_validation')
validationSpeakers.importedDataFrame.to_csv('../out/validationDataFrame.csv')

logging.info('Generating dataFrame from audio files...')

logging.info('Generating train files...')
train_features_extractor = ExtractFeatures('train', '../out/extracted_train_features.csv', 'speakers_train', trainSpeakers.importedDataFrame)
# if (train_features_extractor.importFromFile()):
train_features_extractor.extractFeatures()
train_features_extractor.extracted_features.to_csv('../out/extracted_train_features.csv')
train_features_extractor.generateTrain()

logging.info('Generating test files...')
test_features_extractor = ExtractFeatures('test', '../out/extracted_test_features.csv', 'speakers_test', testSpeakers.importedDataFrame)
# if (test_features_extractor.importFromFile()):
test_features_extractor.extractFeatures()
test_features_extractor.extracted_features.to_csv('../out/extracted_test_features.csv')
test_features_extractor.generateTrain()

logging.info('Generating validation files...')
validation_features_extractor = ExtractFeatures('validation', '../out/extracted_validation_features.csv', 'speakers_validation', validationSpeakers.importedDataFrame)
# if (validation_features_extractor.importFromFile()):
validation_features_extractor.extractFeatures()
validation_features_extractor.extracted_features.to_csv('../out/extracted_validation_features.csv')
validation_features_extractor.generateTrain()

logging.info('end.')
