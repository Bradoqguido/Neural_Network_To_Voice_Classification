print('Importing libraries...')
from importSpeakers import ImportSpeakers
from extractFeatures import ExtractFeatures

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import logging

# from sqlalchemy import create_engine
# engine = create_engine('postgresql://username:password@localhost:5432/mydatabase')
# df.to_sql('table_name', engine)

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

logging.info('Indexing and importing audio files from speakers...')

trainSpeakers = ImportSpeakers('train', 'speakers_train')
trainSpeakers.importedDataFrame.to_csv('../out/trainDataFrame.csv')

testSpeakers = ImportSpeakers('test', 'speakers_test')
testSpeakers.importedDataFrame.to_csv('../out/testDataFrame.csv')

validationSpeakers = ImportSpeakers('validation', 'speakers_validation')
validationSpeakers.importedDataFrame.to_csv('../out/validationDataFrame.csv')

logging.info('Generating dataFrame from audio files...')

train_features_extractor = ExtractFeatures('train', 'speakers_train', trainSpeakers.importedDataFrame)
train_features_extractor.extracted_features.to_csv('../out/extracted_train_features.csv')

test_features_extractor = ExtractFeatures('test', 'speakers_test', testSpeakers.importedDataFrame)
test_features_extractor.extracted_features.to_csv('../out/extracted_test_features.csv')

validation_features_extractor = ExtractFeatures('validation', 'speakers_validation', validationSpeakers.importedDataFrame)
validation_features_extractor.extracted_features.to_csv('../out/extracted_validation_features.csv')

logging.info('Generating x train data from train features...')
X_trainData = np.array(train_features_extractor.features_train)

logging.info('Generating x test data from test features...')
X_testData = np.array(test_features_extractor.features_train)

logging.info('Generating x validation data from validation features...')
X_validationData = np.array(validation_features_extractor.features_train)

logging.info('Generating y train data from speakers features array...')
y_trainData = np.array(trainSpeakers.importedDataFrame['speakerId'])

logging.info('Generating y validation data from speakers features array...')
y_validationData = np.array(validationSpeakers.importedDataFrame['speakerId'])

logging.info('Encoding y_train files to be ready for the neural network...')
lb = LabelEncoder()
y_trainData = to_categorical(lb.fit_transform(y_trainData))
y_validationData = to_categorical(lb.fit_transform(y_validationData))

logging.info('Generating scaled data...')
ss = StandardScaler()

logging.info('Generating scaloned X_trainData...')
X_trainData = ss.fit_transform(X_trainData)

logging.info('Generating scaloned X_testData...')
X_testData = ss.transform(X_testData)

logging.info('Generatind scaloned X_validationData...')
X_validationData = ss.transform(X_validationData)

logging.info('end.')
