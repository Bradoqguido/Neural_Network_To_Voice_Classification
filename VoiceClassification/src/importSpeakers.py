import numpy as np
import os
import logging
import pandas as pd

class ImportSpeakers:
    def __init__(self, train_path, test_path, validation_path):
        self.trainDataFrame = self.importTrainData(train_path)
        self.testDataFrame = self.importTestData(test_path)
        self.validationDataFrame = self.importValidationData(validation_path)

    def importTrainData(self, train_path): 
        logging.info('Importing trains files...')
        filelist = os.listdir(train_path)

        logging.info('Reading them into pandas...')
        trainDataFrame = pd.DataFrame(filelist)

        logging.info('Renaming the column name to file...')
        trainDataFrame = trainDataFrame.rename(columns={0: 'file'})

        # Code in case we have to drop the '.DS_Store' and reset the index
        trainDataFrame[trainDataFrame['file'] == '.DS_Store']
        trainDataFrame.drop(16, inplace=True)
        trainDataFrame = trainDataFrame.sample(frac=1).reset_index(drop=True)

        # We create an empty list where we will append all the speakers ids for each row of our dataframe by slicing the file name since we know the id is the first number before the hash
        logging.info('Extracting speakers...')
        speaker = []
        for i in range(0, len(trainDataFrame)):
            speaker.append(trainDataFrame['file'][i].split('-')[0])

        logging.info('Linking speakers...')
        # We now assign the speaker to a new column
        trainDataFrame['speakerId'] = speaker
        trainDataFrame.to_csv('../out/trainDataFrame.csv')
        return trainDataFrame

    def importTestData(self, test_path): 
        logging.info('Importing test files...')
        filelist = os.listdir(test_path)

        logging.info('Generating dataframe from audio files...')

        # Read them into pandas
        testDataFrame = pd.DataFrame(filelist)

        logging.info('Renaming the column name to file...')
        testDataFrame = testDataFrame.rename(columns={0: 'file'})

        # Code in case we have to drop the '.DS_Store' and reset the index
        testDataFrame[testDataFrame['file'] == '.DS_Store']
        testDataFrame.drop(16, inplace=True)
        testDataFrame = testDataFrame.sample(frac=1).reset_index(drop=True)

        # We create an empty list where we will append all the speakers ids for each row of our dataframe by slicing the file name since we know the id is the first number before the hash
        logging.info('Extracting speakers...')
        speaker = []
        for i in range(0, len(testDataFrame)):
            speaker.append(testDataFrame['file'][i].split('-')[0])

        logging.info('Linking speakers...')
        # We now assign the speaker to a new column
        testDataFrame['speakerId'] = speaker
        testDataFrame.to_csv('../out/testDataFrame.csv')
        return testDataFrame

    def importValidationData(self, train_path):
        logging.info('Importing validation files...')
        filelist = os.listdir(train_path)

        # Read them into pandas
        validationDataFrame = pd.DataFrame(filelist)

        logging.info('Renaming the column name to file...')
        validationDataFrame = validationDataFrame.rename(columns={0: 'file'})

        # Code in case we have to drop the '.DS_Store' and reset the index
        validationDataFrame[validationDataFrame['file'] == '.DS_Store']
        validationDataFrame.drop(16, inplace=True)
        validationDataFrame = validationDataFrame.sample(frac=1).reset_index(drop=True)

        # We create an empty list where we will append all the speakers ids for each row of our dataframe by slicing the file name since we know the id is the first number before the hash
        logging.info('Extracting speakers...')
        speaker = []
        for i in range(0, len(validationDataFrame)):
            speaker.append(validationDataFrame['file'][i].split('-')[0])

        logging.info('Linking speakers...')
        # We now assign the speaker to a new column
        validationDataFrame['speakerId'] = speaker
        validationDataFrame.to_csv('../out/validationDataFrame.csv')
        return validationDataFrame
