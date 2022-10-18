import numpy as np
import os
import logging
import pandas as pd

class ImportSpeakers:
    def __init__(self, file_type, path_to_csv_file, file_path):
        self.file_type = file_type
        self.file_path = file_path

        if (self._importFromFile()):
            self.importedDataFrame = self.importDataFrame()
        else:
            self.importedDataFrame = self.importDataFrame()

        self.fileCount = self.importedDataFrame['file'].count()
        self.speakersByFileCount = self.importedDataFrame['speakerId'].value_counts()
        self.speakersCount = self.speakersByFileCount.count()

    def _importFromFile(self):
        try:
            self.importedDataFrame = pd.read_csv(self.path_to_csv_file)
            return False
        except:
            return True

    def importDataFrame(self): 
        logging.info('Importing %s files...', self.file_type)
        filelist = os.listdir(self.file_path)

        logging.info('Reading them into pandas...')
        tmpDataFrame = pd.DataFrame(filelist)

        logging.info('Renaming the column name to file...')
        tmpDataFrame = tmpDataFrame.rename(columns={0: 'file'})

        # Code in case we have to drop the '.DS_Store' and reset the index
        tmpDataFrame[tmpDataFrame['file'] == '.DS_Store']
        tmpDataFrame.drop(16, inplace=True)
        tmpDataFrame = tmpDataFrame.sample(frac=1).reset_index(drop=True)

        # We create an empty list where we will append all the speakers ids for each row of our dataframe by slicing the file name since we know the id is the first number before the hash
        logging.info('Extracting speakers...')
        speaker = []
        for i in range(0, len(tmpDataFrame)):
            speaker.append(tmpDataFrame['file'][i].split('-')[0])
        
        logging.info('Linking speakers...')
        # We now assign the speaker to a new column
        tmpDataFrame['speakerId'] = speaker
        return tmpDataFrame
