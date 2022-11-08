
import json

class Feature:
    def __init__(self, fileName = '', mfccs = [], chroma = [], mel = [], contrast = [], tonnetz = []):
        self.fileName = fileName
        self.mfccs = mfccs
        self.chroma = chroma
        self.mel = mel
        self.contrast = contrast
        self.tonnetz = tonnetz
    
    def fromObject(self, data):
        tmpObject = json.loads(data)
        self.fileName = tmpObject['fileName']
        self.mfccs = tmpObject['mfccs']
        self.chroma = tmpObject['chroma']
        self.mel = tmpObject['mel']
        self.contrast = tmpObject['contrast']
        self.tonnetz = tmpObject['tonnetz']


    def toObject(self):
        return json.dumps({
            'fileName': self.fileName,
            'mfccs': self.mfccs,
            'chroma': self.chroma,
            'mel': self.mel,
            'contrast': self.contrast,
            'tonnetz': self.tonnetz
        })
