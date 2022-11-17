
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

# logging.info('Generating x train data from train features...')
# X_trainData = np.array(train_features_extractor.features_train)

# logging.info('Generating x test data from test features...')
# X_testData = np.array(test_features_extractor.features_train)

# logging.info('Generating x validation data from validation features...')
# X_validationData = np.array(validation_features_extractor.features_train)

# logging.info('Generating y train data from speakers features array...')
# y_trainData = np.array(trainSpeakers.importedDataFrame['speakerId'])

# logging.info('Generating y validation data from speakers features array...')
# y_validationData = np.array(validationSpeakers.importedDataFrame['speakerId'])

# logging.info('Encoding y_train files to be ready for the neural network...')
# lb = LabelEncoder()
# y_trainData = to_categorical(lb.fit_transform(y_trainData))
# y_validationData = to_categorical(lb.fit_transform(y_validationData))

# logging.info('Generating scaled data...')
# ss = StandardScaler()

# logging.info('Generating scaloned X_trainData...')
# X_trainData = ss.fit_transform(X_trainData)

# logging.info('Generating scaloned X_testData...')
# X_testData = ss.transform(X_testData)

# logging.info('Generatind scaloned X_validationData...')
# X_validationData = ss.transform(X_validationData)

# # Building a simple feed forward neural network
# # Build a simple dense model with early stopping and softmax for categorical classification, remember the count of classes you have

# model = Sequential()

# model.add(Dense(193, input_shape=(193,), activation = 'relu'))
# model.add(Dropout(0.1))

# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.25))

# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(30, activation = 'softmax'))

# model.summary()

# logging.info('Compiling model...')
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

# # Fitting the model with the training and validation data
# history = model.fit(x=X_trainData, y=y_trainData, batch_size=256, epochs=100, validation_data=(X_validationData, y_validationData), callbacks=[early_stop])
# # historyDataFrame = pd.DataFrame(history)
# # historyDataFrame.to_csv('../out/modelHistoryDataFrame.csv')

# # Check out our train accuracy and validation accuracy over epochs.
# train_accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']

# # Set figure size.
# plt.figure(figsize=(12, 8))

# # Generate line plot of training, testing loss over epochs.
# plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
# plt.plot(val_accuracy, label='Validation Accuracy', color='orange')

# # Set title
# plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)

# plt.xlabel('Epoch', fontsize = 18)
# plt.ylabel('Categorical Crossentropy', fontsize = 18)
# plt.xticks(range(0,100,5), range(0,100,5))
# plt.legend(fontsize = 18);

# plt.savefig('../out/try_08_about_to_9986_percent_training_accuracy_graph_6_layers.png')

# # We get our predictions from the test data
# predictions = model.predict_classes(X_testData)

# # We transform back our predictions to the speakers ids
# predictions = lb.inverse_transform(predictions)

# # Finally, we can add those predictions to our original dataframe
# testSpeakers.importedDataFrame['predictions'] = predictions

# # Code to see which values we got wrong
# wrong_predictions = testSpeakers.importedDataFrame[testSpeakers.importedDataFrame['speaker'] != testSpeakers.importedDataFrame['predictions']]
# print("Wrong Predictions: ")
# print(wrong_predictions)

# # Code to see the numerical accuracy
# final_accuracy = (1-round(len(testSpeakers.importedDataFrame[testSpeakers.importedDataFrame['speaker'] != testSpeakers.importedDataFrame['predictions']])/len(testSpeakers.importedDataFrame),3))*100
# print("final accuracy: %d", final_accuracy)

logging.info('end.')