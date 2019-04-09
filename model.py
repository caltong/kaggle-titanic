import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization


def load_data(path):
    data_csv = pd.read_csv(path)
    data_csv = data_csv.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
    data_csv['Sex'].replace(['female', 'male'], [-1, 1], inplace=True)
    temple_list = data_csv.values
    temple_list = temple_list.astype(np.float32)
    where_are_nan = np.isnan(temple_list)
    temple_list[where_are_nan] = 0
    label = temple_list[:, 0]
    data = temple_list[:, 1:]
    return data, label


data, label = load_data('train.csv')
print(data.shape, label.shape)
print(data)
print((data.dtype))

model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(5,)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
model.fit(data, label, batch_size=8, epochs=1024, verbose=1)

model.save('model.h5')
