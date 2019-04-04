import numpy as np
import pandas as pd


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
