import numpy as np
import pandas as pd
import keras


def load_test_data(path):
    data_csv = pd.read_csv(path)
    data_csv = data_csv.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
    data_csv['Sex'].replace(['female', 'male'], [-1, 1], inplace=True)
    temple_list = data_csv.values
    temple_list = temple_list.astype(np.float32)
    where_are_nan = np.isnan(temple_list)
    temple_list[where_are_nan] = 0
    data = temple_list
    return data


data = load_test_data('test.csv')

model = keras.models.load_model('model.h5')
predict = model.predict_classes(data, verbose=1)
predict = np.reshape(predict, predict.shape[0])
print(predict.shape)

data_frame = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': predict})
data_frame.to_csv('submission_nn.csv', index=False)
