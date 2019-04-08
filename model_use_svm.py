import numpy as np
import pandas as pd
from sklearn.svm import SVC


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


data, label = load_data('train.csv')
test_data = load_test_data('test.csv')

print(data.shape, label.shape, test_data.shape)

model = SVC()
model.fit(data, label)
print(model.fit(data, label))
test_predict = model.predict(test_data)
test_predict = test_predict.astype(np.int)
# submission = []
# for i in range(892, 1310):
#     submission.append([i, test_predict[i-892]])
# submission = np.array(submission)
# print(submission.shape)
data_frame = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': test_predict})
data_frame.to_csv('submission.csv',index=False)