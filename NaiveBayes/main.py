import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv('CarDataSet.csv')

Numerics = LabelEncoder()

inputs = dataset.drop('Stolen', axis='columns')
target = dataset['Stolen']

inputs['Color_n'] = Numerics.fit_transform(inputs['Color'])
inputs['Type_n'] = Numerics.fit_transform(inputs['Type'])
inputs['Origin_n'] = Numerics.fit_transform(inputs['Origin'])

# For Color - Red - 1 and Green - 0
# For Type - SUV - 0 and Sports - 0
# For Origin - Domestic - 0 and Imported - 1

inputs_n = inputs.drop(['Sl No.','Color', 'Type', 'Origin'], axis='columns')

CLASSIFIER = GaussianNB()
CLASSIFIER.fit(inputs_n, target)

res = (CLASSIFIER.predict([[1, 0, 0]]))

print('Prediction Accuracy - ', CLASSIFIER.score(inputs_n, target))
print('For Red Domestic SUV - ', res)
