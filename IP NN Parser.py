import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout


#import data files, dataset is known knowns and testset is unknowns to be calculated
dataset = pd.read_csv('KnownIPList.csv', delimiter = ',')
testset = pd.read_csv('UnknownIPList.csv', delimiter = ',')
x = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 0].values
z = testset.iloc[:, 0:4].values

#transform agency data to a matrix of 16 columns based on 16 agencies in dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
y = y.reshape(len(y), 1)
y = onehot_encoder.fit_transform(y)

#randomize dataset input and split to training sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1, random_state = 0)

#convert int to float vector for tensor processing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#define the neural network with input, hidden layer, and output to match agency matrix
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=4, units=547, kernel_initializer="uniform")) 
classifier.add(Dropout(0.2))
classifier.add(Dense(activation="relu", units=547, kernel_initializer="uniform")) 
classifier.add(Dense(activation="softmax", units=16, kernel_initializer="uniform")) 

#define model and loss type
classifier.compile(optimizer = 'nadam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#execute the neural network 
classifier.fit(x_train, y_train, batch_size = 10, epochs = 10)

#uncomment below to test single ip address, note each octect is a seperate column
#new_prediction = classifier.predict(sc.transform(np.array([[10,70,1,13]])))

#run the trained neural network against the unknowns testset
new_prediction = classifier.predict(sc.transform(np.array(z[:,:])))

#output results to a csv file
np.savetxt('iplistfound1107a.csv', new_prediction, delimiter = ',')

classifier.save('model.hdf')
#keras.models.load_model('model.hdf')

