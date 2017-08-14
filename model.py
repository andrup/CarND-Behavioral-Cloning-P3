import csv
import cv2
import numpy as np


lines = []
dataPath = './data/';

with open(dataPath + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #skip headerline
    next(reader, None)
    for line in reader:
        lines.append(line)
images=[]
measurements=[]
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = dataPath +'IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
X_train = np.array(images)
y_train = np.array(measurements)	

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

model.save('model.h5')

		

