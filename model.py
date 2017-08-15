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
    # augment data with flipped images
    flipped_image = cv2.flip(image,1)
    images.append(  flipped_image )
    inverted_measurement = measurement*-1.0 
    measurements.append( inverted_measurement )
X_train = np.array(images)
y_train = np.array(measurements)	

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Dropout

def LeNet5Model():
    model = Sequential()
    # Prepocesssing
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    # Model Layers
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))     
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    return model

def NvidiaModel():
    model = Sequential()
        # Prepocesssing
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    # Model Layers
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = LeNet5Model()
#model = NvidiaModel()

model.compile(loss='mse', optimizer='adam')
    
myEpochs=1
myBatchsize = 128 # 64 faster  # 32 default
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=myBatchsize, epochs=myEpochs)

model.save('model.h5')

		

