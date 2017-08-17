import csv
import cv2
import numpy as np
import random
import os.path

if os.path.isfile('model.h5'):
    modelmode = 'NEW'
else:
    modelmode = 'TUNE'
    
    
lines = []

dataPath = './data/mydata/';
logfile = 'predictiontest.csv' # 'driving_log.csv'


dataPath = './data/';
logfile = 'driving_log.csv'


def showImage(img):
    window_width = img.shape[1] 
    window_height = img.shape[0]
    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)
    cv2.imshow('dst_rt', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return



with open(dataPath + logfile ) as csvfile:
    reader = csv.reader(csvfile)
    #skip headerline
    next(reader, None)
    for line in reader:
        lines.append(line)
images=[]
measurements=[]

imgsizeX = 320
imgsizeY = 160


# take 100 random images only to spped up first tests
n_rand = len(lines)
print('n = ', n_rand)
newlines = np.copy(lines)
#random.shuffle(newlines)
randlines = newlines[:n_rand]
flipcount=0
#for line in randlines:
for line in lines:

    # center camera
    
    source_path = line[0]
    filename = source_path.split('/')[-1] # linux
    # filename = source_path.split('\\               ')[-1] # windows
    current_path = dataPath +'IMG/' + filename
    
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    # augment data with flipped images
    if measurement != 0.0:
        flipped_image = cv2.flip(image,1)
        images.append(  flipped_image )
        inverted_measurement = measurement*-1.0 
        measurements.append( inverted_measurement )
        flipcount+=1

    # left camera
    source_path = line[1]
    filename = source_path.split('/')[-1] # linux
    #filename = source_path.split('\\               ')[-1] # windows
    current_path = dataPath +'IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3]) + 0.08
    measurements.append(measurement)
    # augment data with flipped images
    if measurement != 0.0:
        flipped_image = cv2.flip(image,1)
        images.append(  flipped_image )
        inverted_measurement = measurement*-1.0 
        measurements.append( inverted_measurement )
        flipcount+=1

    #right camera
    source_path = line[2]
    filename = source_path.split('/')[-1] # linux
    #filename = source_path.split('\\               ')[-1] # windows
    current_path = dataPath +'IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3]) -  0.08
    measurements.append(measurement)
    # augment data with flipped images
    if measurement != 0.0:
        flipped_image = cv2.flip(image,1)
        images.append(  flipped_image )
        inverted_measurement = measurement*-1.0 
        measurements.append( inverted_measurement )
        flipcount+=1   
    
        
X_train = np.array(images)
y_train = np.array(measurements)	
print('Added flipped images: ',flipcount)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D
from keras.layers import MaxPooling2D, Dropout  # used for LeNet5 only

def LeNet5Model():
    model = Sequential()
    # Prepocesssing
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)) )
    #model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80,160,3), output_shape=(80,160,3)))
    #model.add(Cropping2D(cropping=((50,20), (0,0))))
    #model.add(Cropping2D(cropping=((25,10), (0,0))))
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
    #model.add(Cropping2D(cropping=((50,20), (0,0))))
    # Model Layers
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Flatten())
    # x = Dropout(0.5)(x)
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#model = LeNet5Model()
model = NvidiaModel()

model.compile(loss='mse', optimizer='adam')
    
myEpochs=1
myBatchsize = 128 # 64 faster  # 32 default
lastrun = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=myBatchsize, epochs=myEpochs)

model.save('model.h5')

 # print the keys contained in the history object
print(lastrun.history.keys())

# plot the training and validation loss for each epoch
plt.plot(lastrun.history['loss'])
plt.plot(lastrun.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
    
plt.show()


# TODO
# Cropping checken
# images verkleinern


		

