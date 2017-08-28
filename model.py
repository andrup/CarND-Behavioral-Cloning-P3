######## first import all modules and functions  ###########
import math
import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time

from shutil import copyfile
from datetime import datetime
from keras.models import Sequential, load_model 
from keras.layers import Flatten, Dense, Conv2D, Lambda, Dropout
from keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


######### define Constants #############

modelfilename = "model"

dataPath = './data/'
#dataPath = "./dataps3/"
logfile = 'driving_log.csv'

logStartPosition = 0

print( os.getcwd()) # Prints the working directory

steeringAngleCorrection = [0, 0.28, -0.28]

imgsizeX = 160 # 320 original
imgsizeY = 80  # 160 original

epochs = 1
batchsize = 128 # 64 faster  # 32 default 

######## innitialize variables ########
lines = []
images = []
measurements = []
start_time = time.time()

######## define Functions ###########

def fileBackup(modelfilename):
    filename = modelfilename+'.h5'
    modifiedTime = os.path.getmtime(filename) 
    timeStamp =  datetime.fromtimestamp(modifiedTime).strftime("%Y%m%d%H-%M%S")
    newfilename = filename + '_' + timeStamp
    copyfile(filename, newfilename)    
    print('Model backup finished.', newfilename)

def saveModelAndWeights(modelfilename):   
    modelweightsfilename = modelfilename + '.h5'
    model.save(modelweightsfilename)
    print("Model and weights saved to ", modelweightsfilename)
    return

def loadModelAndWeights(modelfilename):    
    # load weights into new model.
    filename = modelfilename + '.h5'    
    loaded_model = load_model('model.h5')
    print("Loaded model  %s from disk" % filename)
    return loaded_model    

# for debugging only
def showImage(img):
    window_width = img.shape[1] 
    window_height = img.shape[0]
    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)
    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# for smaple log Linux
def getImage( dataDir, source_path ):    
    #for linux logs
    filename = dataDir + source_path.split('/')[-1]
    # for windows logs
    #filename = dataDir + source_path.split('\\')[-1]
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def random_brightness(image):
    # returns an image with a random degree of brightness.    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def plotResult(hist):
    # plot the training and validation loss for each epoch
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    return


def Nvidea_Model(activationFunction,dropProbability=0.3):
    print('dropProbability:',dropProbability)
    print('activationFunction:',activationFunction)
    
    model = Sequential()
    # Preprocessing
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80,160,3 )))
    # Model Layers
    model.add(Conv2D(24,(5,5), strides=(2,2), activation=activationFunction))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation=activationFunction))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation=activationFunction))
    model.add(Conv2D(64,(3,3), activation=activationFunction))
    model.add(Conv2D(64,(3,3), activation=activationFunction))
    model.add(Flatten())
    model.add(Dense(100, activation=activationFunction))
    model.add(Dropout(dropProbability))
    model.add(Dense(50, activation=activationFunction))
    model.add(Dropout(dropProbability))
    model.add(Dense(10, activation=activationFunction))
    model.add(Dropout(dropProbability))
    model.add(Dense(1))
    return model


def data_generator(data, batch_size, steeringAngleCorrection, modus, dataDir):
    num_samples = len(data)
    #print('modus %s'  %modus)
    #print('data len:' ,num_samples )    
    while True:  # endless loop
        shuffle(data)
        for offset in range(0, num_samples, batch_size):  # loop over batches
            batch_samples = data[offset : (offset+batch_size)]
            images = []
            measurements = []
            #print('offset ' , offset)
            # loop over samples in batch 
            for batch_sample in batch_samples:
                measurement = float(batch_sample[3]) # 3 = steering_angle 
                #if modus == 'TRAIN':                 
                    
                if modus == 'TRAIN':                    
                    for camind in range(0,3):    #0,1,2                             
                        adaptedMeasurement = measurement + steeringAngleCorrection[camind]                                 
                        measurements.append(adaptedMeasurement)
                        image = getImage(dataDir + 'IMG/', batch_sample[camind])
                        
                        image = random_brightness(image)
                        
                        # nvidea
                        image = image[60:140, 0:320] # Crop from x, y, w, h -> 100, 200, 300, 400 
                        image = cv2.resize(image, (160, 80), interpolation=cv2.INTER_AREA)                                                
                        images.append( img_to_array(image) )   
                        
                        # added flipped image and steering angle with for 50% of all samples
                        if random.randrange(2) == 1:
                            imageFlipped = cv2.flip(image, 1)                    
                            images.append( img_to_array( imageFlipped ))
                            measurements.append( adaptedMeasurement * (-1) )
                else:
                    measurements.append(measurement)
                    image = getImage(dataDir + 'IMG/', batch_sample[0])  # 0  = center image
                    
                    # nvidea
                    image = image[60:140, 0:320] # Crop from x, y, w, h -> 100, 200, 300, 400 
                    
                    # Resize, referring to http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
                    # INTER_LINEAR - a bilinear interpolation (used by default)
                    # INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
                    image = cv2.resize(image, (160, 80), interpolation=cv2.INTER_AREA)
                    
                    images.append( img_to_array(image) )   
                    
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)
            

###################################
########## MAIN PROGRAM ###########
###################################
    
if os.path.isfile(modelfilename+'.h5'):
    modelmode = 'TUNING'
    fileBackup(modelfilename)
    model = loadModelAndWeights(modelfilename)
else:
    modelmode = 'NEW MODEL'

    
print('Model mode %s: ' % modelmode)
print("Batch size: ", batchsize)
print("Epochs:", epochs)
print("------------")


# Import images from logfiles      
print('Using Logfile ' ,dataPath + logfile)
with open(dataPath + logfile ) as csvfile:
    reader = csv.reader(csvfile)
    #skip headerline
    #next(reader, None)        
    i=0
    lines_tmp=[]
    for line in reader:                
        # 2/3 of all data is straigt with angle 0.
        # To prevent a bias for straight 2 of 3 rows with angle 0 are dropped. Later it's flipped and though doubled again     
        # sort out extreme steering angles
         lines_tmp.append(line)
    print( 'Total lines in Logfiles : ' , len(lines_tmp) )             
    lines_tmp = lines_tmp[logStartPosition:]
    for line in lines_tmp:
        if ( float(line[3]) != 0.0 and float(line[3]) < 30.0  and float(line[3]) > -30.0 ) or  ( float(line[3]) == 0.0 and random.randint(0,2) == 1):
            lines.append(line)
    print( 'Lines in Logfiles after cleaning: ' , len(lines) )                             


n = len(lines)
print('n = ', n)    

model = Nvidea_Model('relu', 0.3)

# print a mopdel summary for documentationmodel.fit(
# model.summary()

# standard learning rate is 0.001, tried to 0.0001, changed back
#model.compile(loss='mse',  optimizer = Adam(lr=1e-4) ) #optimizer='adam')
model.compile(optimizer='adam', loss='mse')

train_samples, validation_samples = train_test_split(lines, test_size=0.1)

shuffle(train_samples)
shuffle(validation_samples)

print( 'Len Train Samples: ',len(train_samples) )

train_generator = data_generator(train_samples, batchsize, steeringAngleCorrection,'TRAIN', dataPath)
validation_generator = data_generator(validation_samples, batchsize, steeringAngleCorrection,'VALID', dataPath)

# When you use fit_generator, the number of samples processed for each epoch is batch_size * steps_per_epochs. 
stepsPerEpoch =  math.ceil(len(train_samples)/batchsize )
print ('Steps per Epoch: ' , stepsPerEpoch)
validSteps =   math.ceil( len(validation_samples)/batchsize  )
print ('Validation Steps: ' , validSteps)
 
history_object = model.fit_generator(
   generator = train_generator, 
   steps_per_epoch = stepsPerEpoch,
   validation_data = validation_generator,
   validation_steps = validSteps, 
   epochs=epochs, 
   verbose=1)

saveModelAndWeights(modelfilename)

# plot only local, not on AWS EC2
if os.name == 'nt':
    plotResult(history_object)

timer =  time.time() - start_time 
print (  "Run time in seconds: %5.0f" %  timer)

# Explicitly end tensorflow session
from keras import backend as K 
K.clear_session()

