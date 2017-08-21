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
from keras.models import Sequential, model_from_json, load_model 
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D,  Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.backend import tf as ktf
 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


######### define Constants #############

modelfilename = "model"

#dataPath = './data/'
dataPath = './dataps3/'
logfile = 'driving_log.csv'

steeringAngleCorrection = [0, 0.20, -0.20]

imgsizeX = 320
imgsizeY = 160

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
    #filename = modelfilename+'.json'
    #newfilename = filename + '_' + timeStamp
    #copyfile(filename, newfilename)
    print('Model backup finished.', newfilename)

def saveModelAndWeights(modelfilename):
    # Save model and weights, serialize model to JSON
   # model_json = model.to_json()
   # with open(modelfilename + ".json", "w") as json_file:
   #     json_file.write(model_json)
   #     print("JSON Model saved.")
    # serialize weights to HDF5
    filename = modelfilename + '.h5'
    #model.save_weights(filename)
    model.save(filename)
    #print("Weights saved to ", filename)
    print("Model saved to ", filename)
    return

def loadModelAndWeights(modelfilename):
    # load json and create model
    #filename = modelfilename + '.json'
    #json_file = open(filename, 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # load weights into new model.
    filename = modelfilename + '.h5'
    #loaded_model.load_weights(filename)
    loaded_model = load_model('model.h5')
    print("Loaded model %s from disk" % filename)
    return loaded_model    

def showImage(img):
    window_width = img.shape[1] 
    window_height = img.shape[0]
    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)
    cv2.imshow('dst_rt', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# for smaple log Linux
def getImage( dataDir, source_path ):
    image_sizeX = 160
    image_sizeY = 80
    filename = dataDir + source_path.split('\\')[-1]
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # crops 60 pixels from top, 25 pixels from bottom, 0 pixels from left, and 0 pixels from right 
    image = image[60:140, 0:320] # Crop from x, y, w, h -> 100, 200, 300, 400 
    image = cv2.resize(image, (image_sizeX, image_sizeY))
    
    
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]    
    return image

model = Sequential()
# Prepocesssing    
# this line made problems when running a AWS model.h5 on a local windows machine
#crops 60 pixels from top, 25 pixels from bottom, 0 pixels from left, and 0 pixels from right    
#model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
#model.add(Lambda(resize_normalize, input_shape=(160,320,3), output_shape=(80,160,3)))     

#model.add(Lambda(lambda image: ktf.image.resize_images(image, (80, 160)), input_shape=(160,320,3)))
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80,160,3),output_shape=(80,160,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80,160,3)) )

#https://stackoverflow.com/questions/44620582/nameerror-when-opening-keras-model-that-uses-tensorflow-backend
#model.add( Lambda(lambda image: ktf.image.resize_images(image, (80, 160,3) )))
     
# Model Layers
model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
#return model

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
                for camind in range(0,3):    #0,1,2                             
                    adaptedMeasurement = measurement + steeringAngleCorrection[camind]                                 
                    measurements.append(adaptedMeasurement)
                    image = getImage(dataDir + 'IMG/', batch_sample[camind])
                    images.append( img_to_array(image) )   
                    # added flipped image and steering angle
                    imageFlipped = cv2.flip(image, 1)                    
                    images.append( img_to_array( imageFlipped ))
                    measurements.append( adaptedMeasurement * (-1) )
                #else:
                #    measurements.append(measurement)
                #    image = cv2.imread(batch_sample[0])  # 0  = center image
                #    images.append( img_to_array(image) )   
                    
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)
            

###################################
########## MAIN PROGRAM ###########
###################################
    
if os.path.isfile(modelfilename+'.h5') and 1==2:
    modelmode = 'TUNING'
    fileBackup(modelfilename)
    model = loadModelAndWeights(modelfilename)
else:
    modelmode = 'NEW MODEL'

    
print('Model mode %s: ' % modelmode)
print("Batch size: ", batchsize)
print("Epochs:", epochs)
print("------------")

try:    
    # Import images from logfiles      
    with open(dataPath + logfile ) as csvfile:
        print(dataPath + logfile)
        reader = csv.reader(csvfile)
        #skip headerline
        #next(reader, None)        
        for line in reader:          
            # 2/3 of all data is straigt with angle 0.
            # To prevent a bias for straight 2 of 3 rows with angle 0 are dropped. Later it's flipped and though doubled again     
            # sort out extreme steering angles
            if ( float(line[3]) != 0.0 and float(line[3]) < 30.0  and float(line[3]) > -30.0 ) or  ( float(line[3]) == 0.0 and random.randint(0,3) == 1):
                lines.append(line)
except Exception:
     file = dataPath + logfile
     print('%s not found' % file)
     sys.exit()


n = len(lines)
print('n = ', n)    
lines=lines[0:200]
    
# print a mopdel summary for documentationmodel.fit(
#model.summary()
# standard learning rate is 0.001, changed to 0.0001
model.compile(loss='mse',  optimizer = Adam(lr=1e-4) ) #optimizer='adam')  , s

train_samples, validation_samples = train_test_split(lines, test_size=0.1)

shuffle(train_samples)
shuffle(validation_samples)

print( 'Len Train Samples: ',len(train_samples) )

train_generator = data_generator(train_samples, batchsize, steeringAngleCorrection,'TRAIN', dataPath)
validation_generator = data_generator(validation_samples, batchsize, steeringAngleCorrection,'VALID', dataPath)

#When you use fit_generator, the number of samples processed for each epoch is batch_size * steps_per_epochs. 
# factor 6 because of 3 cameras and each is added flipped
stepsPerEpoch =  len(train_samples)/batchsize 
print ('Steps per Epoch: ' , stepsPerEpoch)
validSteps =   len(validation_samples)/batchsize 
print ('Validation Steps: ' , validSteps)

 
history_object = model.fit_generator(
   generator = train_generator, 
   steps_per_epoch = stepsPerEpoch,
   validation_data = validation_generator,
   validation_steps = validSteps, 
   epochs=epochs, 
   #use_multiprocessing:=True
   verbose=1)

saveModelAndWeights(modelfilename)

#scores = model.evaluate(X_train, y_train, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print (scores)

# print the keys contained in the history object
# print(hist.history.keys())
# plot only local, not on AWS EC2
if os.name == 'nt':
    plotResult(history_object)

print ( time.time() - start_time, "seconds run time" )

import gc; gc.collect()

# TODO
# aws grössere machine ordern, alte löschen, - h5 Format ??? vorher testen
# brightnesss, camera,  tf.image.random_brightness 
 #