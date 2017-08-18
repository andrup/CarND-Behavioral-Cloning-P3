######## first import all modules and functions  ###########
import math
import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from shutil import copyfile
from datetime import datetime
from keras.models import Sequential, model_from_json, load_model 
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D
from keras.layers import MaxPooling2D, Dropout  # used for LeNet5 only
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
######### define Constants #############

modelfilename = "model"
runmode = 'ALLATONCE'
#runmode = 'GENERATOR'

#dataPath = './owndata2laps/'
#dataPath = './owndata1lapback/'
dataPath = './data/';
logfile = 'driving_log.csv'

steeringAngleCorrection = [0, 0.2, -0.2]

imgsizeX = 320
imgsizeY = 160

epochs = 2
batchsize = 128 # 64 faster  # 32 default

######## innitialize variables ########
lines = []
images = []
measurements = []
flipcount=0


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
    # load weights into new model
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

def getImage( source_path ):
    if os.name == 'nt':
        current_path = source_path
    else:
        filename = source_path.split('/')[-1] # linux      
        current_path = dataPath +'IMG/' + filename
    image = cv2.imread(current_path)
    return image

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

def resize_normalize(image):
    import cv2
    from keras.backend import tf as ktf   
    
    # resize to width 200 and high 66 liek recommended
    # in the nvidia paper for the used CNN
    # image = cv2.resize(image, (66, 200)) #first try
    #resized = ktf.image.resize_images(image, (32, 128))
    #normalize 0-1
    resized = resized/255.0 - 0.5

    return resized

def NvidiaModel():
    model = Sequential()
    # Prepocesssing
    model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))

    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    #model.add(Lambda(resize_normalize, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
    
    #crops 60 pixels from top, 25 pixels from bottom, 0 pixels from left, and 0 pixels from right
    #model.add(Lambda(resize_normalize, input_shape=(160, 320, 3), output_shape=(32, 128, 3)))

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
    return model

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

def data_generator(data, batch_size, modus):
    num_samples = len(data)
    print('modus %s'  %modus)
    print('data len:' ,num_samples )    
    steeringAngleCorrection = [0, 0.08, -0.08]
   
    while 1:  # endless loop
        #shuffle(data)
        for offset in range(0, num_samples, batch_size):  # loop over batches
            print('offset ',offset )
            print('batch_size', batch_size)
            batch_samples = data[offset : (offset+batch_size)]
            
            images = []
            measurements = []
            
            i=0
            for i in range(0,len(batch_samples)):  # loop over samples in batch    
            #for batch_sample in batch_samples:
            #for i, batch_sample in batch_samples.iterrows():            
                print('i ' , i)
                measurement = float(lines[i][3])                       
                for camind in range(0,1):    #0,3          
                    print('camind ' , camind)
                    adaptedMeasurement = measurement + steeringAngleCorrection[camind]                                 
                    measurements.append(adaptedMeasurement)
                    image = getImage(lines[i][1])
                    images.append( image )  
                #i+=1
                   # if measurement != 0.0 & modus == 'TRAIN':
                   #     images.append(cv2.flip(image, 1))
                   #     measurements.append( adaptedMeasurement * (-1) )
                   #     flipcount+=1
                    
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield (X_train, y_train)
            


###################################
########## MAIN PROGRAM ###########
###################################
    
if os.path.isfile(modelfilename+'.h5'):
    modelmode = 'TUNING'
    fileBackup(modelfilename)
    model = loadModelAndWeights(modelfilename)
else:
    modelmode = 'NEW MODEL'
    #model = LeNet5Model()
    model = NvidiaModel()
    
print('Model mode %s: ' % modelmode)
print("Run Mode: ", runmode)
print("Batch size: ", batchsize)
print("Epochs:", epochs)
print("------------")
    
# Import images from logfiles
with open(dataPath + logfile ) as csvfile:
    reader = csv.reader(csvfile)
    #skip headerline
    next(reader, None)
    for line in reader:
        lines.append(line)

# take 100 random images only to spped up first test

newlines = np.copy(lines)
#random.shuffle(newlines)
#randlines = newlines[:n_rand]

if runmode == 'ALLATONCE':
   # lines=lines[0:100]
    n = len(lines)
    print('n = ', n)    
    flipcount = 0
    for i in range(0,n):
       # print(i)
        measurement = float(lines[i][3])    
        # center camera = 0, left = 1, right = 2
        for camind in range(0,3):
            path = lines[i][camind]        
            image = getImage(path)
            images.append( image )
            adaptedMeasurement = measurement + steeringAngleCorrection[camind]         
            measurements.append( adaptedMeasurement)
            if measurement != 0.0:
                images.append(cv2.flip(image, 1))
                measurements.append( adaptedMeasurement * (-1) )
                flipcount+=1      
    X_train = np.array(images)
    y_train = np.array(measurements)	
    print('Added %d flipped images: ' %flipcount)

# print a mopdel summary for documentationmodel.fit(
#model.summary()
model.compile(loss='mse',  optimizer = Adam(lr=1e-4) ) #optimizer='adam')  

print(len(X_train))
if runmode == 'ALLATONCE':    
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=batchsize, epochs=epochs)
else:    
    # Generator mode if memory problems occur
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    shuffle(train_samples)
    shuffle(validation_samples)
    
    gen_train = data_generator(train_samples, 1, 'TRAIN')
    gen_valid = data_generator(validation_samples, 1, 'VALID')
        #When you use fit_generator, the number of samples processed for each epoch is batch_size * steps_per_epochs. 
    
    history_object = model.fit_generator(
        generator = gen_train, 
        steps_per_epoch= 1,#math.ceil( len(train_samples)/batchsize ), #calc_samples_per_epoch(len(train_samples), batchsize),
        validation_data=gen_valid,
        validation_steps=1,#math.ceil(len(validation_samples)/batchsize), #calc_samples_per_epoch(len(validation_samples), batchsize) 
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

# TODO
# Cropping checken
# images verkleinern

