######## first import all modules and functions  ###########
import csv
import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt
from shutil import copyfile
from datetime import datetime
from keras.models import Sequential, model_from_json
from keras.layers import Flatten, Dense, Conv2D, Lambda#, Cropping2D
from keras.layers import MaxPooling2D, Dropout  # used for LeNet5 only

######### define Constants #############

modelfilename = "model"

#dataPath = './data/mydata/';
#logfile = 'predictiontest.csv' # 'driving_log.csv'

dataPath = './data/';
logfile = 'driving_log.csv'

steeringAngleCorrection = [0, 0.08, -0.08]

imgsizeX = 320
imgsizeY = 160

myEpochs=1
myBatchsize = 128 # 64 faster  # 32 default

######## innitialize variables ########
lines = []
images = []
measurements = []
flipcount=0


######## define Functions ###########

def fileBackup(modelfilename):
    filename = modelfilename+'.h5'
    modifiedTime = os.path.getmtime(filename) 
    timeStamp =  datetime.fromtimestamp(modifiedTime).strftime("%y-%m-%d-%H-%M-%S")
    newfilename = filename + '_' + timeStamp
    copyfile(filename, newfilename)
    filename = modelfilename+'.json'
    newfilename = filename + '_' + timeStamp
    copyfile(filename, newfilename)
    print('Model backup finished.', newfilename)

def saveModelAndWeights(modelfilename):
    # Save model and weights, serialize model to JSON
    model_json = model.to_json()
    with open(modelfilename + ".json", "w") as json_file:
        json_file.write(model_json)
        print("JSON Model saved.")
    # serialize weights to HDF5
    filename = modelfilename + '.h5'
    model.save_weights(filename)
    print("Weights saved to ", filename)
    return

def loadModelAndWeights(modelfilename):
    # load json and create model
    filename = modelfilename + '.json'
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    filename = modelfilename + '.h5'
    loaded_model.load_weights(filename)
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
    filename = source_path.split('/')[-1] # linux
    # filename = source_path.split('\\               ')[-1] # windows
    current_path = dataPath +'IMG/' + filename
    image = cv2.imread(current_path)
    return image
    # augment data with flipped images

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
    
print('Running in %s mode.' % modelmode)
    
# Import images from logfiles
with open(dataPath + logfile ) as csvfile:
    reader = csv.reader(csvfile)
    #skip headerline
    next(reader, None)
    for line in reader:
        lines.append(line)

n = len(lines)
# take 100 random images only to spped up first test
print('n = ', n)
newlines = np.copy(lines)
#random.shuffle(newlines)
randlines = newlines[:n_rand]

lines= lines[0:100]
for line in lines:
    measurement = float(line[3])    
    # center camera = 0, left = 1, right = 2
    for camind in range(0,3):
        path = line[camind]
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


# print a mopdel summary for documentation
#model.summary()
model.compile(loss='mse', optimizer='adam')
    
hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=myBatchsize, epochs=myEpochs)
#scores = model.evaluate(X_train, y_train, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print (scores)

saveModelAndWeights(modelfilename)

# print the keys contained in the history object
# print(hist.history.keys())

plotResult(hist)

# TODO
# Cropping checken
# images verkleinern


		

