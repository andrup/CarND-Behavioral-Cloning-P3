from keras.models import Sequential
import cv2
import h5py

images=[]
x_data=[]

# IMG/center_2016_12_01_13_31_12_937.jpg, IMG/left_2016_12_01_13_31_12_937.jpg, IMG/right_2016_12_01_13_31_12_937.jpg, 0, 0, 0, 1.453011
# IMG/center_2016_12_01_13_32_45_679.jpg, IMG/left_2016_12_01_13_32_45_679.jpg, IMG/right_2016_12_01_13_32_45_679.jpg, 0.1765823, 0.9855326, 0, 25.34023
# IMG/center_2016_12_01_13_32_52_753.jpg, IMG/left_2016_12_01_13_32_52_753.jpg, IMG/right_2016_12_01_13_32_52_753.jpg, -0.0787459, 0.9855326, 0, 30.18809


def showImage(img):

    window_width = img.shape[1] 
    window_height = img.shape[0]
    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)
    cv2.imshow('dst_rt', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

filename = './data/IMG/center_2016_12_01_13_31_12_937.jpg' # zero 0
#filename = './data/IMG/center_2016_12_01_13_32_45_679.jpg' # positiv 0.17
filename = './data/IMG/center_2016_12_01_13_32_58_923.jpg' # right 0.9

image = cv2.imread(filename)
showImage(image)
images.append(image)

x_data = np.array(images)

with h5py.File('model.h5', 'r') as f:   
    prediction = model.predict(x_data)
    
    #prediction = model.predict(np.array(tk.texts_to_sequences(text)))
    print('Prediction for %s : angle: %0.4f' %( filename, prediction))
    