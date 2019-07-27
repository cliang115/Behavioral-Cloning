from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda
import numpy as np
import cv2
import csv
from skimage import io, color, exposure, filters, img_as_ubyte
from skimage.util import random_noise
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
lines = []
with open ('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        if line[3] != 0:
            lines.append(line)
            if abs(float(line[3])) >= 0.4:
                lines.append(line)
                if abs(float(line[3])) >= 1:
                    lines.append(line)
                    lines.append(line)

        elif np.random.rand() > 0.92:
            lines.append(line)
def data_generator(data, batch_size = 128):
    while True:
        data = shuffle(data)
        for i in range(0, len(data), batch_size):
            images, angles = [], []
            samples = data[i:i+batch_size]
            for line in samples:
                angle = np.float32(line[3])
                image = cv2.cvtColor(cv2.resize(cv2.imread('./data/IMG/' + line[0].split('/')[-1])[40:140, :], (320, 160)), cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(angle)
                images.append(cv2.cvtColor(cv2.resize(cv2.imread('./data/IMG/' + line[1].split('/')[-1])[40:140, :], (320, 160)), cv2.COLOR_BGR2RGB))
                angles.append(angle + 0.3)
                images.append(cv2.cvtColor(cv2.resize(cv2.imread('./data/IMG/' + line[2].split('/')[-1])[40:140, :], (320, 160)), cv2.COLOR_BGR2RGB))
                angles.append(angle - 0.4)
                images.append(cv2.flip(image, 1))
                angles.append(-angle)
                
            images = np.array(images)
            angles = np.array(angles)
            yield shuffle(images, angles)
train, validation = train_test_split(lines, test_size = 0.1)
model = Sequential()
model.add(Lambda(lambda x:x /255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
#model.add(Dropout(0.3))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
#model.add(Dropout(0.3))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
#model.add(Dropout(0.3))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
#model.add(Dropout(0.3))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.4))
model.add(Dense(60))
model.add(Dropout(0.4))
model.add(Dense(12))
model.add(Dropout(0.4))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(data_generator(train), steps_per_epoch = len(train)//64, epochs = 1, validation_data = data_generator(validation), validation_steps = len(validation)//128)
model.save('model.h5')


            
                                        
                
                
