import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_steering_distri(data, name):
	fig = plt.figure()
	bins = np.linspace(-1, 1, 21)
	plt.xlabel('Steering Angles')
	plt.ylabel('Number')
	plt.hist(data, bins, color='r', alpha=0.6, label='Steering Angle')
	plt.legend()
	fig.savefig(name)

def miscellaneous_plot(sample, directory):
	name = './data/IMG/' + sample[0].split('/')[-1]
	center_img = cv2.imread(name)
	cv2.imwrite(directory+'center.png', center_img)
	cv2.imwrite(directory+'center_flip.png', cv2.flip(center_img, 1))

	name = './data/IMG/' + sample[1].split('/')[-1]
	left_img = cv2.imread(name)
	cv2.imwrite(directory+'left.png', left_img)
	name = './data/IMG/' + sample[2].split('/')[-1]
	right_img = cv2.imread(name)
	cv2.imwrite(directory+'right.png', right_img)

	cv2.imwrite(directory+'crop.png', center_img[51:139,:,:])
	cv2.imwrite(directory+'resize.png', cv2.resize(center_img[51:139,:,:], (0,0), fx=0.5, fy=0.5))
line = ["IMG/center_2016_12_01_13_31_14_194.jpg", "IMG/left_2016_12_01_13_31_14_194.jpg",
        "IMG/right_2016_12_01_13_31_14_194.jpg", 0, 0, 0, 1.279884]
miscellaneous_plot(line, "photo/")

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

from sklearn.model_selection import train_test_split
train_lines, validation_lines = train_test_split(lines, test_size=0.25)

from sklearn.utils import shuffle

before_filter_steer_hist = []
after_filter_steer_hist = []
steer_bins = np.linspace(-1, 1, 21)

def data_gen(samples, batch_size, aug):
	num_samples = len(samples)

	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			steering = []
			for batch_sample in batch_samples:
				name = './data/IMG/' + batch_sample[0].split('/')[-1]
				center_image = cv2.imread(name) # BGR format
				center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
				center_steer = float(batch_sample[3])

				if abs(center_steer) >= 0.08:
					images.append(center_image)
					steering.append(center_steer)

				if aug == 1 and abs(center_steer) >= 0.08:
					# Append flipped image
					images.append(cv2.flip(center_image, 1))
					steering.append(center_steer*-1.0)
					
				correction = 0.3
				if aug == 1 and abs(center_steer) >= 0.08:
					# left images
					left_steer = center_steer + correction
					name = './data/IMG/' + batch_sample[1].split('/')[-1]
					left_image = cv2.imread(name)
					left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
					images.append(left_image)
					steering.append(left_steer)
	
				if aug == 1 and abs(center_steer) >= 0.08:
					# right images
					right_steer = center_steer - correction
					name = './data/IMG/' + batch_sample[2].split('/')[-1]
					right_image = cv2.imread(name)
					right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
					images.append(right_image)
					steering.append(right_steer)

				# Accumulate the steering angle
				#before_filter_steer_hist.append((int(np.digitize(center_steer, steer_bins, right=True))-10)/10)

			# Crop image to only see section with road
			X_train = np.array(images)
			y_train = np.array(steering)
			yield shuffle(X_train, y_train)

batch_size = 64
train_generator = data_gen(train_lines, batch_size, 1)
validation_generator = data_gen(validation_lines, batch_size, 0)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import math

def resize_img(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, [44, 160])

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0), input_shape=(160,320,3))) # 160, 320, 3
model.add(Cropping2D(cropping=((51,21), (0,0))))
model.add(Lambda(resize_img))
model.add(Conv2D(12, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(9, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(192)) # 120
#model.add(Dense(192, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01))) this doesn't work.
model.add(Dropout(0.3))
model.add(Dense(96)) # 84
model.add(Dropout(0.1))
model.add(Dense(16)) # 10
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

plot_model(model, to_file='./photo/model_architecture.png', show_shapes=True)

model.fit_generator(train_generator, steps_per_epoch=(math.ceil((len(train_lines)/batch_size))),
					validation_data=validation_generator,
					validation_steps=(math.ceil((len(validation_lines)/batch_size))), epochs=10, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto')])
#model.fit_generator(train_generator, samples_per_epoch=(1)*len(train_lines), nb_epoch=3)

model.save('model.h5')
