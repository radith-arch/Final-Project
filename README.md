## Final Project - Using Architecture Vgg16 (Acc 0.79) Vs 5 Layers Conv2D (Acc 0.86) to Predict Invasive Ductal Carcinoma (IDC)

### Table of Contents
- [Description of Dataset](#description-of-dataset)
	- [Context](#context)
	- [Content](#content)
	- [Acknowledgements](#acknowledgements)
	- [Inspiration](#inspiration)
- [Preparation Dataset](#preparation-dataset)
	- [Load Data](#load-data)
	- [Split Train-Test Data](split-train-test-data)
	- [Feature Engineering](#feature-engineering)
- [Images Visualization](#images-visualization)
- [Architecture Convolutional Neural Network (CNN)](#architecture-convolutional-neural-network)
	- [VGG16](#vgg16)
		- [Visualize Filters in VGG16](#visualize-filters-in-vgg16)
		- [Visualize Feature Maps](#visualize-feature-maps)
		- [Tensorboard VGG16](#tensorboard-vgg16)
		- [Visualize Images Augmentation](#visualize-images-augmentation)
		- [Visualisasi Tensorboard Epoch to Loss-Accuracy](#visualisasi-tensorboard-epoch-to-loss-accuracy)
	- [Sequential 5 Layers Conv2D](#sequential-5-layers-conv2d)
		- [Visualize Filters in Sequential 5 Layers Conv2D](#visualize-filters-in-sequential-5-layers-conv2d)
		- [Visualize Feature Maps](#visualize-feature-maps)
		- [Tensorboard Sequential 5 Layers Conv2D](#tensorboard-sequential-5-layers-conv2d)
		- [Visualisasi Tensorboard Epoch to Loss-Accuracy](#visualisasi-tensorboard-epoch-to-loss-accuracy)

### Description of Dataset
#### Context
Invasive Ductal Carcinoma (IDC) is the most common subtype of all breast cancers. To assign an aggressiveness grade to a whole mount sample, pathologists typically focus on the regions which contain the IDC. As a result, one of the common pre-processing steps for automatic aggressiveness grading is to delineate the exact regions of IDC inside of a whole mount slide.

#### Content
The original dataset consisted of 162 whole mount slide images of Breast Cancer (BCa) specimens scanned at 40x. From that, 277,524 patches of size 50 x 50 were extracted (198,738 IDC negative and 78,786 IDC positive). Each patch’s file name is of the format: uxXyYclassC.png — > example 10253idx5x1351y1101class0.png . Where u is the patient ID (10253idx5), X is the x-coordinate of where this patch was cropped from, Y is the y-coordinate of where this patch was cropped from, and C indicates the class where 0 is non-IDC and 1 is IDC.

#### Acknowledgements
The original files are located here [(download)](http://gleason.case.edu/webdata/jpi-dl-tutorial/IDC_regular_ps50_idx5.zip).
Citation: [National Library of Medicine](https://www.ncbi.nlm.nih.gov/pubmed/27563488) and [SPIE (The International Society for Optics and Photonics)](http://spie.org/Publications/Proceedings/Paper/10.1117/12.2043872).

#### Inspiration
Breast cancer is the most common form of cancer in women, and invasive ductal carcinoma (IDC) is the most common form of breast cancer. Accurately identifying and categorizing breast cancer subtypes is an important clinical task, and automated methods can be used to save time and reduce error.
[Source](https://www.kaggle.com/paultimothymooney/breast-histopathology-images)

### Preparation Dataset

#### Load Data
```
zip_path = '/content/drive/My\ Drive/Datasets/Breast\ Histopathology\ Images.zip'

!cp {zip_path} /content/

!cd /content/

!unzip -q /content/Breast\ Histopathology\ Images.zip -d /content

!rm /content/Breast\ Histopathology\ Images.zip
```

Use glob() for input all of datas to 1 varibable.
```
import numpy as np
import os 
import matplotlib.pyplot as plt
import glob

all_image_path = glob.glob('/content/IDC_regular_ps50_idx5/**/*.png', recursive=True)

for filename in all_image_path[0:2]:
  print(filename)
for filename in all_image_path[-3:-1]:
  print(filename)
```

Use fnmatch to seperate data into 2 variable, with each parts is 0 and 1. This part is to calculate how much data for IDC(-) and IDC(+).
```
import fnmatch

zero = '*class0.png'
one = '*class1.png'

nonIDC = fnmatch.filter(all_image_path, zero)
IDC = fnmatch.filter(all_image_path, one)

print("Data Gejala Invasive Ductal Carcinoma (IDC)")
print("Jumlah Data Gejala IDC(-): ", len(nonIDC))
print("Jumlah Data Gejala IDC(+): ", len(IDC))

patient_path = '/content/IDC_regular_ps50_idx5'
patient = os.listdir(patient_path)
print("Jumlah Patient: ", len(patient))

print("Total Gejala IDC(-) dan IDC(+) :", len(nonIDC)+len(IDC))
```

#### Split Train-Test Data

Made a function for splitting data.
```
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(files, lower_limit, upper_limit):
    x = []
    y = []
    for file in files[lower_limit:upper_limit]:
        if file.endswith(".png"):
            img = load_img(file, target_size = (50,50))
            pixels = img_to_array(img)
            pixels /= 255
            x.append(pixels)
            if(file[-5] == '1'):
                y.append(1)
            elif(file[-5] == '0'):
                y.append(0)
    return np.stack(x), y
```
Split data train : test = 90,000 : 20,000.
```
x_train,y_train = load_data(all_image_path,0, 90000)
x_test, y_test = load_data(all_image_path, 90000, 110000)
```
Check data distribution.
```
import seaborn as sns
sns.countplot(y_train)
```
![sns countplot(y_train)](https://user-images.githubusercontent.com/72731175/95765136-9c06e200-0cdb-11eb-94ee-5ad2671e6281.jpeg)
```
sns.countplot(y_test)
```
![sns countplot(y_test)](https://user-images.githubusercontent.com/72731175/95765247-c789cc80-0cdb-11eb-96b9-9c1f41755428.jpeg)

From the picture above, we could see that the data is imbalance. To solve the issue, we could use function split test same like before but now we add another variable (counter). It's function to make sure that data between 0 and 1 is balance.
```
def load_balanced_data(files, size, start_index):
    half_size = int(size/2)
    count=0
    res = []
    y = []
    for file in files[start_index:]:
        if (count!=half_size):
            if file[-5] == '1' and file.endswith(".png"):
                img = load_img(file, target_size = (50,50))
                pixels = img_to_array(img)
                pixels /= 255
                res.append(pixels)
                y.append(1)
                count += 1
                
    for file in files[start_index:]:
        if(count!=0):
            if(file[-5] == '0'):
                img = load_img(file, target_size = (50,50))
                pixels = img_to_array(img)
                pixels /= 255
                res.append(pixels)
                y.append(0)
                count -= 1
    return np.stack(res), y
```
Split data train again, and now after splitting, data train 0 : data train 1 = 45,000 : 45,000.
```
x_train2, y_train2 = load_balanced_data(all_image_path, 90000,0)
```
Check data train distribution

```
sns.countplot(y_train2)
```
![sns countplot(y_train2)](https://user-images.githubusercontent.com/72731175/95766178-16843180-0cdd-11eb-8f99-f42317d92fd1.jpeg)

And for data test, after splitting, data test 0 : data test 1 = 10,000 : 10,000.
```
x_test2, y_test2 = load_balanced_data(all_image_path, 20000, 110000)
```
Check data test distribution.
```
sns.countplot(y_test2)
```
![sns countplot(y_test2)](https://user-images.githubusercontent.com/72731175/95766347-58ad7300-0cdd-11eb-965e-4361c34456ae.jpeg)

#### Feature Engineering

Made label become numpy array with type uint8.
```
y_train2_arr = np.array(y_train2).astype('uint8')
y_test2_arr = np.array(y_test2).astype('uint8')
```
After that encode label with tf.keras.utils.to_categorical.
```
from tensorflow.keras.utils import to_categorical

y_train2_arr_cat = to_categorical(y_train2_arr)
y_test2_arr_cat = to_categorical(y_test2_arr)
```

### Images Visualization

Made a new function for pick random sample and show it using matplotlib.pyplot.
```
def show_img(files):
    plt.figure(figsize= (10,10))
    ind = np.random.randint(0, len(files), 25)
    i=0
    for loc in ind:
        plt.subplot(5,5,i+1)
        sample = load_img(files[loc], target_size=(50,50))
        sample = img_to_array(sample)
        plt.axis("off")
        plt.imshow(sample.astype("uint8"))
        i+=1
```
Random Sample Images from IDC(-) and IDC(+).
```
show_img(all_image_path)
```
![show_img(all_image_path)](https://user-images.githubusercontent.com/72731175/95767362-d58d1c80-0cde-11eb-9fa8-f41ed1343ac0.jpeg)

Random Sample Images from IDC(+).
```
show_img(nonIDC)
```
![show_img(IDC)](https://user-images.githubusercontent.com/72731175/95767394-e2117500-0cde-11eb-8204-25a3c5d8f9bf.jpeg)

Random Sample Images from IDC(-).
```
show_img(IDC)
```
![show_img(nonIDC)](https://user-images.githubusercontent.com/72731175/95767418-edfd3700-0cde-11eb-9bf4-974ac924aaea.jpeg)

### Architecture Convolutional Neural Network (CNN)

As same like the title in project, we would use two models architecture CNN to predict IDC so we could see what the difference between two architectures. The first model we used is VGG16.

#### VGG16
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD

vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(50, 50, 3))

for layer in vgg_conv.layers[:]:
  layer.trainable = False

vgg_model = Sequential()

vgg_model.add(vgg_conv)

vgg_model.add(Flatten())
vgg_model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
vgg_model.add(Dropout(0.5))
vgg_model.add(Dense(2, activation='softmax'))

opt = SGD(lr=0.001, momentum = 0.9)

vgg_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

vgg_model.summary()
```


####  Visualize Filters in VGG16

```
filters, biases = vgg_conv.layers[1].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 6, 1
for i in range(n_filters):
	f = filters[:, :, :, i]
	for j in range(3):
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(f[:, :, j], cmap='gray')
		ix += 1

plt.show()
```

Plot of the First 6 Filters From CNN With One Subplot per Channel.



####  Visualize Feature Maps 

```
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from numpy import expand_dims

model_baru = vgg_conv
model_baru = Model(inputs=model_baru.inputs, outputs=model_baru.layers[1].output)
model_baru.summary()

img = load_img('/content/drive/My Drive/Datasets/8863_idx5_x1201_y751_class1.png', target_size=(50, 50))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)
feature_maps = model_baru.predict(img)

square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		ax = plt.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1

plt.show()
```
Visualization of the Feature Maps Extracted From the First Convolutional Layer in CNN.




#### Tensorboard VGG16

Using tf.keras.callbacks.ModelCheckpoint to save weight of the best validation accuracy. 
```
from tensorflow.keras.callbacks import ModelCheckpoint

filepath = "vgg-model-weights-improvement-the-best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list_vgg = [checkpoint]
```
Made new variable as place to save logs from tf.keras.callbacks.ModelCheckpoint.
```
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime

vgg_logdir = os.path.join("logs-vgg-model", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks_list_vgg.append(TensorBoard(vgg_logdir, histogram_freq=1))
```
To ensure that model could perform better, we could use image generator to mutiple data as to increase the probability.
```
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen = ImageDataGenerator(rescale=None)
```
Use flow() to takes data & label arrays, generates batches of augmented data as x is a list of numpy arrays and y is a numpy array of corresponding labels.
```
train_iterator = train_datagen.flow(x_train2, y_train2_arr_cat, batch_size=128)
test_iterator = datagen.flow(x_test2, y_test2_arr_cat, batch_size=128)
```
Fit the model using fit_generator as we have data augmentation needs to be applied and keep verbose=2, so we still could see the progress of training manually.
```
vgg_model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), validation_data=test_iterator, validation_steps=len(test_iterator), epochs=25,  callbacks=callbacks_list_vgg, verbose=2)
```

#### Visualize Images Augmentation

```
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

img = load_img('/content/drive/My Drive/Datasets/8863_idx5_x1201_y751_class1.png')
data = img_to_array(img)
samples = expand_dims(data, 0)
it = train_datagen.flow(samples, batch_size=1)

for i in range(9):
	plt.subplot(330 + 1 + i)
	batch = it.next()
	image = batch[0].astype('uint8')
	plt.imshow(image)

plt.show()
```
Plot of Augmented Generated With a Width Shift Range, Height Shift Range, Horizontal Flip.




#### Visualisasi Tensorboard Epoch to Loss-Accuracy

```
%load_ext tensorboard
```
We could use feature tensorboard to visualize training process, so as it goes to training, we could see the progress of epoch to loss and accuracy.
```
%tensorboard --vgg_logdir logs-vgg-model
```





Prediction to validation data using the saved weight from last time. We could call it by code load_weights() and it's must noted that opt and loss must same as architecture CNN we used for training.
```
model_filename = "vgg-model-weights-improvement-the-best.h5"

vgg_model.load_weights(model_filename)

opt = SGD(lr=0.001, momentum = 0.9)
vgg_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
```

To evaluate the model which one is good or not. We could use confusion matrix and classification report. 
```
from sklearn.metrics import confusion_matrix
import numpy as np

pred = vgg_model.predict(x_test2)
prediction_result = np.argmax(pred, axis=1)
confusion = confusion_matrix(y_test2_arr, prediction_result)

from sklearn.metrics import classification_report
report = classification_report(y_test2_arr, prediction_result)
report
```


Confusion Matrix Visualization
```
import matplotlib.pyplot as plt

plt.figure(figsize=(5,3))
sns.set(font_scale=1.2)
ax = sns.heatmap(confusion, annot=True, xticklabels=['Non-IDC', 'IDC'], yticklabels=['Non-IDC', 'IDC'], cbar=False, cmap='Blues', linewidths=1, linecolor='black', fmt='.0f')
plt.yticks(rotation=0)
plt.xlabel('Predicted')
plt.ylabel('Actual')
ax.xaxis.set_ticks_position('top')
plt.title('Confusion Matrix') 
plt.show()
```

#### Sequential 5 Layers Conv2D
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same", input_shape=(50, 50, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))
model.add(Flatten()) 

model.add(Dense(128, activation='relu', kernel_initializer='he_uniform')) 
model.add(Dense(2, activation='softmax'))

opt = SGD(lr=0.001, momentum = 0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
```


####  Visualize Filters in Sequential 5 Layers Conv2D

```
filters, biases = model.layers[0].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 6, 1
for i in range(n_filters):
	f = filters[:, :, :, i]
	for j in range(3):
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(f[:, :, j], cmap='gray')
		ix += 1

plt.show()
```

Plot of the First 6 Filters From CNN With One Subplot per Channel.



####  Visualize Feature Maps 

```
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from numpy import expand_dims

model_baru = model
model_baru = Model(inputs=model_baru.inputs, outputs=model_baru.layers[1].output)
model_baru.summary()

img = load_img('/content/drive/My Drive/Datasets/8863_idx5_x1201_y751_class1.png', target_size=(50, 50))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)
feature_maps = model_baru.predict(img)

square = 5
ix = 1
for _ in range(square):
	for _ in range(square):
		ax = plt.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
plt.show()
```
Visualization of the Feature Maps Extracted From the First Convolutional Layer in CNN.




#### Tensorboard Sequential 5 Layers Conv2D

Using tf.keras.callbacks.ModelCheckpoint to save weight of the best validation accuracy. 
```
from tensorflow.keras.callbacks import ModelCheckpoint

filepath = "weights-improvement-the-best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
```
Made new variable as place to save logs from tf.keras.callbacks.ModelCheckpoint.
```
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime

logdir = os.path.join("logs-deep-model", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks_list.append(TensorBoard(logdir, histogram_freq=1))
```
To ensure that model could perform better, we could use image generator to mutiple data as to increase the probability.
```
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen = ImageDataGenerator(rescale=None)
```
Use flow() to takes data & label arrays, generates batches of augmented data as x is a list of numpy arrays and y is a numpy array of corresponding labels.
```
train_iterator = train_datagen.flow(x_train2, y_train2_arr_cat, batch_size=128)
test_iterator = datagen.flow(x_test2, y_test2_arr_cat, batch_size=128)
```
Fit the model using fit_generator as we have data augmentation needs to be applied and keep verbose=2, so we still could see the progress of training manually.
```
model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), validation_data=test_iterator, validation_steps=len(test_iterator), epochs=25,  callbacks=callbacks_list, verbose=2)
```

#### Visualisasi Tensorboard Epoch to Loss-Accuracy

```
%load_ext tensorboard
```
We could use feature tensorboard to visualize training process, so as it goes to training, we could see the progress of epoch to loss and accuracy.
```
%tensorboard  --logdir logs-deep-model
```





Prediction to validation data using the saved weight from last time. We could call it by code load_weights() and it's must noted that opt and loss must same as architecture CNN we used for training.
```
model_filename = "weights-improvement-the-best.h5"

model.load_weights(model_filename)

opt = SGD(lr=0.001, momentum = 0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
```

To evaluate the model which one is good or not. We could use confusion matrix and classification report. 
```
from sklearn.metrics import confusion_matrix
import numpy as np

pred = model.predict(x_test2)
prediction_result = np.argmax(pred, axis=1)
confusion = confusion_matrix(y_test2_arr, prediction_result)

from sklearn.metrics import classification_report
report = classification_report(y_test2_arr, prediction_result)
report
```


Confusion Matrix Visualization
```
import matplotlib.pyplot as plt

plt.figure(figsize=(5,3))
sns.set(font_scale=1.2)
ax = sns.heatmap(confusion, annot=True, xticklabels=['Non-IDC', 'IDC'], yticklabels=['Non-IDC', 'IDC'], cbar=False, cmap='Blues', linewidths=1, linecolor='black', fmt='.0f')
plt.yticks(rotation=0)
plt.xlabel('Predicted')
plt.ylabel('Actual')
ax.xaxis.set_ticks_position('top')
plt.title('Confusion Matrix') 
plt.show()
```


