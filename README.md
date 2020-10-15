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
- [Architecture Convolutional Neural Network (CNN)](#architecture-convolutional-neural-network-cnn)
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
- [Conclusion](#conclusion)

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

Use glob() for input all of datas to 1 variable.
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
![vgg_model summary()](https://user-images.githubusercontent.com/72731175/95825256-c8157800-0d5a-11eb-9bb4-3319ae678eb2.jpeg)


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
Plot of the First 6 Filters From CNN VGG16 With One Subplot per Channel.

![vgg_filter](https://user-images.githubusercontent.com/72731175/95825286-d2377680-0d5a-11eb-9a0e-c1500fb12ad8.jpeg)


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
Visualization of the Feature Maps Extracted From the First Convolutional Layer in CNN VGG16.

![vgg feature maps](https://user-images.githubusercontent.com/72731175/95825410-00b55180-0d5b-11eb-9e92-2c657f8e504a.jpeg)



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

logdir = os.path.join("logs-vgg-model", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks_list_vgg.append(TensorBoard(logdir, histogram_freq=1))
```
To ensure that model could perform better, we could use image generator to mutiple data as to increase the probability.
```
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(height_shift_range=0.2, width_shift_range=0.2, zoom_range=0.2, shear_range=0.2)
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

```
Epoch 1/25

Epoch 00001: val_accuracy improved from -inf to 0.75160, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 114s - loss: 0.5214 - accuracy: 0.7513 - val_loss: 0.5105 - val_accuracy: 0.7516
Epoch 2/25

Epoch 00002: val_accuracy improved from 0.75160 to 0.76650, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 113s - loss: 0.5048 - accuracy: 0.7648 - val_loss: 0.4983 - val_accuracy: 0.7665
Epoch 3/25

Epoch 00003: val_accuracy improved from 0.76650 to 0.76900, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 114s - loss: 0.4961 - accuracy: 0.7719 - val_loss: 0.4944 - val_accuracy: 0.7690
Epoch 4/25

Epoch 00004: val_accuracy improved from 0.76900 to 0.77060, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 115s - loss: 0.4916 - accuracy: 0.7749 - val_loss: 0.4914 - val_accuracy: 0.7706
Epoch 5/25

Epoch 00005: val_accuracy improved from 0.77060 to 0.77315, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 114s - loss: 0.4872 - accuracy: 0.7783 - val_loss: 0.4894 - val_accuracy: 0.7732
Epoch 6/25

Epoch 00006: val_accuracy improved from 0.77315 to 0.77540, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 113s - loss: 0.4854 - accuracy: 0.7797 - val_loss: 0.4835 - val_accuracy: 0.7754
Epoch 7/25

Epoch 00007: val_accuracy improved from 0.77540 to 0.77710, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 114s - loss: 0.4810 - accuracy: 0.7811 - val_loss: 0.4820 - val_accuracy: 0.7771
Epoch 8/25

Epoch 00008: val_accuracy did not improve from 0.77710
704/704 - 115s - loss: 0.4786 - accuracy: 0.7839 - val_loss: 0.4810 - val_accuracy: 0.7753
Epoch 9/25

Epoch 00009: val_accuracy did not improve from 0.77710
704/704 - 115s - loss: 0.4770 - accuracy: 0.7836 - val_loss: 0.4798 - val_accuracy: 0.7757
Epoch 10/25

Epoch 00010: val_accuracy improved from 0.77710 to 0.78140, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 113s - loss: 0.4757 - accuracy: 0.7842 - val_loss: 0.4750 - val_accuracy: 0.7814
Epoch 11/25

Epoch 00011: val_accuracy did not improve from 0.78140
704/704 - 113s - loss: 0.4730 - accuracy: 0.7870 - val_loss: 0.4745 - val_accuracy: 0.7802
Epoch 12/25

Epoch 00012: val_accuracy improved from 0.78140 to 0.78345, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 114s - loss: 0.4719 - accuracy: 0.7876 - val_loss: 0.4723 - val_accuracy: 0.7835
Epoch 13/25

Epoch 00013: val_accuracy did not improve from 0.78345
704/704 - 112s - loss: 0.4697 - accuracy: 0.7883 - val_loss: 0.4746 - val_accuracy: 0.7818
Epoch 14/25

Epoch 00014: val_accuracy did not improve from 0.78345
704/704 - 111s - loss: 0.4682 - accuracy: 0.7886 - val_loss: 0.4726 - val_accuracy: 0.7821
Epoch 15/25

Epoch 00015: val_accuracy did not improve from 0.78345
704/704 - 111s - loss: 0.4670 - accuracy: 0.7898 - val_loss: 0.4710 - val_accuracy: 0.7809
Epoch 16/25

Epoch 00016: val_accuracy improved from 0.78345 to 0.78420, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 114s - loss: 0.4654 - accuracy: 0.7906 - val_loss: 0.4680 - val_accuracy: 0.7842
Epoch 17/25

Epoch 00017: val_accuracy did not improve from 0.78420
704/704 - 114s - loss: 0.4672 - accuracy: 0.7898 - val_loss: 0.4697 - val_accuracy: 0.7825
Epoch 18/25

Epoch 00018: val_accuracy improved from 0.78420 to 0.78655, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 113s - loss: 0.4645 - accuracy: 0.7916 - val_loss: 0.4669 - val_accuracy: 0.7865
Epoch 19/25

Epoch 00019: val_accuracy improved from 0.78655 to 0.78720, saving model to vgg-model-weights-improvement-the-best.h5
704/704 - 113s - loss: 0.4626 - accuracy: 0.7932 - val_loss: 0.4644 - val_accuracy: 0.7872
Epoch 20/25

Epoch 00020: val_accuracy did not improve from 0.78720
704/704 - 113s - loss: 0.4633 - accuracy: 0.7910 - val_loss: 0.4658 - val_accuracy: 0.7850
Epoch 21/25

Epoch 00021: val_accuracy did not improve from 0.78720
704/704 - 113s - loss: 0.4613 - accuracy: 0.7931 - val_loss: 0.4675 - val_accuracy: 0.7846
Epoch 22/25

Epoch 00022: val_accuracy did not improve from 0.78720
704/704 - 112s - loss: 0.4599 - accuracy: 0.7932 - val_loss: 0.4656 - val_accuracy: 0.7860
Epoch 23/25

Epoch 00023: val_accuracy did not improve from 0.78720
704/704 - 114s - loss: 0.4609 - accuracy: 0.7933 - val_loss: 0.4635 - val_accuracy: 0.7854
Epoch 24/25

Epoch 00024: val_accuracy did not improve from 0.78720
704/704 - 114s - loss: 0.4606 - accuracy: 0.7926 - val_loss: 0.4645 - val_accuracy: 0.7865
Epoch 25/25

Epoch 00025: val_accuracy did not improve from 0.78720
704/704 - 114s - loss: 0.4587 - accuracy: 0.7951 - val_loss: 0.4642 - val_accuracy: 0.7850
<tensorflow.python.keras.callbacks.History at 0x7f702315b2e8>
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
Plot of Augmented Generated With a Width Shift Range, Height Shift Range, Zoom Range and Shear Range.

![image_aug](https://user-images.githubusercontent.com/72731175/95825454-132f8b00-0d5b-11eb-8604-fe723f517675.jpeg)



#### Visualisasi Tensorboard Epoch to Loss-Accuracy

```
%load_ext tensorboard
```
We could use feature tensorboard to visualize training process, so as it goes to training, we could see the progress of epoch to loss and accuracy.
```
%tensorboard --logdir logs-vgg-model
```
![vgg loss](https://user-images.githubusercontent.com/72731175/95825639-61dd2500-0d5b-11eb-94cd-44e10f8a3b8e.jpeg)

![vgg acc](https://user-images.githubusercontent.com/72731175/95825679-74575e80-0d5b-11eb-8ae9-d9b5f4e01d3c.jpeg)



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
![vgg report](https://user-images.githubusercontent.com/72731175/95825722-82a57a80-0d5b-11eb-9ed5-7f58fbdd044b.jpeg)


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
![vgg confusion](https://user-images.githubusercontent.com/72731175/95825752-8b964c00-0d5b-11eb-94c0-868335ad3bea.jpeg)


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
![model summary()](https://user-images.githubusercontent.com/72731175/95825770-95b84a80-0d5b-11eb-98df-e8a332516c60.jpeg)


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

Plot of the First 6 Filters From CNN Sequential 5 Layers Conv2D With One Subplot per Channel.

![model filter](https://user-images.githubusercontent.com/72731175/95825800-9e108580-0d5b-11eb-9fcb-d12cfb876dc8.jpeg)


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
Visualization of the Feature Maps Extracted From the First Convolutional Layer in CNN Sequential 5 Layers Conv2D.

![model feature maps](https://user-images.githubusercontent.com/72731175/95825867-b7193680-0d5b-11eb-988d-884a35abfb87.jpeg)


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

train_datagen = ImageDataGenerator(height_shift_range=0.2, width_shift_range=0.2, zoom_range=0.2, shear_range=0.2)
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

```
Epoch 1/25
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0264s vs `on_train_batch_end` time: 0.1117s). Check your callbacks.

Epoch 00001: val_accuracy improved from -inf to 0.81535, saving model to weights-improvement-the-best.h5
704/704 - 89s - loss: 0.5220 - accuracy: 0.7516 - val_loss: 0.4236 - val_accuracy: 0.8153
Epoch 2/25

Epoch 00002: val_accuracy improved from 0.81535 to 0.81600, saving model to weights-improvement-the-best.h5
704/704 - 87s - loss: 0.4530 - accuracy: 0.7988 - val_loss: 0.4210 - val_accuracy: 0.8160
Epoch 3/25

Epoch 00003: val_accuracy improved from 0.81600 to 0.83130, saving model to weights-improvement-the-best.h5
704/704 - 87s - loss: 0.4425 - accuracy: 0.8050 - val_loss: 0.3995 - val_accuracy: 0.8313
Epoch 4/25

Epoch 00004: val_accuracy did not improve from 0.83130
704/704 - 87s - loss: 0.4295 - accuracy: 0.8109 - val_loss: 0.4194 - val_accuracy: 0.8183
Epoch 5/25

Epoch 00005: val_accuracy improved from 0.83130 to 0.83755, saving model to weights-improvement-the-best.h5
704/704 - 90s - loss: 0.4203 - accuracy: 0.8158 - val_loss: 0.3855 - val_accuracy: 0.8375
Epoch 6/25

Epoch 00006: val_accuracy did not improve from 0.83755
704/704 - 89s - loss: 0.4151 - accuracy: 0.8196 - val_loss: 0.3950 - val_accuracy: 0.8332
Epoch 7/25

Epoch 00007: val_accuracy improved from 0.83755 to 0.83985, saving model to weights-improvement-the-best.h5
704/704 - 89s - loss: 0.4077 - accuracy: 0.8224 - val_loss: 0.3772 - val_accuracy: 0.8399
Epoch 8/25

Epoch 00008: val_accuracy improved from 0.83985 to 0.84455, saving model to weights-improvement-the-best.h5
704/704 - 91s - loss: 0.4015 - accuracy: 0.8258 - val_loss: 0.3756 - val_accuracy: 0.8446
Epoch 9/25

Epoch 00009: val_accuracy did not improve from 0.84455
704/704 - 93s - loss: 0.3925 - accuracy: 0.8329 - val_loss: 0.3727 - val_accuracy: 0.8429
Epoch 10/25

Epoch 00010: val_accuracy did not improve from 0.84455
704/704 - 92s - loss: 0.3876 - accuracy: 0.8345 - val_loss: 0.4110 - val_accuracy: 0.8261
Epoch 11/25

Epoch 00011: val_accuracy did not improve from 0.84455
704/704 - 89s - loss: 0.3838 - accuracy: 0.8364 - val_loss: 0.3760 - val_accuracy: 0.8432
Epoch 12/25

Epoch 00012: val_accuracy did not improve from 0.84455
704/704 - 89s - loss: 0.3767 - accuracy: 0.8397 - val_loss: 0.3847 - val_accuracy: 0.8394
Epoch 13/25

Epoch 00013: val_accuracy did not improve from 0.84455
704/704 - 89s - loss: 0.3717 - accuracy: 0.8415 - val_loss: 0.3729 - val_accuracy: 0.8425
Epoch 14/25

Epoch 00014: val_accuracy improved from 0.84455 to 0.84865, saving model to weights-improvement-the-best.h5
704/704 - 89s - loss: 0.3707 - accuracy: 0.8435 - val_loss: 0.3562 - val_accuracy: 0.8486
Epoch 15/25

Epoch 00015: val_accuracy improved from 0.84865 to 0.85405, saving model to weights-improvement-the-best.h5
704/704 - 88s - loss: 0.3627 - accuracy: 0.8471 - val_loss: 0.3599 - val_accuracy: 0.8540
Epoch 16/25

Epoch 00016: val_accuracy did not improve from 0.85405
704/704 - 88s - loss: 0.3590 - accuracy: 0.8489 - val_loss: 0.3626 - val_accuracy: 0.8493
Epoch 17/25

Epoch 00017: val_accuracy did not improve from 0.85405
704/704 - 90s - loss: 0.3593 - accuracy: 0.8494 - val_loss: 0.3601 - val_accuracy: 0.8521
Epoch 18/25

Epoch 00018: val_accuracy improved from 0.85405 to 0.85950, saving model to weights-improvement-the-best.h5
704/704 - 89s - loss: 0.3547 - accuracy: 0.8505 - val_loss: 0.3443 - val_accuracy: 0.8595
Epoch 19/25

Epoch 00019: val_accuracy did not improve from 0.85950
704/704 - 88s - loss: 0.3543 - accuracy: 0.8512 - val_loss: 0.3498 - val_accuracy: 0.8573
Epoch 20/25

Epoch 00020: val_accuracy improved from 0.85950 to 0.86200, saving model to weights-improvement-the-best.h5
704/704 - 88s - loss: 0.3508 - accuracy: 0.8529 - val_loss: 0.3303 - val_accuracy: 0.8620
Epoch 21/25

Epoch 00021: val_accuracy did not improve from 0.86200
704/704 - 88s - loss: 0.3475 - accuracy: 0.8542 - val_loss: 0.3348 - val_accuracy: 0.8613
Epoch 22/25

Epoch 00022: val_accuracy did not improve from 0.86200
704/704 - 87s - loss: 0.3436 - accuracy: 0.8552 - val_loss: 0.3571 - val_accuracy: 0.8519
Epoch 23/25

Epoch 00023: val_accuracy improved from 0.86200 to 0.86345, saving model to weights-improvement-the-best.h5
704/704 - 88s - loss: 0.3440 - accuracy: 0.8553 - val_loss: 0.3289 - val_accuracy: 0.8634
Epoch 24/25

Epoch 00024: val_accuracy did not improve from 0.86345
704/704 - 91s - loss: 0.3421 - accuracy: 0.8569 - val_loss: 0.3313 - val_accuracy: 0.8623
Epoch 25/25

Epoch 00025: val_accuracy did not improve from 0.86345
704/704 - 93s - loss: 0.3399 - accuracy: 0.8570 - val_loss: 0.3424 - val_accuracy: 0.8589
```

#### Visualisasi Tensorboard Epoch to Loss-Accuracy

```
%load_ext tensorboard
```
We could use feature tensorboard to visualize training process, so as it goes to training, we could see the progress of epoch to loss and accuracy.
```
%tensorboard  --logdir logs-deep-model
```
![model acc](https://user-images.githubusercontent.com/72731175/95826060-f9427800-0d5b-11eb-803c-4be11255a6b8.jpeg)

![model loss](https://user-images.githubusercontent.com/72731175/95826086-02334980-0d5c-11eb-8905-90633b8156d6.jpeg)


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
![model report](https://user-images.githubusercontent.com/72731175/95826113-0bbcb180-0d5c-11eb-86d0-ea8802e1f79d.jpeg)


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
![model confusion](https://user-images.githubusercontent.com/72731175/95826134-137c5600-0d5c-11eb-9cf6-a6dcc9d0b9e7.jpeg)


### Conclusion

From this project, several things that can be conclude:
- The Invasive Ductal Carcinoma dataset has an unbalanced number of IDC (-) & IDC (+) data, so it is necessary to generalize the data for further processing.
- The use of VGG16 in this dataset is not as good as the deep layer model because VGG16 is built with a large library data so that the learning process is not as good as the one from scratch.
