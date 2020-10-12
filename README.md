## Final Project - Using Deep Learning to Predict Invasive Ductal Carcinoma (IDC)

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

#### Visualitation Image

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

