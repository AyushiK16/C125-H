import cv2
import numpy as np
import pandas as pd
#make dataframe
import seaborn as sns
#to make the graphs attractive
import matplotlib.pyplot as plt
#to plot the graph
from sklearn.datasets import fetch_openml
#to fetch the data from the open ml repo
from sklearn.model_selection import train_test_split
#split the data into train and test sets
from sklearn.linear_model import LogisticRegression
#creating log reg. model or classifier
from sklearn.metrics import accuracy_score
#to measure the accuracy of the model that we create
from PIL import Image
import PIL.ImageOps

import cv2
import numpy as np
import pandas as pd
#make dataframe
import seaborn as sns
#to make the graphs attractive
import matplotlib.pyplot as plt
#to plot the graph
from sklearn.datasets import fetch_openml
#to fetch the data from the open ml repo
from sklearn.model_selection import train_test_split
#split the data into train and test sets
from sklearn.linear_model import LogisticRegression
#creating log reg. model or classifier
from sklearn.metrics import accuracy_score
#to measure the accuracy of the model that we create
from PIL import Image
import PIL.ImageOps

import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '')and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


X,y = fetch_openml('mnist_784', version=1, return_X_y = True)
print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n_classes = len(classes)

x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=7500, test_size = 2500, random_state = 9)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0
#gives all the values between 0,1

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scaled, y_train)
#saga works best with multinomial

predictions = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)





clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scaled, y_train)
#saga works best with multinomial

def getPrediction(img):
    im_pil = Image.open(img)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]
