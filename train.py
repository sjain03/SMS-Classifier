# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:55:55 2019

@author: Sahil
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:21:45 2019

@author: Sahil
"""
#Importing Libraries

import numpy as np
import pandas as pd
import pickle

module1 = "G:\Project\encoding1.pickle" 
module2= "G:\Project\encoding2.pickle"



dataset = pd.read_csv("spam.csv",encoding = 'latin-1')

dataset = dataset.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
dataset.columns = ["category", "text"]

#Importing Libraries
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer


corpus = []

#row wise noise removal and stemming by running loop for all rows

for i in range(0, 5572):
    review = re.sub('[^a-zA-Z0-9,$£₹]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]


    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)



# Creating the Bag of Words model
# Also known as the vector space model
# Text to Features (Feature Engineering on text data)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3500)
features = cv.fit_transform(corpus).toarray()
#labels = dataset.iloc[:, 1].values
labels = dataset.iloc[:, 0]

output1 = open(module1, "wb") 
pickle.dump(cv, output1)
output1.close()


# Fitting Kernel SVM to the Training set
# kernels: linear, poly and rbf

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(features, labels)

# Predicting the Test set results
labels_pred = classifier.predict(features)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(labels, labels_pred)

# Model Score
#score = classifier.score(features,labels)



output2 = open(module2, "wb") 
pickle.dump(classifier, output2)
output2.close()
