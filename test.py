
"""
Created on Tue Jun 18 10:21:45 2019

@author: Sahil
"""
#Importing Libraries

import numpy as np
import pandas as pd
import pickle

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def spam_detect(msg1):
    encoding1 = "G:\Project\encoding1.pickle"
    cv = pickle.loads(open(encoding1, "rb").read())
    #print(data)

    encoding2 = "G:\Project\encoding2.pickle"
    classifier = pickle.loads(open(encoding2, "rb").read())
    #print(data)


    input_msg = []
    input_msg.append(msg1)
    corpus2 = []

    #perform row wise noise removal and stemming

    input_msg = re.sub('[^a-zA-Z0-9,$£₹]', ' ', input_msg[0])
    input_msg = input_msg.lower()
    input_msg = input_msg.split()
    input_msg = [word for word in input_msg if not word in set(stopwords.words('english'))]

    ps = PorterStemmer()
    input_msg = [ps.stem(word) for word in input_msg]

    input_msg = ' '.join(input_msg)

    corpus2.append(input_msg)




    # Creating the Bag of Words model
    input_msg = cv.transform(corpus2).toarray()

    #Prediction
    labels_pred2 = classifier.predict(input_msg)
    return labels_pred2[0]
