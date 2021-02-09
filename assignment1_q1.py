

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os


'''
 The following lines grab the data files from TA's github, but
 you can also just change the open statements below to point to
 the files included with the assignment on eclass.
 '''
os.system("git clone https://github.com/pseprivamirakbarnejad/cmput466566.git")

with open('cmput466566/Assignment1/rt-polarity_utf8.neg', 'r') as f:
    lines_neg = f.read().splitlines()

with open('cmput466566/Assignment1/rt-polarity_utf8.pos', 'r') as f:
    lines_pos = f.read().splitlines()

data_train = lines_neg[0:5000] + lines_pos[0:5000]
data_test = lines_neg[5000:] + lines_pos[5000:]

y_train = np.append(np.ones((1,5000)), (np.zeros((1,5000))))
y_test = np.append(np.ones((1, len(lines_neg[5000:]))),\
                   np.zeros((1,len(lines_pos[5000:]))))
print("len(lines_pos) = {}".format(len(lines_pos)))
print("len(lines_neg) = {}".format(len(lines_neg)))

print("len(data_set) = {}".format(len(data_test)))
print("len(data_train) = {}".format(len(data_train)))

#
vectorizer = CountVectorizer(
        lowercase=True, stop_words=None,
        max_df=1.0, min_df=1, max_features=None,  binary=True
      )
X = vectorizer.fit_transform(data_train+data_test).toarray()
X_train = X[0:10000, :]
X_test = X[10000:, :]
feature_names = vectorizer.get_feature_names()
print("X_train.shape = {}".format(X_train.shape))
print("X_test.shape = {}".format(X_test.shape))



"""# Part a. Replace the Parts "YOUR CODE GOES HERE" by Your Code

"""
class MyBayesClassifier():
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for additive smoothing
        

    def train(self, X, y):
        alpha_smooth = self._smooth
        cls = np.unique(y)
        Ncls, Nfeat = len(cls), X.shape[1] #Ncls: number of classes, Nfeat: number of features.
        '''
          =====================
           YOUR CODE GOES HERE
           Here you should create some member variables to store, e.g., p(y) and p(x|y)
            so that the function `predict` can use the variables later on.
          =====================
        '''
        
        
          
        
    def predict(self, X):
        '''
        This function has to return a numpy array of shape X.shape[0] (i.e. of shape "number of testing examples")
        '''
        '''
          =====================
           YOUR CODE GOES HERE
          =====================
        '''
        return pred

clf = MyBayesClassifier(1.0)
clf.train(X_train, y_train);
y_pred = clf.predict(X_test)
print("accuracy = {}".format(np.mean((y_test-y_pred)==0)))
conf_matirx = confusion_matrix(y_test, y_pred)
print("\n\nconf_matrix = \n{}".format(conf_matirx))

"""# Part b."""
'''
  =====================
   YOUR CODE GOES HERE
  =====================
'''


"""# Part c."""
#now with removing stop words ==
vectorizer = CountVectorizer(
        '''
          =====================
           YOUR CODE GOES HERE
          =====================
        '''
      )
X = vectorizer.fit_transform(data_train+data_test).toarray()
X_train = X[0:10000, :]
X_test = X[10000:, :]
feature_names = vectorizer.get_feature_names()

clf = MyBayesClassifier(1.0)
clf.train(X_train, y_train);
y_pred = clf.predict(X_test)
print("accuracy = {}".format(np.mean((y_test-y_pred)==0)))
conf_matirx = confusion_matrix(y_test, y_pred)
print("\n\nconf_matrix = \n{}".format(conf_matirx))


"""# Part d."""
'''
  =====================
  YOUR CODE GOES HERE
  =====================
'''

