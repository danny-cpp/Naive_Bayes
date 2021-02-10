import math

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
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


class MyBayesClassifier:

    def __init__(self, smooth=1):
        self._smooth = smooth  # This is for additive smoothing
        self._neg_dict = None
        self._pos_dict = None

    def train(self, X, y):
        alpha_smooth = self._smooth
        cls = np.unique(y)
        Ncls, Nfeat = len(cls), X.shape[1]  # Ncls: number of classes, Nfeat: number of features.

        # DEPRECATED!! We know that our matrix first 5000 lines are negative. We will construct a dictionary of probaility

        # Array of negative sentences
        negative_sentences = X_train[np.where(y_train == 1)]

        # Total word occurences in negative pool
        # DEPRECATED!! total_neg_word = np.count_nonzero(X_train[0:5000])

        total_neg_word = np.count_nonzero(negative_sentences)

        # Occurence of each unique word in negative pool
        # DEPRECATED!! unique, counts = np.unique(np.where(X_train[0:5000] == 1), return_counts=True)
        # DEPRECATED!! unique, counts = np.unique(np.where(negative_sentences == 1), return_counts=True)

        counts = negative_sentences.sum(axis=0)
        dummy = list(range(0, len(counts)))

        # Calculate the probability of occurrence with smoothing:
        prob = (counts + self._smooth) / (total_neg_word + self._smooth * len(counts))

        self._neg_dict = dict(zip(dummy, prob))

        # Now we do the same for positive

        # Array of positive sentences
        positive_sentences = X_train[np.where(y_train == 0)]

        # DEPRECATED!! total_pos_word = np.count_nonzero(X_train[5000:])
        total_pos_word = np.count_nonzero(positive_sentences)

        counts = positive_sentences.sum(axis=0)

        prob = (counts + self._smooth) / (total_pos_word + self._smooth * len(counts))

        self._pos_dict = dict(zip(dummy, prob))

    def predict(self, X):

        if (self._pos_dict is None or self._neg_dict is None):
            raise Exception("Model is not trained yet!")

        # To predict, we use the Bayes formula: P(C) x P(x0|C) x ... x P(xn|C) = some_ratio x log(P(C)) x ... , which
        # can be transformed further to log(P(C)) + sum(0 -> n)(f x log (P(xi|C))) where f is the occurence of the word in that
        # sentence, P(xi|C) is the word conditional probability.

        # First, analyse the "bag of word" of each sentence. which means find the word "encode" with corresponding frequency in
        # a sentence

        def getBagOWord(line):
            unique, counts = np.unique(np.where(line == 1), return_counts=True)
            bag_o_word = dict(zip(unique, counts))
            return bag_o_word

        def getPred(bag_o_word):
            # Find negative "score"
            neg_score = math.log10(0.5) + sum(
                (bag_o_word[word] * math.log10(self._neg_dict[word])) for word in bag_o_word)

            # Find positive "score"
            pos_score = math.log10(0.5) + sum(
                (bag_o_word[word] * math.log10(self._pos_dict[word])) for word in bag_o_word)

            # Return whichever has higher score
            if neg_score > pos_score:
                return 1
            else:
                return 0

        # Apply to the array

        func = lambda t: getPred(getBagOWord(t))
        result = np.array([func(line) for line in X])

        return result


clf = MyBayesClassifier(1.0)
clf.train(X_train, y_train);
y_pred = clf.predict(X_test)
print("accuracy = {}".format(np.mean((y_test-y_pred)==0)))
conf_matirx = confusion_matrix(y_test, y_pred)
print("\n\nconf_matrix = \n{}".format(conf_matirx))


coeff_range = np.arange(0.1, 3, 0.1).tolist()
acc_output = list()
for i in coeff_range:
    model = MyBayesClassifier(i)
    model.train(X_train, y_train)
    res = model.predict(X_test)
    acc_output.append(accuracy_score(y_test, res))

plt.plot(coeff_range, acc_output)
index = np.argmax(acc_output)
plt.scatter(coeff_range[index], acc_output[index], color='#ee0000')
plt.title("Model's accuarcy for different smoothing coefficient")
plt.show()


"""# Part c."""
ectorizer = CountVectorizer(
        lowercase=True, stop_words='english',
        max_df=1.0, min_df=1, max_features=None,  binary=True
      )
X = vectorizer.fit_transform(data_train+data_test).toarray()
X_train = X[0:10000, :]
X_test = X[10000:, :]
feature_names = vectorizer.get_feature_names()
print("X_train.shape = {}".format(X_train.shape))
print("X_test.shape = {}".format(X_test.shape))


clf = MyBayesClassifier(1.0)
clf.train(X_train, y_train);
y_pred = clf.predict(X_test)
print("accuracy = {}".format(np.mean((y_test-y_pred)==0)))
conf_matirx = confusion_matrix(y_test, y_pred)
print("\n\nconf_matrix = \n{}".format(conf_matirx))

coeff_range2 = np.arange(0.1, 3, 0.1).tolist()
acc_output2 = list()
for i in coeff_range2:
    model = MyBayesClassifier(i)
    model.train(X_train, y_train)
    res = model.predict(X_test)
    acc_output2.append(accuracy_score(y_test, res))

plt.plot(coeff_range2, acc_output2)
index = np.argmax(acc_output2)
plt.scatter(coeff_range2[index], acc_output2[index], color='#ee0000')
plt.title("Model's accuarcy for different smoothing coefficient with stop words removed")
plt.show()