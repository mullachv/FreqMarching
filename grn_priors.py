import theano
floatX = theano.config.floatX
import theano.tensor as T
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons


X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
X = scale(X).astype(floatX)
Y = Y.astype(floatX)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

fig, ax = plt.subplots()
ax.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')
ax.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')
sns.despine(); ax.legend()
ax.set(xlabel='X', ylabel='Y', title='Toy data set')

#plt.show()

def exp_kernel(x1, x2):
	a = x1-x2
	return np.exp(-np.dot(a.T, a))

print(exp_kernel(np.array([-1, -1]), np.array([-1, -1])))

#
# Source: https://pythonprogramming.net/svm-in-python-machine-learning-tutorial/
#
class mxSVM:
	def __init__(self):
		pass

	#train
	def train(self, data):
		pass

	def predict(self, features):
		classification = np.sign(np.dot(np.array(features), self.w) + self.b)
		return classification
	