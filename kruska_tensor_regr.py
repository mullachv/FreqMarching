import matplotlib.pyplot as plt
from tensorly.base import tensor_to_vec, partial_tensor_to_vec
from tensorly.datasets.synthetic import gen_image
from tensorly.random import check_random_state
from tensorly.regression.kruskal_regression import KruskalRegressor
import tensorly.backend as T

#Useful for debugging
import numpy as np

#Params
image_height = 25
image_width = 25

patterns = ['rectangle', 'swiss', 'circle']
ranks = [1,2,3,4,5]

#Generate random samples
rng = check_random_state(7)
X = T.tensor(rng.normal(size=(1000, image_height, image_width), loc=0, scale=1))

#For plotting
n_rows = len(patterns)
n_columns = len(ranks) + 1

fig = plt.figure()

for i, pattern in enumerate(patterns):
	#Generate original image
	weight_img = gen_image(region=pattern, image_height=image_height, image_width=image_width)
	weight_img = T.tensor(weight_img)

	#Labels
	y = T.dot(partial_tensor_to_vec(X, skip_begin=1), tensor_to_vec(weight_img))

	#Plot original weights
	ax = fig.add_subplot(n_rows, n_columns, i*n_columns+1)
	ax.imshow(T.to_numpy(weight_img), cmap=plt.cm.OrRd, interpolation='nearest')
	ax.set_axis_off()
	if i == 0:
		ax.set_title('Original\nWeights')

	for j, rank in enumerate(ranks):
		#Create a tensor regressor for estimator
		estimator = KruskalRegressor(weight_rank=rank, tol=10e-7, n_iter_max=100, reg_W=1, verbose=0)
		estimator.fit(X, y)

		ax = fig.add_subplot(n_rows, n_columns, i*n_columns+j+2)
		ax.imshow(T.to_numpy(estimator.weight_tensor_), cmap=plt.cm.OrRd, interpolation='nearest')
		ax.set_axis_off()

		if i==0:
			ax.set_title('Learned\nRank={}'.format(rank))

plt.suptitle('Kruskal tensor Regressor')
plt.show()
