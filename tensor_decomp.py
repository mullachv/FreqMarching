import matplotlib.pyplot as plt
from tensorly.base import tensor_to_vec, partial_tensor_to_vec
from tensorly.datasets.synthetic import gen_image
from tensorly.random import check_random_state
from tensorly.kruskal_tensor import kruskal_to_tensor
import tensorly.backend as T
from tensorly.decomposition import tucker, parafac

#Useful for debugging
import numpy as np


def tensor_distance(A, B):
	return np.sum(np.square(tensor_to_vec(A) - tensor_to_vec(B)))/tensor_frob_norm(A)/tensor_frob_norm(B)

def tensor_frob_norm(A):
	return np.sqrt(np.sum(np.square(tensor_to_vec(A))))

def vec_to_tensor(a):
	b = np.outer(a, np.transpose(a))
	tr = np.zeros((len(a), len(a), len(a)))
	for i in range(len(a)):
		tr[:, i,:] = np.outer(b[i], np.transpose(a))
	return tr

def three_vecs_to_tensor(a, b, c):
	assert (len(a) == len(b) == len(c))
	inter = np.outer(a, np.transpose(b))
	tr = np.zeros((len(a), len(a), len(a)))
	for i in range(len(a)):
		tr[:, i,:] = np.outer(inter[i], np.transpose(c))
	return tr

def decomp_plot(edge_len=25, iterations=[1,2,3,4], ranks=[1, 5, 25, 50, 125, 130, 150, 200], decomp='CP'):
	#Params
	print(ranks)

	#Generate random samples
	rng = check_random_state(7)
	X = T.tensor(rng.normal(size=(1000, edge_len, edge_len), loc=0, scale=1))

	#For plotting
	n_rows = len(iterations)
	n_columns = len(ranks) + 1

	fig = plt.figure()

	for i, _ in enumerate(iterations):
		#Generate tensor
		weight_img = X[i*edge_len:(i+1)*edge_len,:,:]

		ax = fig.add_subplot(n_rows, n_columns, i*n_columns+1)

		#Plot image corresponding to 3-D Tensor
		ax.imshow(T.to_numpy(np.sum(weight_img,axis=0)), cmap=plt.cm.OrRd, interpolation='nearest')
		ax.set_axis_off()
		if i == 0:
			ax.set_title('Original')

		for j, rank in enumerate(ranks):
			#Tensor decomposition, image_edge x rank (25x1, 25x5, 25x25 ...)

			if decomp == 'CP':
				#CP decomposition
				components = parafac(weight_img, rank=rank)

				ax = fig.add_subplot(n_rows, n_columns, i * n_columns + j + 2)
				# Aggregate the factors for visualization
				simg = np.sum(components[k] for k in range(len(components)))
				ax.imshow(T.to_numpy(simg), cmap=plt.cm.OrRd, interpolation='nearest')
				ax.text(.5, 2.0, '{:.2f}'.format(tensor_distance(kruskal_to_tensor(components), weight_img)), color='r')
				# ax.set_autoscaley_on(False)
				ax.set_axis_off()
			else:
				#Tucker decomposition
				components, f = tucker(weight_img, ranks=[3, 25, rank])
				#print(components.shape)

				ax = fig.add_subplot(n_rows, n_columns, i * n_columns + j + 2)
				# Aggregate the factors for visualization
				simg = np.sum(components[k] for k in range(len(components)))
				ax.imshow(T.to_numpy(simg), cmap=plt.cm.OrRd, interpolation='nearest')
				ax.text(.5, 2.0, '{:.2f}'.format(tensor_distance(kruskal_to_tensor(components), weight_img)), color='r')
				# ax.set_autoscaley_on(False)
				ax.set_axis_off()

			if i==0:
				ax.set_title('\n{}'.format(rank))

	plt.suptitle('Tensor Decompositions')
	plt.show()


def test_decomp():
	tr = vec_to_tensor(np.array([1,-1]))
	print(tr)
	factors = parafac(tr, rank=1)
	print ('Parafac: {}'.format(factors))
	tucker_facs, _ = tucker(tr, ranks=[2,2,1])
	print('Tucker factors: {}'.format(tucker_facs))
	a = vec_to_tensor(factors[0])
	b = vec_to_tensor(factors[1])
	c = vec_to_tensor(factors[2])
	d1 = three_vecs_to_tensor(factors[0], factors[1], factors[2])
	print(d1)
	t1 = vec_to_tensor(tucker_facs[0])
	t2 = vec_to_tensor(tucker_facs[1])
	print(t1)
	print(t2)
	#d2 = three_vecs_to_tensor(tucker_facs[0], tucker_facs[1],tucker_facs[2])
	#print(d2)

#test_decomp()
decomp_plot(decomp='Tucker')
