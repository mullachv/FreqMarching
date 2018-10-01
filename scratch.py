import tensorly as tl
import numpy as np
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker
from tensorly.random import tucker_tensor
import tensorly.backend as T

# seed = 7
# np.random.seed(seed)
# rng = tl.random.check_random_state(seed)

def initialize():
	sqrt_d = range(1, 10)
	d = np.square(sqrt_d)
	easy = np.floor(np.power(sqrt_d, 1.5) - 1).astype(int)
	hard = np.ceil(np.power(sqrt_d, 1.5) + 1).astype(int)
	# print (d)
	# print (easy)
	# print (hard)
	return d, easy, hard


## tensor decomp ##
def test_decompose_test():
	tr = tl.tensor(np.arange(24).reshape(3,4,2))
	print(tr)
	unfolded = tl.unfold(tr, mode=0)
	tl.fold(unfolded, mode=0, shape=tr.shape)

	#Apply Tucker decomposition
	core, factors = tucker(tr, rank=[3,4,2])
	print ("Core")
	print (core)
	print ("Factors")
	print (factors)

	print(tl.tucker_to_tensor(core, factors))

# Generate n unit norm d-dimension vectors
def n_unit_norm_vecs(n, d):
	vecs = np.random.normal(size=(n, d))
	mags = np.linalg.norm(vecs, axis=-1)
	return vecs / mags[..., np.newaxis]

def test_n_unit_norm_vecs():
	print(n_unit_norm_vecs(1, 4))
	a = [ 0.93887362, -0.25876939,  0.01822746, 0.22632385]
	a = np.array(a)
	print(np.dot(a, a))

def t_decompose(t, n):
	d = t.ndim
	core, factors = tucker(t, rank=[d]*d)
	return core

def compose_tensor(n, vec):
	vecs = n_unit_norm_vecs(n, vec.shape[0])
	print(type(vecs))
	print(vecs)
	return kronecker(np.array(vecs, vecs))

# ds, easy, hard = initialize()
# print(ds)
# print(easy)
#n_unit_norm_vecs(25,10)
# ndx = 2
#a = compose_tensor(easy[ndx], n_unit_norm_vecs(ds[ndx], easy[ndx]))
#print(a.shape)
#a = [ds[ndx]]*3
#print(a)

# b = tucker_tensor([ds[ndx]]*3, [ds[ndx]]*3, True)
# print(b.shape)
#
# c = tucker_tensor([2,2,2], [2,2,2], True)
# print(c)
# de, _ = tucker(c, 2)
# print(de.shape)
# print(de)

# X = T.tensor(rng.normal(size=(2,2,2), loc=0, scale=1))
# print(X)
