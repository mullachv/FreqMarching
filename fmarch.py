from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from math import cos, sin

''' Black background canvas
'''
canvas_height, canvas_width, depth = 512, 512, 3
object_height, object_width = 120, 320

#Orange Frame
def create_boxarr():
    object_girth = 20
    data = np.zeros((canvas_height, canvas_width, depth), dtype=np.uint8)
    data[:, :] = [255, 100, 30]
    horizontal_margin = (canvas_width - object_width)/2
    vertical_margin = (canvas_height - object_height)/2
    data[:vertical_margin, :] = [0,0,0]
    data[-vertical_margin:, :] = [0,0,0]
    data[:, :horizontal_margin] = [0,0,0]
    data[:, -horizontal_margin:] = [0,0,0]
    data[vertical_margin+object_girth:-(vertical_margin+object_girth),
        horizontal_margin+object_girth:-(horizontal_margin+object_girth)] = [0,0,0]
    # data[canvas_height / 16:canvas_height - canvas_height / 16, canvas_width / 16:canvas_width - canvas_width / 16] \
    #     = [0, 0, 0] #dark in the middle
    return data

enc = 256
def encode_nums(data):
    encoded = data[:,:,0].astype(int)
    encoded = encoded[:,:]*enc + data[:,:,1]
    encoded = encoded[:,:]*enc + data[:,:,2]
    return encoded

def decode_shape(data):
    db = np.zeros((data.shape[0], data.shape[1], depth), dtype=np.uint8)
    db[:,:,2] = data[:,:]%enc
    data[:,:] = data[:,:]/enc
    db[:,:,1] = data[:,:]%enc
    data[:,:] = data[:,:]/enc
    db[:,:,0] = data[:,:]%enc
    return db

def plot_shape(enc_arr):
    X, Y = np.meshgrid(np.arange(1, enc_arr.shape[0]), np.arange(1, enc_arr.shape[1]))
    Z = enc_arr[X, Y]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X, Y, enc_arr[X, Y], rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    mycolors = cm.jet(Z/np.amax(Z))
    surf = ax.plot_surface(X, Y, enc_arr[X, Y], rstride=10, cstride=10, linewidth=0.1, facecolors=mycolors)
    #ax.set_zlim(-5.0, 5.0)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def create_shape():
    '''
    Creates a 2D shape of size 512 by 512 on a dark background,
    as a 2d-array with RGB encoded as a number ranging
    from 0 to 255*256^2 + 255*256 + 255
    :return:
    '''
    data = create_boxarr()
    #img = Image.fromarray(data, 'RGB')
    #img.show()
    enums = encode_nums(data)
    #plot_shape(enums)
    return enums

def bounded_coord(x):
    return max(min(canvas_height-1, int(x)), 0)

def rotate_point(point, angle_rad):
    c, s = cos(angle_rad), sin(angle_rad)
    return \
        int(point[0]*c - point[1]*s), \
        int(point[0]*s + point[1]*c)

def make_rotation_tx(angle_rad, origin=(0,0)):
    cos_theta, sin_theta = cos(angle_rad), sin(angle_rad)
    x0, y0 = origin
    def xform(point):
        x, y = point[0] - x0, point[1] - y0
        return \
            int(x * cos_theta - y * sin_theta), \
            int(x * sin_theta + y * cos_theta)
    return xform

def rott_shape(data, angle_rad):
    '''
    :param data: 2D object as 2d-array
    :param angle: radians counter-clockwise
    :return: rotated object
    '''
    center = data.shape[0]/2, data.shape[1]/2
    xform = make_rotation_tx(angle_rad, center)
    ndata = np.zeros(data.shape)
    #X, Y = np.meshgrid(np.arange(data.shape[0]) - center[0], np.arange(data.shape[1]) - center[1])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            #m, n = rotate_point((i - center[0], j - center[1]), angle_rad)
            m, n = xform((i, j))
            ndata[bounded_coord(m + center[0]), bounded_coord(n + center[1])] = data[i, j]
    return ndata


def project_shape(data):
    '''
    1D shadow of a 2D shape
    :param data:
    :return:
    '''
    return np.absolute(data.sum(axis=1))

MX = 255 * 256 * 256 + 255 * 256 + 255
def plot_projection(data):
    '''
    Projection as a 1D array
    :param data:
    :return:
    '''
    plt.scatter(np.arange(data.shape[0]), data.shape[0]*[0], c=np.absolute(data)/(MX*1.), cmap='Greens')
    plt.show()

def fft2D(data):
    '''
    Returns the 2D Discrete Fourier Tranform. Output is of the same shape as input,
    but complex numbers
    :param data:
    :return:
    '''
    return np.fft.fft2(data)

def fft1D(data):
    '''
    Returns the 1D Discrete Fourier Tranform. Output is of the same shape as input,
    but complex numbers
    :param data:
    :return:
    '''
    return np.fft.fft(data)


Nq = 40
def freq_range():
    return np.arange(-Nq, Nq)

def get_slices(theta):
    k = freq_range()
    def m_(beta):
        '''
        Polar coordinates sinc function for a rectangle
        sinc(r cos theta / 2 pi) 1/2 sinc(r cos theta/4 pi)

        :param r:
        :param theta:
        :return:
        '''
        return np.sinc(k * np.cos(beta)/(2*np.pi)) * .5 * np.sinc(k * np.sin(beta)/(4.*np.pi))

    def M_():
        return np.asarray([ m_(beta) for beta in theta ])

    return M_

def discretize_2pi():
    factor = 8.
    c = 2*np.pi/factor
    r = 0
    d = []
    while r <= 2*np.pi:
        d += r,
        r += c
    return d

def make_e_matrix(dsc):
    e_m = np.zeros((len(dsc), len(freq_range())), dtype=complex)
    i = 0
    for d in dsc:
        e_r = []
        for n in np.arange(-Nq, Nq):
            e_r += np.exp(1.j * np.pi * d * n),
        e_m[i, :] = e_r
        i += 1
    return e_m

def compute_fr_coeffs(E, M, freq=0):
    '''
    Approximate Fourier coefficients using frequency freq. This needs to be changed to capture
    all frequencies [-freq, freq]

    :param E: a complex matrix of 9x80 (8 polar discretizations of 2*pi, -40 to 40 of freq)
    :param M: real matrix of 9x80 where each row corresponds to a slice at that angle.
                each row contains value at each -40 to 40 frequency slots
    :param freq:
    :return: a vector of coefficients Cf, that is a least squares estimation of E * Cf = M
    '''
    #This needs adjustment
    E = np.matrix(E)
    T = np.matmul(E.getH(), E)
    U = np.matmul(E.getH(), M[:, freq])
    #print(E.shape, T.shape, U.shape)
    return np.matmul(np.linalg.inv(T), U.transpose())

## Main ##

#Ignore these for now#
box = create_shape()
#plot_shape(box)
#dcn = decode_shape(box)

#rotate
rot = rott_shape(box, 1.2)
#plot_shape(rot)

#Project
ps = project_shape(rot)
#ps = project_shape(rot)
#print(ps.shape, ps[280:320])
#plot_projection(ps)

#projection of FFT of rotated
f_ = project_shape(fft2D(rot))

#FFT of projection of rotated
g_ = np.absolute(fft1D(ps))

# print(f_.shape, g_.shape)
# print (MX)
# print(f_[280:300])
# print(g_[280:300])

####### Following are relevant per Prof. Xu #######
polar_discr = discretize_2pi()
fslice = get_slices(polar_discr)
M = fslice()

#9x80
print(M.shape)

#print(a[0])
#print(a[-1])
#print(a.shape)
#print(a)
#plt.scatter(np.arange(a[-1].shape[-1]), a[-1])
#plt.show()

#Complex exponentials matrix for each multiple of the -Nq to Nq multiples of the discretized frequency
E = make_e_matrix(polar_discr)

#9x80
print(E.shape)
print(E[:, 0])
print(E[:, -1])

Cfs = compute_fr_coeffs(E, M, Nq)
print(Cfs.shape)
print(Cfs[0:5,0])

