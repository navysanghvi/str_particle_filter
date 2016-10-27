#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import h5py

def visualize(particles):
	y = np.floor(particles[:,0] / 10)
	x = np.floor(particles[:,1] / 10)
	v = -np.cos(particles[:,2])
	u = np.sin(particles[:,2])
	scat = plt.quiver(x,y,u,v, color='r')
	plt.pause(0.0001)
	scat.remove()

if __name__ == "__main__":
	f = h5py.File('output3_2.h5', 'r')
	weanmap = np.loadtxt('wean.dat', delimiter=' ')
	plt.imshow(weanmap, cmap='Greys_r')
	plt.ion()
	plt.pause(20)

	for i in range(1,2975,10):
		p = f[str(i)][:]
		visualize(p)
		print i

	f.close()
