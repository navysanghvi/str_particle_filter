#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
import time

class particle:

	def __init__(self):
		self.map = np.loadtxt('wean.dat', delimiter=' ')
		self.occ = np.loadtxt('occu.dat', delimiter=' ')
		self.unocc = np.loadtxt('unoccu.dat', delimiter=' ')
		self.p_init = np.loadtxt('part_init.dat', delimiter=' ')
		self.lsr = np.loadtxt('lsr.dat', delimiter=' ')
		self.a = np.array([1,0.01,1000,1000])
		self.zmax = 0.33
		self.zrand = 0.33
		self.zhit = 0.34
		self.sig_h = 1000
		self.srt = 10
		self.end = 170
		self.step = 10


	def motion_update(self, X_t, O1, O2):
		O_d = O2 - O1
		t = np.sqrt(np.sum(np.power(O_d[0:2],2)))
		r1 = np.arctan2(O_d[1], O_d[0]) - O1[2]
		r2 = -r1 + O_d[2]
		sig_t = np.sqrt(self.a[2] * np.absolute(t) + self.a[3] * np.absolute(r1+r2))
		sig_r1 = np.sqrt(self.a[0] * np.absolute(r1) + self.a[1] * np.absolute(t))
		sig_r2 = np.sqrt(self.a[0] * np.absolute(r2) + self.a[1] * np.absolute(t))
		X_upd = np.empty([len(X_t), len(X_t[0])])
		for i in range(len(X_t)):
			while True:
				h_t = t + np.random.normal(0,sig_t)
				h_r1 = r1 + np.random.normal(0,sig_r1)
				h_r2 = r2 + np.random.normal(0,sig_r2)
				th = np.reshape(X_t[i,2] + h_r1, (1,1))
				pos = X_t[i,:2] + np.concatenate((np.cos(th), np.sin(th)), axis=1) * h_t
				ang = np.reshape(X_t[i,2]+h_r1+h_r2, (1,1))
				ang = (ang + (- 2*np.pi)*(ang > np.pi) + (2*np.pi)*(ang < -np.pi))
				X_upd[i,:] = np.concatenate((pos, ang), axis=1)
				map_c = np.reshape(np.ceil(pos/10).astype(int), (2,))
				if(map_c.tolist() in self.unocc.tolist()):
					break
		return X_upd


	def get_lsr_poses(self, X_upd, L):
		th = np.reshape(X_upd[:,2], (len(X_upd),1))
		t = np.sqrt(np.sum(np.power(L[3:5] - L[0:2],2)))
		return (X_upd[:,:2] + np.concatenate((np.cos(th), np.sin(th)), axis=1) * t)


	def get_wt(self, x_upd, lsr_pos, L):
		angs = np.arange(-np.pi/2, np.pi/2 + np.pi/180, np.pi/180)
		inds = np.arange(self.srt-1,self.end,self.step)
		th = np.reshape(x_upd[2] + angs[inds], (len(inds),1))
		t = np.reshape(L[6+inds], (len(inds),1))
		meas_pos = lsr_pos + np.concatenate((np.cos(th), np.sin(th)), axis=1) * t
		q = 1e5
		for i in range(len(meas_pos)):
			if(t[i] > self.zmax):
				continue
			if(meas_pos[i].tolist() in self.occ.tolist()):
				d = 0
			else:
				d = np.power(np.minimum(cdist(np.reshape(meas_pos[i], (1,2)), self.occ)),2)
			q = q*(self.zhit*np.exp(-d**2/(2*self.sig_h**2)/(np.sqrt(2*np.pi)*self.sig_h)) + self.zrand/self.zmax)

	def initialize(self, num_particles):
		ind = np.random.randint(self.unocc.shape[0], size=num_particles)
		particles = self.unocc[ind,:]
		# convert r,c to x,y
		particles = (particles - 1) * 10 + 5
		theta = np.random.rand(num_particles) * 2*np.pi
		particles = np.insert(particles,[2],np.transpose(theta[np.newaxis]),axis=1)
		return particles



	def visualize(self, particles):
		imgplot = plt.imshow(self.map)
		plt.ion()

		# (ymax,xmax) = np.shape(self.map)

		# for _ in range(10):
		# 	x = []
		# 	y = []
		# 	u = []
		# 	v = []
		# 	for _ in range(100):
		# 		x.append(random.uniform(0,xmax))
		# 		y.append(random.uniform(0,ymax))
		# 		theta = random.uniform(0,180)
		# 		u.append(np.cos(theta*np.pi/180))
		# 		v.append(np.sin(theta*np.pi/180))
		# 	scat = plt.quiver(x,y,u,v)
		# 	# scat = plt.scatter(x,y)
		# 	plt.pause(1)
		# 	scat.remove()

		y = np.floor(particles[:,0] / 10)
		x = np.floor(particles[:,1] / 10)
		u = np.cos(particles[:,2])
		v = np.sin(particles[:,2])
		scat = plt.quiver(x,y,u,v)
		plt.pause(1000)
		scat.remove()


	# plt.show()

	# def get_wt_vect(self, X_upd, i):
		
		
if __name__ == "__main__":

	particle_filter = particle()
	particles = particle_filter.initialize(10000)

	# print particles

	particle_filter.visualize(particles)