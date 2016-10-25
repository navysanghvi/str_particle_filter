#!/usr/bin/python

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
import numpy.random
from numpy.random import multinomial
import matplotlib.pyplot as plt
import random
import time

class particle:

	def __init__(self):
		self.map = np.loadtxt('wean.dat', delimiter=' ')
		plt.imshow(self.map)
		plt.ion()
		self.occ = np.loadtxt('occu.dat', delimiter=' ')
		self.unocc = np.loadtxt('unoccu.dat', delimiter=' ')
		#self.X_init = np.loadtxt('part_init.dat', delimiter=' ')
		self.num_p = 100
		self.sense = np.loadtxt('sense.dat', delimiter=' ')
		self.isodom = np.loadtxt('is_odom.dat', delimiter=' ')
		self.mindist = np.loadtxt('min_d.dat', delimiter=' ')
		self.a = np.array([1,0.01,1000,1000])
		self.lsr_max = 1000
		self.zmax = 0.33
		self.zrand = 0.33
		self.zhit = 0.34
		self.sig_h = 1000
		self.q = 1e60
		self.srt = 10
		self.end = 170
		self.step = 10
		self.scat = plt.quiver(0,0,0,0)

	def motion_update(self, X_t, O1, O2):
		O_d = O2 - O1
		t = np.sqrt(np.sum(np.power(O_d[0:2],2)))
		r1 = np.arctan2(O_d[1], O_d[0]) - O1[2]
		r2 = -r1 + O_d[2]
		sig_t = np.finfo(float).eps + np.sqrt(self.a[2] * np.absolute(t) + self.a[3] * np.absolute(r1+r2))
		sig_r1 = np.finfo(float).eps + np.sqrt(self.a[0] * np.absolute(r1) + self.a[1] * np.absolute(t))
		sig_r2 = np.finfo(float).eps + np.sqrt(self.a[0] * np.absolute(r2) + self.a[1] * np.absolute(t))
		X_upd = np.empty([len(X_t), len(X_t[0])])
		for i in range(len(X_upd)):
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
		q = self.q
		for i in range(len(meas_pos)):
			if(t[i] > self.lsr_max):
				continue
			min_c = np.floor(meas_pos[i]/10).astype(int)
			if(min_c[0] >= len(self.mindist) or min_c[1] >= len(self.mindist[0]) 
				or min_c[0] < 0 or min_c[1] > 0):
				continue
			d = self.mindist[min_c[0], min_c[1]]
			q = q*(self.zhit*norm.pdf(d, 0, self.sig_h) + self.zrand/self.lsr_max)
		return q


	def get_wt_vect(self, X_upd, i):
		L = self.sense[i]
		lsr_poses = self.get_lsr_poses(X_upd, L)
		wt_vect = np.empty([len(X_upd),])
		for i in range(len(wt_vect)):
			wt_vect[i] = self.get_wt(X_upd[i], lsr_poses[i], L)
		return wt_vect


	def get_p_upd(self, wt_vect, X_upd):
		wt_vect = wt_vect/np.sum(wt_vect)
		dist = np.reshape(np.random.multinomial(len(wt_vect),wt_vect,1), (len(wt_vect),))
		X_new = np.empty([0,3])
		for p in range(len(dist)):
			X_new = np.concatenate((X_new,np.tile(X_upd[p],(dist[p],1))))
		return X_new

	
	def initialize(self, num_particles):
		ind = np.random.randint(self.unocc.shape[0], size=num_particles)
		particles = self.unocc[ind,:]
		# convert r,c to x,y
		particles = (particles - 1) * 10 + 5
		theta = np.random.rand(num_particles) * 2*np.pi
		particles = np.insert(particles,[2],np.transpose(theta[np.newaxis]),axis=1)
		return particles


	def visualize(self, particles):
		self.scat.remove()
		y = np.floor(particles[:,0] / 10)
		x = np.floor(particles[:,1] / 10)
		u = np.cos(particles[:,2])
		v = np.sin(particles[:,2])
		self.scat = plt.quiver(x,y,u,v)
		plt.pause(.0001)

	def main(self):
		X_init = self.initialize(self.num_p)
		if(not self.isodom[0]):
			wt_vect = self.get_wt_vect(X_init, 0)
			X_t = self.get_p_upd(wt_vect, X_init)
		print('hi')
		for i in range(1,len(self.sense)):
			O1 = self.sense[i-1]
			O2 = self.sense[i]
			X_upd = self.motion_update(X_t, O1, O2)
			if(not self.isodom[i]):
				wt_vect = self.get_wt_vect(X_upd,i)
				X_upd = self.get_p_upd(wt_vect, X_upd)
			X_t = X_upd
			self.visualize(X_t)
			print(i)
		return X_t

if __name__ == "__main__":


	p = particle()
	p.main()

