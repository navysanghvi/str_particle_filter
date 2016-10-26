#!/usr/bin/python

import numpy as np
from multiprocessing import Pool
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
		self.occ = np.loadtxt('occu.dat', delimiter=' ')
		self.unocc = np.loadtxt('unoccu.dat', delimiter=' ')
		self.unocc_dict = {tuple(el):1 for el in self.unocc}
		self.unocc_corr = np.loadtxt('unoccu_corr.dat', delimiter=' ')
		#self.X_init = np.loadtxt('part_init.dat', delimiter=' ')
		self.num_p = 1e4
		self.sense = np.loadtxt('sense.dat', delimiter=' ')
		self.isodom = np.loadtxt('is_odom.dat', delimiter=' ')
		self.mindist = np.loadtxt('min_d.dat', delimiter=' ')
		self.a = np.array([1e-6,1e-6,0.1,0.1])
		self.c_lim = 10
		self.lsr_max = 1000
		self.zmax = 0.25
		self.zrand = 0.25
		self.sig_h = 5
		self.zhit = 0.75
		self.q = 1
		self.srt = 10
		self.end = 170
		self.step = 10
		plt.imshow(self.map)
		plt.ion()
		self.scat = plt.quiver(0,0,1,0)


	def motion_update(self, X_t, O1, O2):
		O_d = O2 - O1
		t = np.sqrt(np.sum(np.power(O_d[0:2],2)))
		r1 = np.arctan2(O_d[1], O_d[0]) - O1[2]
		r2 = -r1 + O_d[2]
		sig_t = np.finfo(float).eps + np.sqrt(self.a[2] * np.absolute(t) + self.a[3] * np.absolute(r1+r2))
		sig_r1 = np.finfo(float).eps + np.sqrt(self.a[0] * np.absolute(r1) + self.a[1] * np.absolute(t))
		sig_r2 = np.finfo(float).eps + np.sqrt(self.a[0] * np.absolute(r2) + self.a[1] * np.absolute(t))
		h_t = np.reshape(t + np.random.normal(0,sig_t,len(X_t)), (len(X_t),1))
		h_r1 = r1 + np.random.normal(0,sig_r1,len(X_t))
		h_r2 = r2 + np.random.normal(0,sig_r2,len(X_t))
		th = np.reshape(X_t[:,2] + h_r1, (len(X_t),1))
		pos = X_t[:,:2] + np.concatenate((np.cos(th), np.sin(th)), axis=1) * h_t
		ang = np.reshape(X_t[:,2]+h_r1+h_r2, (len(X_t),1))
		ang = (ang + (- 2*np.pi)*(ang > np.pi) + (2*np.pi)*(ang < -np.pi))
		X_upd = np.concatenate((pos,ang), axis=1)
		map_c = np.ceil(pos/10).astype(int)
		count = 0;
		for i in range(len(X_upd)):
			if(self.unocc_dict.has_key(tuple(map_c[i]))):
				X_upd[count] = X_upd[i]
				count = count+1
		print(count)
		return X_upd[:count,:]


	def get_lsr_poses(self, X_upd, L):
		th = np.reshape(X_upd[:,2], (len(X_upd),1))
		t = np.sqrt(np.sum(np.power(L[3:5] - L[0:2],2)))
		return (X_upd[:,:2] + np.concatenate((np.cos(th), np.sin(th)), axis=1) * t)


	def get_wt(self, x_upd, lsr_pos, L, angs, inds):
		th = np.reshape(x_upd[2] + angs[inds], (len(inds),1))
		t = np.reshape(L[6+inds], (len(inds),1))
		meas_pos = lsr_pos + np.concatenate((np.cos(th), np.sin(th)), axis=1) * t
		q = self.q
		for i in range(len(meas_pos)):
			if(t[i] > self.lsr_max):
				continue
			min_c = np.floor(meas_pos[i]/10).astype(int)
			if(min_c[0] >= len(self.mindist) or min_c[1] >= len(self.mindist[0]) 
				or min_c[0] < 0 or min_c[1] < 0):
				q = q*self.zrand/self.lsr_max
				continue
			d = self.mindist[min_c[0], min_c[1]]
			q = q*(self.zhit*norm.pdf(d, 0, self.sig_h) + self.zrand/self.lsr_max)
		return q


	def get_wt_vect(self, X_upd, i, angs, inds):
		L = self.sense[i]
		lsr_poses = self.get_lsr_poses(X_upd, L)
		wt_vect = np.empty([len(X_upd),])
		wt_vect.fill(self.q)
		for i in inds.tolist():
			th = np.reshape(X_upd[:,2] + angs[i], (len(X_upd),1))
			t = L[6+i]
			meas_pos = lsr_poses + np.concatenate((np.cos(th), np.sin(th)), axis=1) * t
			if(t > self.lsr_max):
				continue
			min_c = np.floor(meas_pos/10).astype(int)
			for j in range(len(meas_pos)):
				if(min_c[j,0] >= len(self.mindist) or min_c[j,1] >= len(self.mindist[0]) 
					or min_c[j,0] < 0 or min_c[j,1] < 0):
					wt_vect[j] = wt_vect[j]*self.zrand/self.lsr_max
					continue
				d = self.mindist[min_c[j,0], min_c[j,1]]
				x = d/(2*self.sig_h**2)
				wt_vect[j] = wt_vect[j]*(self.zhit/(1+x) + self.zrand/self.lsr_max)
				#print(wt_vect[j])
		return wt_vect


	def get_p_upd(self, wt_vect, X_upd):
		wt_vect = wt_vect/np.sum(wt_vect)
		dist = np.reshape(np.random.multinomial(self.num_p,wt_vect,1), (len(wt_vect),))
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
		v = -np.cos(particles[:,2])
		u = np.sin(particles[:,2])
		self.scat = plt.quiver(x,y,u,v)
		plt.pause(0.000001)

	def main(self):
		X_init = self.initialize(self.num_p)
		angs = np.arange(-np.pi/2, np.pi/2 + np.pi/180, np.pi/180)
		inds = np.arange(self.srt-1,self.end,self.step)
		if(not self.isodom[0]):
			wt_vect = self.get_wt_vect(X_init, 0, angs, inds)
			X_t = self.get_p_upd(wt_vect, X_init)
		print('hi')
		t1 = t2 = t3 = t4 = t5 = 0
		for i in range(1,len(self.sense)):
			O1 = self.sense[i-1]
			O2 = self.sense[i]
			t1 = time.time()
			X_upd = self.motion_update(X_t, O1, O2)
			t2 = time.time()
			t3 = t4 = t5 = 0
			if(not self.isodom[i]):
				t3 = time.time()
				wt_vect = self.get_wt_vect(X_upd, i, angs, inds)
				t4 = time.time()
				X_upd = self.get_p_upd(wt_vect, X_upd)
				t5 = time.time()
			X_t = X_upd
			print 'Motion Update Time: ' + str(t2 - t1)
			print 'Get Weight Time: ' + str(t4 - t3)
			#print 'Get Update: ' + str(t5 - t4)
			self.visualize(X_t)
			print(i)
		return X_t

if __name__ == "__main__":


	p = particle()
	p.main()

