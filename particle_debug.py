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
import h5py

#Class particle loads the already pre-processed map data (unoocupied map, occupied map), sensor data and min_distance 
# data. Initailizes all the parameters for both motion and sensor model
class particle:

	def __init__(self):
		self.map = np.loadtxt('wean.dat', delimiter=' ')
		self.occ = np.loadtxt('occu.dat', delimiter=' ')
		self.unocc = np.loadtxt('unoccu.dat', delimiter=' ')
		self.unocc_dict = {tuple(el):1 for el in self.unocc}
		self.unocc_corr = np.loadtxt('unoccu_corr.dat', delimiter=' ')
		self.zk_5 = np.loadtxt('Zk_5.dat', delimiter=' ')
		self.zk_5_dict = {tuple(key):val for (key,val) in zip(self.unocc, self.zk_5)}
		self.num_p = 1e4
		self.sense = np.loadtxt('sense.dat', delimiter=' ')
		self.isodom = np.loadtxt('is_odom.dat', delimiter=' ')
		self.mindist = np.loadtxt('min_d.dat', delimiter=' ')
		self.a = np.array([1e-6,1e-6,0.1,0.1])
		self.lsr_max = 1000
		self.zrand = 0.033
		self.zhit = 0.9
		self.zshort = 0.033
		self.zmax = 0.034
		self.sig_h = 50
		self.q = 1
		self.srt = 10
		self.end = 170
		self.step = 10
		plt.imshow(self.map, cmap='gray')
		plt.ion()
		self.scat = plt.quiver(0,0,1,0)


# Function initialize - Initializes particles randomly all over the unoccupied cells of the map for the very first iteration.
# Input - num_particle: Number of particles for the filter.
# Output - Array of particles of length num_particles with position (x,y) and orientation (theta)
	def initialize(self, num_particles):
		ind = np.random.randint(self.unocc.shape[0], size=num_particles)
		particles = self.unocc[ind,:]
		# convert r,c to x,y
		particles = (particles - 1) * 10 + 5
		theta = np.random.rand(num_particles) * 2*np.pi - np.pi
		particles = np.insert(particles,[2],np.transpose(theta[np.newaxis]),axis=1)
		return particles


# Function motion_update - Performs motion model on the particles based on Odometry data
# Input - X_t: Array of poses of particles (x,y,theta)
#         O1 : Odometry at time t-1.
#         O2 : Odometry at time t.
# Output - Returns updated poses of the particles. 
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
		return X_upd[:count,:]


# Function get_lsr_poses - Converts poses of particles from odometry frame to the laser frame in world map coordinates.
# Input - X_upd : Updated poses of particles following the motion update.
#             L : Laser measurements which include laser sensor pose relative to odometry.
# Output - X_upd: New updated pose of particles in laser frame.
	def get_lsr_poses(self, X_upd, L):
		th = np.reshape(X_upd[:,2], (len(X_upd),1))
		t = np.sqrt(np.sum(np.power(L[3:5] - L[0:2],2)))
		return (X_upd[:,:2] + np.concatenate((np.cos(th), np.sin(th)), axis=1) * t)


# Function get_wt_vect - Calculates the weight corresponding to each particle using sensor measurement
# Input - X_upd : Array of poses of particles
#             i : Timestep or iteration number
#          angs : An array of angles between -pi/2 and pi/2 equally spaced intervals of 10 degrees
#          inds : array of indicesof the angs
# Output - wt_vect: Array of weights corresponding to the weight of each particle post sensor processing.
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
		return wt_vect


# Function get_wt_vect - Calculates the weight corresponding to each particle using sensor measurement
# Input - X_upd : Array of poses of particles
#             i : Timestep or iteration number
#          angs : An array of angles between -pi/2 and pi/2 equally spaced intervals of 10 degrees
#          inds : array of indicesof the angs
# Output - wt_vect: Array of weights corresponding to the weight of each particle post sensor processing.
	def get_wt_vect_raycast(self, X_upd, i, angs, inds):
		L = self.sense[i]
		#lsr_poses = self.get_lsr_poses(X_upd, L)
		t = np.sqrt(np.sum(np.power(L[3:5] - L[0:2],2)))
		wt_vect = np.empty([len(X_upd),])
		wt_vect.fill(self.q)
		X_keys = np.ceil(X_upd[:,:2]/10).astype(int)
		X_inds = np.around((X_upd[:,2] + np.pi)/(5*np.pi/180)).astype(int)
		for (ind,key,wt) in zip(X_inds, X_keys, wt_vect):
			zk_s = self.zk_5_dict[tuple(key)][ind]
			for i in inds:
				sum = self.zrand/self.lsr_max
				zk = L[6+i] + t
				if(zk >= self.lsr_max):
					zk = self.lsr_max
					sum += self.zmax
				sum += self.zhit * (max(min((zk - zk_s + self.sig_h)/(self.sig_h**2), (zk_s - zk + self.sig_h)/(self.sig_h**2)),0))
				if(zk <= zk_s):
					sum += self.zshort * (2/zk_s * (1 - zk/zk_s))
				wt = wt * sum
		return wt_vect



# Function - get_p_upd : Resamples the particles according to the weights using multinomial distribution function
# Input - wt_vect - Array of weights corresponding to each particle
#           X_upd - Poses of particles
# Output - X_new - New resampled array of poses for each particle.
	def get_p_upd(self, wt_vect, X_upd):
		wt_vect = wt_vect/np.sum(wt_vect)
		dist = np.reshape(np.random.multinomial(self.num_p,wt_vect,1), (len(wt_vect),))
		X_new = np.empty([0,3])
		for p in range(len(dist)):
			X_new = np.concatenate((X_new,np.tile(X_upd[p],(dist[p],1))))
		return X_new

	

	def visualize(self, particles):
		self.scat.remove()
		y = np.floor(particles[:,0] / 10)
		x = np.floor(particles[:,1] / 10)
		v = -np.cos(particles[:,2])
		u = np.sin(particles[:,2])
		self.scat = plt.quiver(x,y,u,v)
		plt.pause(0.000001)




	def main(self):
		#f = h5py.File('output3_2.h5', 'w')
		X_t = self.initialize(self.num_p)
		angs = np.arange(-np.pi/2, np.pi/2 + np.pi/180, np.pi/180)
		inds = np.arange(self.srt-1,self.end,self.step)
		if(not self.isodom[0]):
			wt_vect = self.get_wt_vect_raycast(X_t, 0, angs, inds)
			X_t = self.get_p_upd(wt_vect, X_t)
		for i in range(1,len(self.sense)):
			O1 = self.sense[i-1]
			O2 = self.sense[i]
			X_upd = self.motion_update(X_t, O1, O2)
			if(not self.isodom[i]):
				wt_vect = self.get_wt_vect_raycast(X_upd, i, angs, inds)
				X_upd = self.get_p_upd(wt_vect, X_upd)
			X_t = X_upd
			#f.create_dataset(str(i), data=X_t)
			if(i%10 == 0):
				self.visualize(X_t)
			print(i)
		#f.close()
		return X_t



if __name__ == "__main__":


	p = particle()
	p.main()

