import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

class particle:

	def __init__(self):
		self.map = np.loadtxt('wean.dat', delimiter=' ')
		self.occ = np.loadtxt('occu.dat', delimiter=' ')
		self.unocc = np.loadtxt('unoccu.dat', delimiter=' ')
		self.p_init = np.loadtxt('part_init.dat', delimiter=' ')
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


	def motion_update(self, X_t, O1, O2):
		O_d = O2 - O1
		t = np.sqrt(np.sum(np.power(O_d[0:2],2)))
		r1 = np.arctan2(O_d[1], O_d[0]) - O1[2]
		r2 = -r1 + O_d[2]
		sig_t = np.sqrt(self.a[2] * np.absolute(t) + self.a[3] * np.absolute(r1+r2))
		sig_r1 = np.sqrt(self.a[0] * np.absolute(r1) + self.a[1] * np.absolute(t))
		sig_r2 = np.sqrt(self.a[0] * np.absolute(r2) + self.a[1] * np.absolute(t))
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
			d = self.mindist[min_c[0], min_c[1]]
			q = q*(self.zhit*norm.pdf(d, 0, self.sig_h) + self.zrand/self.lsr_max)
			print(q)
		return q

	def get_wt_vect(self, X_upd, i):
		L = self.sense[i]
		lsr_poses = self.get_lsr_poses(X_upd, L)
		wt_vect = np.empty([len(X_upd), 1])
		for i in range(len(wt_vect)):
			wt_vect[i] = self.get_wt(X_upd[i], lsr_poses[i], L)
		return wt_vect
