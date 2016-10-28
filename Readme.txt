----------------------------------------------------------------------------------------------------------------------------------------------
Statistical Techniques in Robotics - Assignment 3 - Particle Filters
Collaborators - Navyata Sanghvi, Jimit Gandhi, Eitan Babcock
----------------------------------------------------------------------------------------------------------------------------------------------
Tested Datasets: robotdata1, robotdata3, robotdata5


Steps to run the code:

1. In the command prompt, type the following command
   
   "   python particle.py   "
   This will run dataset robotdata1
   
2. To change the dataset, open particle.py file and make 2 changes
    a. On line 40, change the 'sense.dat' to 'sense3.dat' or 'sense5.dat' in the line   " self.sense = np.loadtxt('sense.dat', delimiter=' ') "
    b. On line 41, change the 'is_odom.dat' to 'is_odom3.dat' or 'is_odom5.dat' in "self.isodom = np.loadtxt('is_odom.dat', delim....)	

   And then run the file. 

3. The visualize() function is set to ON. It will display the progress of the particle filter.
4. Turn off the visualizer by simply commenting the visualize() function call in the   'def main(self)' function
-----------------------------------------------------------------------------------------------------------------------------------------------
 
Parameters of the filter

We have used various parameters in the filter for motion and sensor model
1. num_p - Number of particles (set to 10000)
2. a - array of error parameters for motion model - These parameters have been adjusted according to the odometry data.
3. lsr_max - max range of the laser sensor (set to 1000)
4. zrand - Intrinsic sensor parameter to account for random measurements.
5. zhit - Intrinsic sensor parameter to account for local measurement noise (Narrow Gaussian)
6.zshort - Inrinsic sensor parameter to account for short range unexpected objects.
7. zmax - Intrinsic sensor parameter to account for Failure measurements.
8. sig_h - Intrinsic sensor parameter which scales the variance around local measurement noise. 
