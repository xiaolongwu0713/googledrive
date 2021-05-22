import math
from filterpy.common import Q_discrete_white_noise
from numpy.linalg import norm
from math import atan2
import numpy as np
from numpy.random import randn
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
import kf_book.ekf_internal as ekf_internal
import matplotlib.pyplot as plt


basedir='/Users/long/BCI/python_scripts/grasp/Kalman/'
resultdir=basedir+'result/'

# state transition function
def f_cv_radar(x, dt):
    """ state transition function for a constant velocity
    aircraft"""
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]], dtype=float)
    return F @ x

# measurement function
def h_radar(x):
    dx = x[0] - h_radar.radar_pos[0]
    dy = x[2] - h_radar.radar_pos[1]
    slant_range = math.sqrt(dx**2 + dy**2)
    elevation_angle = math.atan2(dy, dx)
    return [slant_range, elevation_angle]


# simulate the radar and the aircraft
class RadarStation:
    def __init__(self, pos, range_std, elev_angle_std):
        self.pos = np.asarray(pos)
        self.range_std = range_std
        self.elev_angle_std = elev_angle_std
    def reading_of(self, ac_pos):
        """ Returns (range, elevation angle) to aircraft.
        Elevation angle is in radians.
        """
        diff = np.subtract(ac_pos, self.pos)
        rng = norm(diff)
        brg = atan2(diff[1], diff[0])
        return rng, brg
    def noisy_reading(self, ac_pos):
        """ Compute range and elevation angle to aircraft with
        simulated noise"""
        rng, brg = self.reading_of(ac_pos)
        rng += randn() * self.range_std
        brg += randn() * self.elev_angle_std
        return rng, brg

class ACSim:
    def __init__(self, pos, vel, vel_std):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.vel_std = vel_std
    def update(self, dt):
        """ Compute and returns next position. Incorporates
        random variation in velocity. """
        dx = self.vel * dt + (randn() * self.vel_std) * dt
        tmp=self.pos
        self.pos = dx + tmp # 2 D (x,y)
        return self.pos

from kf_book.ukf_internal import plot_radar, plot_altitude
dt = 3. # 12 seconds between readings
range_std = 5 # meters
elevation_angle_std = math.radians(0.5)
ac_pos = (0., 1000.)
radar_pos = (0, 0)
ac_vel = (100., 0.)
h_radar.radar_pos = radar_pos
np.random.seed(200)
radar = RadarStation(radar_pos, range_std, elevation_angle_std)
ac = ACSim(ac_pos, (100, 0), 0.02)
time = np.arange(0, 360 + dt, dt)

points = MerweScaledSigmaPoints(n=3, alpha=.1, beta=2., kappa=0.)

P=np.zeros((4,4)) # 4 element in state
Q=np.zeros((4,4)) # 4 element in state # shape: 4*4
Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.1)
P = np.diag([300**2, 3**2, 150**2, 3**2])
R_std=[range_std**2, elevation_angle_std**2]
R = np.diag(R_std) # shape: 2*2

def myUKF(fx, hx, P,Q,R):
    points = MerweScaledSigmaPoints(n=4, alpha=.1, beta=2., kappa=-1.)
    kf = UKF(4, 2, dt, fx=fx, hx=hx, points=points) #(x_dimm, z_dimm,dt, hx, fx, sigmaPoints)
    kf.P = P
    kf.Q = Q
    kf.R = R
    kf.x = np.array([0., 90., 1100., 0.]) # initial gauss
    return kf

ukf = myUKF(fx=f_cv_radar, hx=h_radar,P=P,Q=Q,R=R)

predict, target, measurement, xvar,yvar = [], [], [],[],[]
for t in time:
    if t >= 60:
        ac.vel[1] = 300/60 # 300 meters/minute climb
    newloc=ac.update(dt)
    target.append(newloc)  # (x_coord, y_coord)
    r = radar.noisy_reading(newloc) # (slant, angle)
    measurement.append(np.asarray([r[0]*math.cos(r[1]),r[0]*math.sin(r[1])]))

    ukf.predict()
    ukf.update([r[0], r[1]])

    xvar.append(ukf.P[0,0])
    yvar.append(ukf.P[2,2])
    predict.append(ukf.x[[0,2]]) #(x,x',y,y')


#plot_altitude(xs, time, ys)
epoch=0
figname = resultdir+"ukf"
fig, ax = plt.subplots(2,1,figsize=(6, 3))
target=np.asarray(target)
predict=np.asarray(predict)
measurement=np.asarray(measurement)
ax[0].plot(target[:,0], target[:,1],'r',label='Kalman prediction')
ax[0].plot(predict[:,0], predict[:,1],'y',label='target')
ax[0].scatter(measurement[:,0], measurement[:,1])
ax[0].legend()
ax[1].plot(time,xvar,label='x var')
ax[1].plot(time,yvar,label='y var')
ax[1].legend()
fig.savefig(figname)
plt.close(fig)


from kf_book.mkf_internal import plot_track

