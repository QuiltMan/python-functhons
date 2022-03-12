import scipy.io as sio 
import numpy as np
data = sio.loadmat("shot.mat")
matlab_t = sio.loadmat("tt.mat")

shot=data['shot']
t_m=np.transpose(matlab_t['t'])
t_mm=t_m[0]
t_mm

nx=200;
nt=501;
dx=10;
dt=.002;
v0=1250
x = dx*(np.linspace(0, nx-1, nx))
xs=(nx/2)*dx;
xoff=x-xs;

pmin=-2/v0
pmax=-pmin
dp=.25*(pmax-pmin)/nx

p_num=801

stp, tau=tptran(np.transpose(shot),t_mm, xoff, pmin, pmax, dp, p_num)
