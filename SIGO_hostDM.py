from __future__ import print_function, division
from multiprocessing import Pool
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from multiprocessing import Process, Array
#import multiprocessing as mp
from sys import argv
import numpy as np
import readsubfHDF5
import snapHDF5
try:
   import cPickle as pickle
except:
   import pickle
pi = 3.14159265358979
HYDROGEN_MASSFRAC = 0.76
import sys

#wraps to account for period boundary conditions. This mutates the original entry
def dx_wrap(dx, box):
    idx = dx > +box/2.0
    dx[idx] -= box
    idx = dx < -box/2.0
    dx[idx] += box
    return dx
#Calculates distance taking into account periodic boundary conditions
def dist2(dx, dy, dz, box):
    return dx_wrap(dx,box)**2 + dx_wrap(dy,box)**2 + dx_wrap(dz,box)**2

# Units
GRAVITY_cgs = 6.672e-8
BOLTZMANN = 1.38065e-16
PROTONMASS = 1.67262178e-24
GAMMA = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0
MSUN = 1.989e33
MPC = 3.085678e24
KPC = 3.085678e21
ZSUN = 0.0127
UnitLength_in_cm = 3.085678e21 # code length unit in cm/h
UnitMass_in_g = 1.989e43       # code length unit in g/h
UnitVelocity_in_cm_per_s = 1.0e5
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitDensity_in_cgs = UnitMass_in_g/ np.power(UnitLength_in_cm,3)
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / np.power(UnitTime_in_s,2)
UnitEnergy_in_cgs = UnitMass_in_g * np.power(UnitLength_in_cm,2) / np.power(UnitTime_in_s,2)
GCONST = GRAVITY_cgs / np.power(UnitLength_in_cm,3) * UnitMass_in_g *  np.power(UnitTime_in_s,2)
critical_density = 3.0 * .1 * .1 / 8.0 / np.pi / GCONST #.1 is to convert 100/Mpc to 1/kpc, this is in units of h^2
hubbleparam = .71 #hubble constant
baryonfraction = .044 / .27 #OmegaB/Omega0
colors = ["red", "orange", "yellow", "green", "blue", "purple"]

res = '14Mpc'
vel = '118kms'
#snapnum = 32
species = 'H2'
NTask = 16

s_vel = vel.replace(".","")
s_res = res.replace(".","")

fp_halo = open('/home/nakazatoyr/arepo-c/'+res+vel+species+'/SIGOs_info_'+res+'_'+vel+'_'+species+'.dat', 'w')
fp_posi = open('/home/nakazatoyr/arepo-c/'+res+vel+species+'/SIGOs_ID_posi_'+res+'_'+vel+'_'+species+'.dat', 'w')
SIGO_data = np.loadtxt('/home/nakazatoyr/arepo-c/Analysis/ArepoPostProcessing-H/SIGOs_info_'+res+'_'+vel+'_'+species+'.dat')
ID_posi = np.loadtxt('/home/nakazatoyr/arepo-c/Analysis/ArepoPostProcessing-H/SIGO_IDinfo_'+res+'_'+vel+'_'+species+'.dat')
filename = "/xc-work/chiakign/arepo-c/" + res + vel  + species + "/"
filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
filename_032 = filename2 + "/snap-groupordered_" + str(32).zfill(3) #Used for snapHDF5
filename_000 = filename + "snap_" + str(0).zfill(3)  #Used for snapHDF5

iGas= snapHDF5.read_block(filename_032,"ID  ", parttype=0) #load particle indices and catalogs
print('loading data done!')

#read header information
header = snapHDF5.snapshot_header(filename_000)
red = header.redshift
atime = header.time
boxSize = header.boxsize
Omega0 = header.omega0
OmegaLambda = header.omegaL
massDMParticle = header.massarr[1] #all DM particles have same mass

print('reading header information done')
#redshift evolution of critical density
critical_density *= Omega0 + atime**3 * OmegaLambda
critical_density_gas = critical_density * baryonfraction

print(filename_000)
#load particle indices and catalogs
pGas= snapHDF5.read_block(filename_000,"POS ", parttype=0) #correct 4
iGas2= snapHDF5.read_block(filename_000,"ID  ", parttype=0)
mGas= snapHDF5.read_block(filename_000,"MASS", parttype=0)
eGas= snapHDF5.read_block(filename_000,"U   ", parttype=0)
dGas= snapHDF5.read_block(filename_000,"RHO ", parttype=0)
xHI = snapHDF5.read_block(filename_000,"HI  ", parttype=0)
if str(species)=='H2':
    xH2I= snapHDF5.read_block(filename_000,"H2I ", parttype=0)
pDM = snapHDF5.read_block(filename_000,"POS ", parttype=1)
iDM = snapHDF5.read_block(filename_000,"ID  ", parttype=1)
#cat = readsubfHDF5.subfind_catalog(filename2, snapnum)
#r200 = cat.Group_R_Crit200
print('loding catalog done')

#get unique ID
#for i in range(ID_posi.shape[0]):
for i in [0]:
    if i > 4:
        break
    else:
        idx = int(ID_posi[i,0]) #SIGO_idx
        startidx = int(ID_posi[i,1]) #startAllGas
        endidx = int(ID_posi[i,2])   #endAllGas
        ID = iGas[startidx :endidx ]
        print("halo" + str(idx) + ", startidx is " + str(startidx) + ", endidx is " + str(endidx))
        print("the number of IDs is" + str(len(ID)))

    def find_adress(j):
            found_ad = np.where(iGas2 == j)
            return found_ad

    '''
    if __name__ == "__main__":
        p = Pool(NTask)
        adress = p.map(find_adress, ID)
        print(len(adress))
    '''

    for k in ID[0:100]:
        adress = find_adress(k)
        print('ID ' + str(k) + ' adress is ' + str(adress))

    def find_DMadress(j):
        found_DMad = np.where((pDM[0] == pGas[k, 0]) & (pDM[1] == pGas[k, 1]) & (pDM[2] == pGas[k, 2]))
        return found_DMad

    for k in adress[0:4]:
        DMadress = find_DMadress(k)
        print('adress ' + str(k) + ' DMadress is ' + str(adress))

    '''
    if __name__ == "__main__":
        p = Pool(NTask)
        DMadress = p.map(find_adress, adress)
        print(len(DM_adress))
    '''
