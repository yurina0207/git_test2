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

s_vel = vel.replace(".","")
s_res = res.replace(".","")

fp_halo = open('/home/nakazatoyr/arepo-c/'+res+vel+species+'/SIGOs_info_'+res+'_'+vel+'_'+species+'.dat', 'w')
fp_posi = open('/home/nakazatoyr/arepo-c/'+res+vel+species+'/SIGOs_ID_posi_'+res+'_'+vel+'_'+species+'.dat', 'w')
SIGO_data = np.loadtxt('/home/nakazatoyr/arepo-c/Analysis/ArepoPostProcessing-H/SIGOs_info_'+res+'_'+vel+'_'+species+'.dat')
ID_posi = np.loadtxt('/home/nakazatoyr/arepo-c/Analysis/ArepoPostProcessing-H/SIGO_IDinfo_'+res+'_'+vel+'_'+species+'.dat')
filename = "/xc-work/chiakign/arepo-c/" + res + vel  + species + "/"
filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
filename3_032 = filename2 + "/snap-groupordered_" + str(32).zfill(3) #Used for snapHDF5
print(filename3_032)
iGas= snapHDF5.read_block(filename3_032,"ID  ", parttype=0) #load particle indices and catalogs
print('loading data done!')

#get unique ID
for i in range(ID_posi.shape[0]):
    if i > 2:
        break
    else:
        idx = int(ID_posi[i,0]) #SIGO_idx
        startidx = int(ID_posi[i,1]) #startAllGas
        endidx = int(ID_posi[i,2])   #endAllGas
        ID = iGas[startidx :endidx ]
        print("halo" + str(idx) + ", startidx is" + str(startidx) + ", endidx is" + str(endidx))
        print("the number of IDs is" + str(len(ID)))

    for snapnum in range(20, 30, 2):
        filename = "/xc-work/chiakign/arepo-c/" + res + vel  + species + "/"
        filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
        filename3 = filename2 + "/snap-groupordered_" + str(snapnum  ).zfill(3) #Used for snapHDF5
        filename4 = filename + "snap_" + str(snapnum).zfill(3) #Used for hdf5lib, snapHDF5
        #filename3 = filename + "snap_" + str(snapnum).zfill(3) #Used for hdf5lib, snapHDF5

        #read header information
        header = snapHDF5.snapshot_header(filename3)
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

        print(filename3)
        #load particle indices and catalogs
        pGas= snapHDF5.read_block(filename4,"POS ", parttype=0) #correct 4
        iGas2= snapHDF5.read_block(filename4,"ID  ", parttype=0) #correct 4 to filename3_032!
        mGas= snapHDF5.read_block(filename3,"MASS", parttype=0)
        eGas= snapHDF5.read_block(filename3,"U   ", parttype=0)
        dGas= snapHDF5.read_block(filename3,"RHO ", parttype=0)
        xHI = snapHDF5.read_block(filename3,"HI  ", parttype=0)
        if str(species)=='H2':
            xH2I= snapHDF5.read_block(filename3,"H2I ", parttype=0)
        pDM = snapHDF5.read_block(filename3,"POS ", parttype=1)
        cat = readsubfHDF5.subfind_catalog(filename2, snapnum)
        r200 = cat.Group_R_Crit200
        print('loding catalog done')

        print('from now, z = ' + str(red) + '_plotting task' + str(int(idx)) + '...')
        for j in ID:
            #print(len(ID))
            adress = np.where(iGas2 == j)
            posi_x = pGas[adress, 0]
            posi_y = pGas[adress, 1]
            posi_z = pGas[adress, 2]
            plt.scatter(posi_x, posi_y, s=1, marker = 'o', c = 'black')
            #plt.scatter(test_x[j], test_y[j], s=1, marker = 'o', c = 'black')
            #plt.scatter(pGas[ID,0], pGas[ID,1], s=1, marker = 'o', c = 'black')
        plt.title('SIGO_plot_' +str(idx)+ str(species) + '_z =' + str(int(red)))
        plt.xlabel('x [kPc]')
        plt.ylabel('y [kPc]')
        plt.savefig('SIGO_plot_' +str(idx)+ str(species) + '_z' + str(int(red)) +'.png')
        plt.close('all')
        print('plotting task '+ str(idx) + 'done!')
