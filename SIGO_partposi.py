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
snapnum = 32
species = 'H'
ref = 'OFF'

s_vel = vel.replace(".","")
s_res = res.replace(".","")

#fp_halo = open('/home/nakazatoyr/arepo-c/'+res+vel+species+'/SIGOs_info_'+res+'_'+vel+'_'+species+'.dat', 'w')
if str(ref) == 'ON':
    fp_halo = open('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'/SIGOs_info_'+res+'_'+vel+'_'+species+'.dat', 'w')
    fp_parID = open('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'/SIGO_IDinfo_'+res+'_'+vel+'_'+species+'.dat', 'w')
    fp_posi = open('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'/SIGOs_ID_posi_'+res+'_'+vel+'_'+species+'.dat', 'w')
    fp_test = open('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'/SIGOs_ID_posi_test.dat', 'w')
if str(ref) == 'OFF':
    fp_halo = open('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'_nonref/SIGOs_info_'+res+'_'+vel+'_'+species+'.dat', 'w')
    fp_parID = open('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'_nonref/SIGO_IDinfo_'+res+'_'+vel+'_'+species+'.dat', 'w')
    fp_posi = open('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'_nonref/SIGOs_ID_posi_'+res+'_'+vel+'_'+species+'.dat', 'w')
    fp_test = open('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'_nonref/SIGOs_ID_posi_test.dat', 'w')

for snapnum in [32]:
    if str(ref) == 'ON':
        #filename = "/xc-work/nakazatoyr/arepo-c/" + res + vel  + species + "_refine/"
        filename = "/xc-work/nakazatoyr/arepo-c/" + res + vel  + species + "_refine/"
    if str(ref) == 'OFF':
        filename = "/xc-work/nakazatoyr/arepo-c/" + res + vel  + species + "_nonref/"
    #filename = "/xc-work/chiakign/arepo-c/" + res + vel  + species + "/"
    filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
    filename3 = filename2 + "/snap-groupordered_" + str(snapnum  ).zfill(3) #Used for snapHDF5
    filename4 = filename + "snap_" + str(snapnum).zfill(3) #Used for hdf5lib, snapHDF5

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
    pGas= snapHDF5.read_block(filename3,"POS ", parttype=0)
    iGas= snapHDF5.read_block(filename3,"ID  ", parttype=0)
    mGas= snapHDF5.read_block(filename3,"MASS", parttype=0)
    eGas= snapHDF5.read_block(filename3,"U   ", parttype=0)
    dGas= snapHDF5.read_block(filename3,"RHO ", parttype=0)
    xHI = snapHDF5.read_block(filename3,"HI  ", parttype=0)
    if str(species)=='H2':
        xH2I= snapHDF5.read_block(filename3,"H2I ", parttype=0)
    pDM = snapHDF5.read_block(filename3,"POS ", parttype=1)
    cat = readsubfHDF5.subfind_catalog(filename2, snapnum)
    #Read in particles
    '''
    posgas = snapHDF5.read_block(filename4, "POS ", parttype=0)
    posdm = snapHDF5.read_block(filename4, "POS ", parttype=1)
    idxdm = snapHDF5.read_block(filename4, "ID  ", parttype=1)
    idxgas = snapHDF5.read_block(filename4, "ID  ", parttype=0)
    '''
    r200 = cat.Group_R_Crit200
    print('loding catalog done')

    halo100_indices = np.where(cat.GroupLenType[:,0] > 100)[0]
    startAllGas = []
    endAllGas   = []
    for i in halo100_indices:
        startAllGas += [np.sum(cat.GroupLenType[:i,0])]
        endAllGas   += [startAllGas[-1] + cat.GroupLenType[i,0]]
    print('defining stat and endAllGas done!')

    #load shrinker and match data
    if str(ref) == 'ON':
        with open("/home/nakazatoyr/arepo-c/Analysis/output/"+res+vel+species+'/shrinker'+s_res+'_'+s_vel+'_'+str(snapnum)+'.dat','rb') as f:
            shrunken = pickle.load(f)
        with open('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'/match'+res+'_'+vel+'_'+str(snapnum)+'.dat','rb') as f:
            matched = pickle.load(f)
    if str(ref) == 'OFF':
        with open("/home/nakazatoyr/arepo-c/Analysis/output/"+res+vel+species+'_nonref/shrinker'+s_res+'_'+s_vel+'_'+str(snapnum)+'.dat','rb') as f:
            shrunken = pickle.load(f)
        with open('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'_nonref/match'+res+'_'+vel+'_'+str(snapnum)+'.dat','rb') as f:
            matched = pickle.load(f)

    print('loading data done!')
    SIGO_call = 0
    for i in halo100_indices:
        print('now, halo' + str(i)+ 'is reading..')
        SIGO = 0
        Fbar = 0.0
        cm = shrunken['cm'][i]
        rotation = shrunken['rotation'][i]
        radii = shrunken['radii'][i]
        mDM = shrunken['mDM'][i]
        DMinEll = shrunken['DMindices'][i]
        Rclosest = matched['Rmin'][i]
        R200dm = matched['R200dm'][i]

        if radii[0] > 0.: #In case of shrinker errors
            if np.sum(cm == np.array([0., 0., 0.]))==3:
                totalGas = np.sum(mGas[startAllGas[i]: endAllGas[i]])
                cm = np.array([np.sum(pGas[startAllGas[i]: endAllGas[i], j]*mGas[startAllGas[i]: endAllGas[i]])/totalGas for j in range(3)])

            #Get positions of gas particles
            P = pGas[startAllGas[i]: endAllGas[i]]
            M = mGas[startAllGas[i]: endAllGas[i]]
            Pdm = pDM[DMinEll]
            #Shift cooedinate system to center on the center of the ellipsoid
            Precentered = dx_wrap(P - cm, boxSize)
            PrecenteredDM = dx_wrap(Pdm -cm, boxSize)
            #Rotate coordinated to the axes point along x,y,z directions:
            Precentered = np.array([np.dot(pp, rotation.T) for pp in Precentered])
            PrecenteredDM = np.array([np.dot(pp, rotation.T) for pp in PrecenteredDM])

            #Figure out which particles are inside the ellipsoid
            inEll = (Precentered[:,0]**2./radii[0]**2. + Precentered[:, 1]**2./radii[1]**2 + Precentered[:,2]**2./radii[2]**2)<=1.
            if np.shape(P[inEll])[0] > 32: #Only consider SIGOs with greater than 32 particles
                if(np.sum(M[inEll])/(np.sum(M[inEll])+ mDM) > .4) and (Rclosest/R200dm>1.): #'inEll' stands for inside the ellipsoid
                    if SIGO_call > 10:
                        break
                    else:
                        #fp_parID = '/home/nakazatoyr/arepo-c/Analysis/ArepoPostProcessing-H/SIGO'+ str(i) +'_IDinfo_'+res+'_'+vel+'_'+species+'.dat'
                        #baryon fraction
                        Fbar = np.sum(M[inEll])/(np.sum(M[inEll])+mDM)
                        #HI and H2I fraction in the clump
                        ID = iGas[startAllGas[i]: endAllGas[i]]
                        M   = mGas[startAllGas[i]: endAllGas[i]]
                        Rho = dGas[startAllGas[i]: endAllGas[i]]
                        E   = eGas[startAllGas[i]: endAllGas[i]]
                        XHI = xHI[startAllGas[i]: endAllGas[i]]
                        posi_x = pGas[startAllGas[i]: endAllGas[i], 0]
                        posi_y = pGas[startAllGas[i]: endAllGas[i], 1]
                        posi_z = pGas[startAllGas[i]: endAllGas[i], 2]
                        if str(species)=='H2':
                            XH2I = xH2I[startAllGas[i]: endAllGas[i]]
                        print('now writing the halo information..')
                        print("%3d %2d %5d %13.5e %13.5e\n", snapnum, red, i, Rclosest, Fbar)
                        fp_halo.write("%3d %2d %5d %13.5e %13.5e\n" % (
                			  snapnum
                			, red
                			, i
                			, Rclosest
                			, Fbar
                			#, HYDROGEN_MASSFRAC * (rho * hubbleparam**2 / atime**3 * UnitDensity_in_cgs) / PROTONMASS
                			#, 1.23*PROTONMASS / BOLTZMANN * GAMMA_MINUS1 * (e * UnitVelocity_in_cm_per_s**2)
                			#, yHI
                			#, yH2I
                			))
                        fp_halo.flush()

                        '''
                        partinfo = np.concatenate([
                                                , posi_x.reshape(len(ID),1)
                                                , posi_y.reshape(len(ID),1)
                                                , posi_z.reshape(len(ID),1)],1)
                        np.savetxt(fp_parID, partinfo)
                        '''

                        fp_parID.write("%3d %2d %5d\n" %(
                                   i
                                 , startAllGas[i]
                                 , endAllGas[i]
                        ))
                        fp_parID.flush()
                        '''
                        fp_test.write("%3d %2d %5d\n" %(
                                    i
                                , iGas[startAllGas[i]]
                                , iGas[startAllGas[i] + 1]
                                , iGas[startAllGas[i] + 2]
                        ))
                        fp_test.flush()
                        '''
                        '''
                        fp_parID = [  i
                                    , startAllGas[i]
                                    , endAllGas[i]]
                        np.savetxt('/home/nakazatoyr/arepo-c/Analysis/ArepoPostProcessing-H/SIGO_IDinfo_'+res+'_'+vel+'_'+species+'.dat', fp_parID)
                        '''
                        print('saved halo'+ str(i) + '_IDcoordinations!')
                        print(len(posi_x))
                        print('from now, plotting task' + str(i) + '...')

                        for j in range(len(posi_x)):
                            plt.scatter(posi_x[j], posi_y[j], s=1, marker = 'o', c = 'black')
                        plt.title('SIGO_plot_' +str(i)+ str(species) + '_z = ' + str(int(red)))
                        plt.xlabel('x [kPc]')
                        plt.ylabel('y [kPc]')
                        plt.savefig('SIGO_plot_' +str(i)+ str(species) + '_z' + str(int(red)) +'.png')
                        plt.close('all')
                        print('plotting task '+ str(i) + 'done!')

                        SIGO_call += 1

                        '''
                        for j in ID:
                            fp_posi.write("3d %13.7f %5d %13.5e %13.5e %13.5e\n" % (
                                  snapnum
                                , red
                                , i #halo ID
                                , pGas[j,0] #x
                                , pGas[j,1] #y
                                , pGas[j,2] #z
                            ))
                            plt.scatter(pGas[j:0],pGas[j:1], pGas[j:2], s=1, maker = 'o', c = 'black')
                        plt.savefig('SIGO_plot_' + str(species) + '_z = ' + redshift +'.png')
                        plt.close('all')
                        print('plotting task done!')
                        '''

fp_halo.close()
fp_parID.close()
fp_test.close()
fp_posi.close()
