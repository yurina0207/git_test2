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
ref = 'ON'
NTask = 16

s_vel = vel.replace(".","")
s_res = res.replace(".","")

#fp_halo = open('/home/nakazatoyr/arepo-c/'+res+vel+species+'/SIGOs_info_'+res+'_'+vel+'_'+species+'.dat', 'w')
#fp_posi = open('/home/nakazatoyr/arepo-c/'+res+vel+species+'/SIGOs_ID_posi_'+res+'_'+vel+'_'+species+'.dat', 'w')
if str(ref) == 'ON':
    #SIGO_data = np.loadtxt('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'/SIGOs_info_'+res+'_'+vel+'_'+species+'.dat')
    ID_posi = np.loadtxt('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'/SIGO_IDinfo_'+res+'_'+vel+'_'+species+'.dat')
    filename = "/xc-work/nakazatoyr/arepo-c/" + res + vel  + species + "_refine/"
if str(ref) == 'OFF':
    #SIGO_data = np.loadtxt('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'_nonref/SIGOs_info_'+res+'_'+vel+'_'+species+'.dat')
    ID_posi = np.loadtxt('/home/nakazatoyr/arepo-c/Analysis/output/'+res+vel+species+'_nonref/SIGO_IDinfo_'+res+'_'+vel+'_'+species+'.dat')
    filename = "/xc-work/nakazatoyr/arepo-c/" + res + vel  + species + "_nonref/"
filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
filename3_032 = filename2 + "/snap-groupordered_" + str(32).zfill(3) #Used for snapHDF5
print(filename3_032)
iGas= snapHDF5.read_block(filename3_032,"ID  ", parttype=0) #load particle indices and catalogs
print('loading data done!')

#get unique ID
#for i in range(ID_posi.shape[0]):
for i in [0]:
    if i > 10:
        break
    else:
        idx = int(ID_posi[i,0]) #SIGO_idx
        startidx = int(ID_posi[i,1]) #startAllGas
        endidx = int(ID_posi[i,2])   #endAllGas
        ID = iGas[startidx :endidx ]
        print("halo" + str(idx) + ", startidx is" + str(startidx) + ", endidx is" + str(endidx))
        print("the number of IDs is" + str(len(ID)))

    for snapnum in range(20, 33, 2):
    #for snapnum in [32]:
        if str(ref) == 'ON':
            #filename = "/xc-work/nakazatoyr/arepo-c/" + res + vel  + species + "_refine/"
            filename = "/xc-work/nakazatoyr/arepo-c/" + res + vel  + species + "_refine/"
        if str(ref) == 'OFF':
            filename = "/xc-work/nakazatoyr/arepo-c/" + res + vel  + species + "_nonref/"
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
        npart                      = header.npart
        mass                       = header.massarr
        flag_sfr                   = header.sfr
        flag_feedback              = header.feedback
        npartTotal                 = header.nall
        flag_cooling               = header.cooling
        num_files                  = header.filenum
        HubbleParam                = header.hubble
        flag_stellarage            = header.stellar_age
        flag_metals                = header.metals
        npartTotalHighWord         = header.nall_highword
        flag_entropy_instead_u     = 0
        flag_doubleprecision       = header.double
        flag_lpt_ics               = 0
        lpt_scalingfactor          = 0
        flag_tracer_field          = 0
        composition_vector_length  = header.comp_vec

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
        dens= snapHDF5.read_block(filename3,"RHO ", parttype=0)
        adia= snapHDF5.read_block(filename3,"TEMP", parttype=0)
        pDM = snapHDF5.read_block(filename3,"POS ", parttype=1)
        gamma= snapHDF5.read_block(filename3,"GCLT", parttype=0)
        cat = readsubfHDF5.subfind_catalog(filename2, snapnum)
        #load chemical species
        elec     = snapHDF5.read_block(filename3, "ELEC", parttype=0)
        HI       = snapHDF5.read_block(filename3, "HI  ", parttype=0)
        HII      = snapHDF5.read_block(filename3, "HII ", parttype=0)
        HeI      = snapHDF5.read_block(filename3, "HEI ", parttype=0)
        HeII     = snapHDF5.read_block(filename3, "HEII", parttype=0)
        HeIII    = snapHDF5.read_block(filename3, "HE3I", parttype=0)
        if str(species)=='H2':
            HM       = snapHDF5.read_block(filename3, "HM  ", parttype=0)
            H2I      = snapHDF5.read_block(filename3, "H2I ", parttype=0)
            H2II     = snapHDF5.read_block(filename3, "H2II", parttype=0)
            DI       = snapHDF5.read_block(filename3, "DI  ", parttype=0)
            DII      = snapHDF5.read_block(filename3, "DII ", parttype=0)
            HDI      = snapHDF5.read_block(filename3, "HDI ", parttype=0)
            HeHII    = snapHDF5.read_block(filename3, "HeHp", parttype=0)
            DM       = snapHDF5.read_block(filename3, "DM  ", parttype=0)
            HDII     = snapHDF5.read_block(filename3, "HDII", parttype=0)
        else:
            HM       = np.zeros(npart[0], dtype=int)
            H2I      = np.zeros(npart[0], dtype=int)
            H2II     = np.zeros(npart[0], dtype=int)
            DI       = np.zeros(npart[0], dtype=int)
            DII      = np.zeros(npart[0], dtype=int)
            HDI      = np.zeros(npart[0], dtype=int)
            HeHII    = np.zeros(npart[0], dtype=int)
            DM       = np.zeros(npart[0], dtype=int)
            HDII     = np.zeros(npart[0], dtype=int)

        #convert dens[/cm^3] and temp[K]
        nh = HYDROGEN_MASSFRAC * (dens * HubbleParam**2 / atime**3 *  UnitDensity_in_cgs) / PROTONMASS

        elec    /= HYDROGEN_MASSFRAC * 0.00054462
        HI     /= HYDROGEN_MASSFRAC * 1.0
        HII     /= HYDROGEN_MASSFRAC * 1.0
        HeI    /= HYDROGEN_MASSFRAC * 4.0
        HeII   /= HYDROGEN_MASSFRAC * 4.0
        HeIII   /= HYDROGEN_MASSFRAC * 4.0
        if str(species) == 'H2':
            HM     /= HYDROGEN_MASSFRAC * 1.0
            H2I     /= HYDROGEN_MASSFRAC * 2.0
            H2II    /= HYDROGEN_MASSFRAC * 2.0
            DI     /= HYDROGEN_MASSFRAC * 2.0
            DII     /= HYDROGEN_MASSFRAC * 2.0
            HDI    /= HYDROGEN_MASSFRAC * 3.0
            HeHII   /= HYDROGEN_MASSFRAC * 5.0
            DM    /= HYDROGEN_MASSFRAC * 2.0
            HDII   /= HYDROGEN_MASSFRAC * 3.0

        MeanWeight = (
            + elec
            + HI
            + HII
            + HeI
            + HeII
            + HeIII)
        if str(species) == 'H2':
            MeanWeight += (
                + HM
                + H2I
                + H2II )
            MeanWeight += (
                + DI
                + DII
                + HDI  )
            MeanWeight += (
                + HeHII
                + DM
                + HDII )
        MeanWeight = PROTONMASS / HYDROGEN_MASSFRAC / MeanWeight
        Temp = MeanWeight/BOLTZMANN * (gamma-1) * (eGas * UnitVelocity_in_cm_per_s**2)

        #r200 = cat.Group_R_Crit200
        print('loding catalog done')


        def find_adress(j):
                found_ad = np.where(iGas2 == j)
                return found_ad

        if __name__ == "__main__":
            p = Pool(NTask)
            adress = p.map(find_adress, ID)
            print(len(adress))

            print('from now, z = ' + str(red) + '_plotting task' + str(int(idx)) + '...')
            plt.title('SIGO_plot_' +str(idx)+ str(species) + '_z =' + str(int(red)))

            for k in adress:
                plt.scatter(nh[k], Temp[k], s=1, marker = 'o', c = 'black')
                #plt.scatter(test_x[j], test_y[j], s=1, marker = 'o', c = 'black')
                #plt.scatter(pGas[ID,0], pGas[ID,1], s=1, marker = 'o', c = 'black')
            plt.xlim(0.001, 1e8)
            plt.ylim(0.1, 1e5)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel('rho [/cm^3]')
            plt.ylabel('Temp [K]')
            plt.savefig('SIGO_rho_temp_' +str(idx)+ str(species) + '_z' + str(int(red)) +'.png')
            plt.close('all')
            print('plotting task '+ str(idx) + 'done!')
