from parcels import AdvectionRK4, Field, FieldSet, JITParticle, ScipyParticle 
from parcels import ParticleFile, ParticleSet, Variable, VectorField, ErrorCode
from parcels.tools.converters import GeographicPolar 
from datetime import timedelta as delta
from os import path
from glob import glob
import numpy as np
import dask
import math
import xarray as xr
from netCDF4 import Dataset
import warnings
import matplotlib.pyplot as plt
import pickle
warnings.simplefilter('ignore', category=xr.SerializationWarning)
from operator import attrgetter

########################### DATA INPUT ######################################

#variables
withstokes = False 
withwind = False #scaling factor

#data input
data_in_waves = "/projects/0/topios/hydrodynamic_data/WaveWatch3data/CFSR"
data_in_wind = "/projects/0/topios/hydrodynamic_data/CMEMS/CERSAT-GLO-BLENDED_WIND_L4_REP-V6-OBS_FULL_TIME_SERIE"
data_in_mit = "../../input/MIT4km"
data_out = "../../output"
filename_out = "Beaching_200826"
galapagos_domain = [-94, -87, -3.5, 3]

#run details
seeding_distance = 1 #unit: lon/lat degree (till which distance from islands we seed particles)
seeding_resolution = 4 #unit: gridpoints (horizontal resolution of seeding)
seeding_frequency = 5 #unit: days (how often do we seed particles)
advection_duration = 90 #unit: days (how long does one particle advect in the fields)
output_frequency = 6 #unit: hours
length_simulation = 4*365 #unit: days (how long are we seeding particles)

#Get indices for Galapagos domain to run simulation
def getclosest_ij(lats,lons,latpt,lonpt):    
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_lat = (lats-latpt)**2                      # find squared distance of every point on grid
    dist_lon = (lons-lonpt)**2
    minindex_lat = dist_lat.argmin()                # 1D index of minimum dist_sq element
    minindex_lon = dist_lon.argmin()
    return minindex_lat, minindex_lon                # Get 2D index for latvals and lonvals arrays from 1D index

dfile = Dataset(data_in_mit+'/RGEMS3_Surf_grid.nc')
lon = dfile.variables['XG'][:]
lat = dfile.variables['YG'][:]
iy_min, ix_min = getclosest_ij(lat, lon, galapagos_domain[2], galapagos_domain[0])
iy_max, ix_max = getclosest_ij(lat, lon, galapagos_domain[3], galapagos_domain[1])

#Load distance and seaborder map
file = open('distance_map', 'rb')
data_distance = pickle.load(file)
file.close()
file = open('seaborder_map', 'rb')
data_seaborder = pickle.load(file)
file.close()
lat_high = data_distance['lat']
lon_high = data_distance['lon']
distance_map = data_distance['distance']
seaborder_map = data_seaborder['seaborder']

##################### ADD FIELDS ##############################################

### add MITgcm field

varfiles = sorted(glob(data_in_mit + "/RGEMS_20*.nc"))
meshfile = glob(data_in_mit+"/RGEMS3_Surf_grid.nc")
files_MITgcm = {'U': {'lon': meshfile, 'lat': meshfile, 'data': varfiles},
                'V': {'lon': meshfile, 'lat': meshfile, 'data': varfiles}}
variables_MITgcm = {'U': 'UVEL', 'V': 'VVEL'}
dimensions_MITgcm = {'lon': 'XG', 'lat': 'YG', 'time': 'time'}
indices_MITgcm = {'lon': range(ix_min,ix_max), 'lat': range(iy_min,iy_max)}
fieldset_MITgcm = FieldSet.from_mitgcm(files_MITgcm,
                                       variables_MITgcm, 
                                       dimensions_MITgcm, 
                                       indices = indices_MITgcm)
fieldset = fieldset_MITgcm

### Add waves and wind

if withstokes:
    files_stokes = sorted(glob(data_in_waves + "/WW3-GLOB-30M_200[8-9]*_uss.nc"))
    files_stokes += sorted(glob(data_in_waves + "/WW3-GLOB-30M_201[0-2]*_uss.nc"))
    variables_stokes = {'U_waves': 'uuss', 'V_waves': 'vuss'}
    dimensions_stokes = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'} 
    indices_stokes = {'lon': range(120, 220), 'lat': range(142, 170)}
    fieldset_stokes = FieldSet.from_netcdf(files_stokes,
                                           variables_stokes,
                                           dimensions_stokes,
                                           indices=indices_stokes)
    fieldset_stokes.U_waves.units = GeographicPolar()
    fieldset_stokes.V_waves.units = GeographicPolar()    
    fieldset.add_field(fieldset_stokes.U_waves)
    fieldset.add_field(fieldset_stokes.V_waves)
    uv_waves = VectorField('UVwaves', fieldset.U_waves, fieldset.V_waves)
    fieldset.add_vector_field(uv_waves)
    filename_out += '_wstokes'

if withwind:
    files_wind = sorted(glob(data_in_wind + "/200[8-9]*-fv1.0.nc"))
    files_wind += sorted(glob(data_in_wind + "/201[0-2]*-fv1.0.nc"))
    variables_wind = {'U_wind': 'eastward_wind', 'V_wind': 'northward_wind'}
    dimensions_wind = {'lon': 'lon', 'lat': 'lat', 'time': 'time'}
    indices_wind = {'lon': range(300, 400), 'lat': range(290, 345)}   
    fieldset_wind = FieldSet.from_netcdf(files_wind,
                                         variables_wind,
                                         dimensions_wind,
                                         indices=indices_wind)
    fieldset_wind.U_wind.set_scaling_factor(withwind)
    fieldset_wind.V_wind.set_scaling_factor(withwind)
    fieldset_wind.U_wind.units = GeographicPolar()
    fieldset_wind.V_wind.units = GeographicPolar()
    fieldset.add_field(fieldset_wind.U_wind)
    fieldset.add_field(fieldset_wind.V_wind)
    uv_wind = VectorField('UVwind', fieldset.U_wind, fieldset.V_wind)
    fieldset.add_vector_field(uv_wind)
    filename_out += '_wind%.4d' % (withwind * 1000)      
    
### add unbeaching field

file_UnBeach = 'unbeachingUV.nc'
variables_UnBeach = {'U_unbeach': 'unBeachU', 'V_unbeach': 'unBeachV'}
dimensions_UnBeach = {'lon': 'XG', 'lat': 'YG'}
fieldset_UnBeach = FieldSet.from_c_grid_dataset(file_UnBeach, 
                                                variables_UnBeach,
                                                dimensions_UnBeach,
                                                indices = indices_MITgcm,
                                                tracer_interp_method='cgrid_velocity')
fieldset.add_field(fieldset_UnBeach.U_unbeach)
fieldset.add_field(fieldset_UnBeach.V_unbeach)
uv_unbeach = VectorField('UVunbeach', fieldset.U_unbeach, fieldset.V_unbeach)
fieldset.add_vector_field(uv_unbeach)

### add distance and seaborder map

fieldset.add_field(Field('distance', 
                         data = distance_map,
                         lon = lon_high,
                         lat = lat_high,                   
                         mesh='spherical',
                         interp_method = 'nearest'))
                   
fieldset.add_field(Field('island', 
                         data = seaborder_map,
                         lon = lon_high,
                         lat = lat_high,                   
                         mesh='spherical',
                         interp_method = 'nearest'))

###################### SEEDING PARTICLES ####################################


# get all lon, lat that are land
fU=fieldset_MITgcm.U
fieldset_MITgcm.computeTimeChunk(fU.grid.time[0], 1)
lon = np.array(fU.lon[:]) 
lat = np.array(fU.lat[:])
LandMask = fU.data[0,:,:]
LandMask = np.array(LandMask)
land = np.where(LandMask == 0)

# seed particles at seeding_distance from land
lons = np.array(fU.lon[::seeding_resolution])
lats = np.array(fU.lat[::seeding_resolution])
yy, xx = np.meshgrid(lats,lons)
xcoord = np.reshape(xx,len(lons)*len(lats))
ycoord = np.reshape(yy,len(lons)*len(lats))

startlon=[]
startlat=[]

for i in range(xcoord.shape[0]):
    dist = (xcoord[i]-lon[land[1]])**2 + (ycoord[i]-lat[land[0]])**2
    minindex = dist.argmin()
    if dist[minindex]<seeding_distance and dist[minindex] != 0:
        startlon.append(xcoord[i])
        startlat.append(ycoord[i])
        
################## KERNELS #####################################################

def AdvectionRK4(particle, fieldset, time):
    if particle.beached == 0:
        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)

        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)

        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)

        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
        particle.beached = 2
        
def BeachTesting(particle, fieldset, time):
    if particle.beached == 2 or particle.beached == 3:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if fabs(u) < 1e-14 and fabs(v) < 1e-14:
            if particle.beached == 2:
                particle.beached = 4
            else:
                particle.beached = 1
        elif fabs(u) == 0 and fabs(v) < 1e-9:
            if particle.beached == 2:
                particle.beached = 4
            else:
                particle.beached = 1
        elif fabs(u) < 1e-9 and fabs(v) == 0:
            if particle.beached == 2:
                particle.beached = 4
            else:
                particle.beached = 1
        else:
            particle.beached = 0

def UnBeaching(particle, fieldset, time):
    if particle.beached == 4:
        (ub, vb) = fieldset.UVunbeach[time, particle.depth, particle.lat, particle.lon]
        particle.lon += ub * particle.dt * 400/particle.dt
        particle.lat += vb * particle.dt * 400/particle.dt
        particle.beached = 0
        particle.unbeachCount += 1

def StokesDrag(particle, fieldset, time):
    if particle.beached == 0:
        (u_waves, v_waves) = fieldset.UVwaves[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_waves * particle.dt
        particle.lat += v_waves * particle.dt
        particle.beached = 3

def WindDrag(particle, fieldset, time):
    if particle.beached == 0:
        (u_wind, v_wind) = fieldset.UVwind[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_wind * particle.dt
        particle.lat += v_wind * particle.dt
        particle.beached = 3
        
def Age(fieldset, particle, time):
    particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > 90*86400:
        particle.delete()
    
def SampleInfo(fieldset, particle, time):
    particle.distance = fieldset.distance[time, particle.depth, particle.lat, particle.lon]
    particle.island = fieldset.island[time, particle.depth, particle.lat, particle.lon]

def DeleteParticle(particle, fieldset, time):
    particle.delete()
      
class GalapagosParticle(JITParticle):
    age = Variable('age', dtype=np.float32, initial = 0.)
    unbeachCount = Variable('unbeachCount', dtype=np.int32, initial = 0.)
    distance = Variable('distance', dtype=np.float32, initial = 0.)
    island = Variable('island', dtype=np.int32, initial = 0.)
    beached = Variable('beached', dtype=np.int32, initial = 0.)

######################## EXECUTE #########################################################

pset = ParticleSet(fieldset=fieldset,
                   pclass=GalapagosParticle,
                   lon=startlon,
                   lat=startlat,
                   repeatdt=delta(days=seeding_frequency))

kernel = (pset.Kernel(AdvectionRK4) +
          pset.Kernel(BeachTesting) + 
          pset.Kernel(UnBeaching))
if withstokes:
    kernel += pset.Kernel(StokesDrag) + pset.Kernel(BeachTesting)
if withwind:
    kernel += pset.Kernel(WindDrag) + pset.Kernel(BeachTesting)
kernel += pset.Kernel(Age) + pset.Kernel(SampleInfo)

fname = path.join(data_out, filename_out + ".nc") 
outfile = pset.ParticleFile(name=fname, outputdt=delta(hours=output_frequency))

pset.execute(kernel,
             runtime=delta(days=length_simulation),
             dt=delta(hours=1),
             output_file=outfile,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

pset.repeatdt = None

pset.execute(kernel,
             runtime=delta(days=advection_duration),
             dt=delta(hours=1),
             output_file=outfile,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

outfile.export()
outfile.close()  