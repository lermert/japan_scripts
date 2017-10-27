import numpy as np
from mpi4py import MPI
#from scipy.interpolate import griddata, RegularGridInterpolator, NearestNDInterpolator, LinearNDInterpolator
from pymesher.models_1D import model
from pymesher.skeleton import Skeleton
from scipy.ndimage.filters import gaussian_filter
from math import pi, floor




# SET STUFF HERE.
scale = 6371000.
blockdir = '../MODEL_7_subJapan/'
DOMINANT_PERIOD = 15.0
ELEMENTS_PER_WAVELENGTH = 2.0
MIN_RADIUS =  5800000. # continue downward about 30 % #5771.0000 / 6371.000
REFERENCE_RADIUS = 6371.000
BLOCK_X = blockdir+"block_x"
BLOCK_Y = blockdir+"block_y"
BLOCK_Z = blockdir+"block_z"
output_model_name = '../Simute_2016_Japan_15s_flat_iso.e'
disc_style = 'keep_1D' #keep_1D, smooth
apply_gauss_smooth = True#False
gauss_stds = [2,2,4]


print('*'*60)
print('Interpolation script for a ses3d model to a cartesian salvus mesh.')
print('It is necessary to run this on as many cores as VARIABLES in the below list.')
print('*'*60)


# First item is name of variable in exodus file - second the filename
VARIABLES = [("VS",  blockdir+"dvs_ql6_1Hz_voigt_on_ref"),
             ("VP",  blockdir+"dvp_ql6_1Hz"),
             ("RHO", blockdir+"drho_ORIG"),
             ("QMU",blockdir+"ql_6")]


def smooth_ses3_blocks(var,nx,ny,nz,sig_x,sig_y,sig_z):
    # Given a 1-D array of elastic model parameters,
    # reshape by ses3d convention and smooth w Gaussian kernel

    var = var.reshape(nx,ny,nz)
    var = gaussian_filter(var,[sig_x,sig_y,sig_z])
    return var.ravel()


def get_cube(coord,x,y,z):
    # Given a coordinate tuple and a regular grid defined by x,y,z, return the 
    # vertices or the cube containing the coordinate point.
    Dx = np.diff(x).max()
    Dy = np.diff(y).max()
    Dz = np.diff(z).max()
    

    indx_x = int(floor(((coord[0]-x.min())/Dx)))
    indx_y = int(floor(((coord[1]-y.min())/Dy)))
    indx_z = int(floor(((coord[2]-z.min())/Dz)))
   

    if indx_x < 0:
        indx_x = 0
        incr_x = 0
    elif indx_x >= len(x)-1:
        indx_x = len(x)-1
        incr_x = 0
    else:
        incr_x = 1

    if indx_y < 0:
        indx_y = 0
        incr_y = 0
    elif indx_y >= len(y)-1:
        indx_y = len(y)-1
        incr_y = 0
    else:
        incr_y = 1
    
    if indx_z < 0:
        indx_z = 0
        incr_z = 0
    elif indx_z >= len(z)-1:
        indx_z = len(z)-1
        incr_z = 0
    else:
        incr_z = 1
    # order: 
    # point
    # 000  
    # 100  
    # 001  
    # 101  
    # 010  
    # 110  
    # 011  
    # 111  
    
    cube = [[indx_x, indx_y, indx_z],
            [indx_x+incr_x, indx_y, indx_z],
            [indx_x, indx_y, indx_z+incr_z],
            [indx_x+incr_x, indx_y, indx_z+incr_z],
            [indx_x, indx_y+incr_y, indx_z],
            [indx_x+incr_x, indx_y+incr_y, indx_z],
            [indx_x, indx_y+incr_y, indx_z+incr_z],
            [indx_x+incr_x, indx_y+incr_y, indx_z+incr_z]]

    return cube



def tril_int(point,points,values):

    # point is expected to be a tuple or array
    # (x,y,z), the new point
    # points is expected to be an array 8 x 3 (cube vertices)
    # values is expected to be an array of 8 values
    # If xyz are the coordinates of the vertices, expect the
    # points and values in the following order:

    # point   value
    # 000       0
    # 100       1
    # 001       2
    # 101       3
    # 010       4
    # 110       5
    # 011       6
    # 111       7

    

    x0 = points[0,0]
    x1 = points[1,0]

    y0 = points[0,1]
    y1 = points[4,1]

    z0 = points[0,2]
    z1 = points[2,2]

    # we have margins
    if x0 == x1:
        xd = 0.
    else:
        xd = (point[0]-x0) / (x1-x0)
    if y0 == y1:
        yd = 0.
    else:
        yd = (point[1]-y0) / (y1-y0)
    if z0 == z1:
        zd = 0.
    else:
        zd = (point[2]-z0) / (z1-z0)
    # print xd, yd, zd
    # x dimension

    c00 = values[0] * (1-xd) + values[1] * xd
    c01 = values[3] * (1-xd) + values[2] * xd
    c10 = values[4] * (1-xd) + values[5] * xd
    c11 = values[7] * (1-xd) + values[6] * xd

    # print c00,c01,c10,c11
    # y dimension
    c0 = c00 * (1-yd) + c10 * yd
    c1 = c01 * (1-yd) + c11 * yd
    # print c0,c1
    # z dimension
    c = c0 * (1-zd) + c1 * zd

    return c




# Begin of interpolation routine
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if len(VARIABLES) != size:
    raise NotImplementedError("you have to run on as many cores as there are variables.")

# Read stuff.
theta = np.loadtxt(blockdir+"block_x", dtype=np.float64)
phi = np.loadtxt(blockdir+"block_y", dtype=np.float64)
r = np.loadtxt(blockdir+"block_z", dtype=np.float64)

nblocks = theta[0]
if phi[0] != nblocks or r[0] != nblocks:
    msg = 'Block files have different nr. of discontinuities.'
    raise ValueError(msg)
# ses3D has radius in km
r *= 1000.


if nblocks == 1:
    theta = theta[2:]
    phi = phi[2:]
    r = r[2:]
else:
    raise NotImplementedError(msg)

# Patch r to go to the surface which is easier for the mesher and which also
# will be the case for any real model.
# (only for tiny test model)
# r += (6371 - r.max())

# Normalize radius
# r /= REFERENCE_RADIUS

# initialize model object
mod = model.built_in("prem_ani")
print("Set up model object with 1-D background.")


# Get hmax
hmax = mod.get_edgelengths(
    dominant_period=DOMINANT_PERIOD,
    elements_per_wavelength=ELEMENTS_PER_WAVELENGTH)

hmax *= scale
print(hmax)
discontinuities = mod.discontinuities*scale

# Only a chunk
full_sphere = False

# adapt discontinuities and hmax for min_radius


# The 1-D model may have discontinuities in the selected depth range

idx = discontinuities > MIN_RADIUS
ndisc = idx.sum() + 1
print(ndisc)
discontinuities_new = np.zeros(ndisc)
discontinuities_new[0] = MIN_RADIUS
discontinuities_new[-ndisc+1:] = discontinuities[idx]
discontinuities = discontinuities_new[:]

hmax_new = np.ones(ndisc-1)
hmax_new[-ndisc+1:] = hmax[-ndisc+1:]
hmax = hmax_new[:] 

#==============================================
# Not sure I understand why we need this:
#==============================================
#discontinuities = discontinuities_new[:] # i.e. discontinuities[0] = 0
#hmax = hmax_new[:]    # i.e. hmax[0] = 1.0
    
# This doesn't work at the moment:
# elif disc_style == 'smooth': # No discontinuities; For a ses3D model that has 
# # only one block region. Unfortunately, safest is then to use the smallest hmax
#      if nblocks > 1:
#         raise ValueError('Model contains discontinuities, cant use smooth')

#      discontinuities = np.array([MIN_RADIUS / REFERENCE_RADIUS, r[-1]])
#      hmax = np.array([1.,min(hmax)])
#      mod.nregions = 1


# Get the minimum span of phi. As it is not at the equator it is smaller.
#
# A --- B
# |   /
# | /
# C
# small circle radius = AB
#
# Calculate it at the smallest colatitude.


if disc_style == 'smooth':
    refinement = 'doubling_single_layer'
else:
    refinement = 'tripling'


_AB = np.sin(np.deg2rad(theta.min()))
# We'll build a spherical chunk at the equator and later rotate it.
sk = Skeleton.create_spherical_mesh(
    discontinuities, hmax, ndim=3,
    hmax_refinement=1.5,
    max_colat=[_AB * phi.ptp() * 0.6, theta.ptp() * 0.5],
    min_colat=[-_AB * phi.ptp() * 0.6, -theta.ptp() * 0.5],
    axisem=False,
    full_sphere=False,
    refinement_style=refinement,
    refinement_top_down=True)

m = sk.get_unstructured_mesh(scale=scale)

print('Created sperical mesh.')

#==============================================
# Does anything happen if we use this:
#==============================================
# find outer boundaries
side_set_mode = "spherical"
full_str = {True: '_full', False: '_chunk_z'}
side_set_mode += full_str[full_sphere]
m.find_side_sets(side_set_mode)


# Rotate to where the mesh should actually be located.
m.rotate_coordinates((0,theta.mean(), phi.mean()))
print('Rotated coordinates.')

theta = np.deg2rad(theta)
phi = np.deg2rad(phi)



# Putting coordinates inside the blocks
# 1   2   3   4   5   6   7
# 1   2.5 3.5 4.5 5.5 7

# Wait, wait...how about
# 0     1     2   3   4   5
# 0.5   1.5   2.5 3.5 4.5
#=================================
# I replaced this:
# ================================
# theta[1:-1] += np.diff(theta[1:])
# phi[1:-1] += np.diff(phi[1:])
# r[1:-1] += np.diff(r[1:])

# theta[-2] = theta[-1]
# phi[-2] = phi[-1]
# r[-2] = r[-1]

# theta = theta[:-1]
# phi = phi[:-1]
# r = r[:-1]

# d_theta = np.diff(theta[1:]).max()
# d_phi = np.diff(phi[1:]).max()
# d_r = np.diff(r[1:]).max()

# theta[0] -= 0.5 * d_theta
# phi[0] -= 0.5 * d_theta
# r[0] -= 0.5 * d_theta
# theta[-1] += 0.5 * d_theta
# phi[-1] += 0.5 * d_theta
# r[-1] += 0.5 * d_theta
# ================================
# With this:
# ================================

d_theta = np.diff(theta[1:]).max()
d_phi = np.diff(phi[1:]).max()
d_r = np.diff(r[1:]).max()

theta += 0.5 * d_theta
phi += 0.5 * d_phi
r += 0.5 * d_r

theta = theta[:-1]
phi = phi[:-1]
r = r[:-1]

thetav, phiv, rv = np.meshgrid(theta, phi, r)

#==============================================
# CARTESIAN coordinates here:
#==============================================
# # transform the blocks:
# x = rv * np.sin(thetav) * np.cos(phiv)
# y = rv * np.sin(thetav) * np.sin(phiv)
# z = rv * np.cos(thetav)

# transform mesh points to geographical coordinates,
# to locate them in ses3d grid
x_m = m.points[:, 0]
y_m = m.points[:, 1]
z_m = m.points[:, 2]

r_mesh = np.sqrt(np.power(x_m,2) + 
                np.power(y_m,2) + 
                np.power(z_m,2))
#==============================================
# Hmmmm...why is the arccos needed here. should 
# be arcsin as we want to get latitude, not colat
# (but if I use arcsin, the resulting angles are 
# somewhere in a completely weird place)
# Ahaaaaaa, ses3d uses colatitude :D :D :D nvm
#==============================================
theta_mesh = np.arccos(z_m / r_mesh) 
phi_mesh = np.arctan2(y_m,x_m)


print('Transformed all mesh coordinates...')


npoints = len(m.points[:,0])
nx = len(theta)
ny = len(phi)
nz = len(r)

print('='*60)
print("Block grid extent:")
print(np.max(theta), np.max(phi), np.max(r))
print(np.min(theta), np.min(phi), np.min(r))
print("Salvus grid extent:")
print(np.max(theta_mesh), np.max(phi_mesh), np.max(r_mesh))
print(np.min(theta_mesh), np.min(phi_mesh), np.min(r_mesh))
print('='*60)
print("Salvus grid extent Cartesian:")
print(np.max(x_m), np.max(y_m), np.max(z_m))
print(np.min(x_m), np.min(y_m), np.min(z_m))
print('='*60)



(var, filename)= VARIABLES[rank]
print(var)
dat = np.loadtxt(filename, skiprows=2,dtype=np.float64)


#==============================================
# Smoothing 
#==============================================
if apply_gauss_smooth:

    dat = smooth_ses3_blocks(dat,len(theta),len(phi),len(r),
        gauss_stds[0],gauss_stds[1],gauss_stds[2])
    print("Applied Gaussian smoothing to " + var)

# walk through the mesh grid points
newvar = np.zeros(npoints)
for i in np.arange(npoints):
    if i%1e6 == 0:
        print("Finished %g points of %g." %(i,npoints))
# find cube and values from spherical regular grid
    point = [theta_mesh[i],phi_mesh[i],r_mesh[i]]
    inds = get_cube(point,theta,phi,r)

    
#==============================================
# CARTESIAN coordinates here:
#==============================================
    # point = [m.points[i,:]]
    #coords = np.array([(x[ix[0],ix[1],ix[2]],
    #          y[ix[0],ix[1],ix[2]],
    #          z[ix[0],ix[1],ix[2]]) for ix in inds])
#==============================================  
    coords = np.array([(theta[ix[0]],
              phi[ix[1]],
              r[ix[2]]) for ix in inds])
    
    inds = np.array(inds)
    values = dat[inds[:,2]+inds[:,1]*nz+inds[:,0]*nz*ny]
    # print point[0]*180./pi,point[1]*180./pi,point[2]*REFERENCE_RADIUS
    # print inds
    # print coords
    # print values

# interpolate trilinearly on the spherical grid...cos the 
# cartesian one is not regular
    
    newvar[i] = tril_int(point,coords,np.array(values))
        
        
comm.barrier()

if rank == 0:
    m.attach_field(var,newvar)

for i in range(1,size):

    if rank == i:
        comm.send(newvar,dest=0,tag=i)
    elif rank == 0:
        newvar = comm.recv(source=i,tag=i)
        print('Received data from rank %g' %i)

        varname = VARIABLES[i][0]

        
        m.attach_field(varname,newvar)
 
    
if rank == 0:
    m.attach_field('fluid', np.zeros(m.nelem))   
    m.write_exodus(output_model_name)
