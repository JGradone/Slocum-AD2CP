import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
import scipy.interpolate as interp
import math
from scipy.sparse.linalg import lsqr
import scipy
from scipy import integrate

def ellipsoid_fit(X, flag=0, equals='xy'):
	if X.shape[1] != 3:
		raise ValueError('Input data must have three columns!')
	
	x = X[:, 0]
	y = X[:, 1]
	z = X[:, 2]
	
	# Check for sufficient points
	if len(x) < 9 and flag == 0:
		raise ValueError('Must have at least 9 points to fit a unique ellipsoid')
	if len(x) < 6 and flag == 1:
		raise ValueError('Must have at least 6 points to fit a unique oriented ellipsoid')
	if len(x) < 5 and flag == 2:
		raise ValueError('Must have at least 5 points to fit a unique oriented ellipsoid with two axes equal')
	if len(x) < 4 and flag == 3:
		raise ValueError('Must have at least 4 points to fit a unique sphere')

	if flag == 0:
		D = np.array([x**2, y**2, z**2, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z]).T
	elif flag == 1:
		D = np.array([x**2, y**2, z**2, 2*x, 2*y, 2*z]).T
	elif flag == 2:
		if equals in ['yz', 'zy']:
			D = np.array([y**2 + z**2, x**2, 2*x, 2*y, 2*z]).T
		elif equals in ['xz', 'zx']:
			D = np.array([x**2 + z**2, y**2, 2*x, 2*y, 2*z]).T
		else:
			D = np.array([x**2 + y**2, z**2, 2*x, 2*y, 2*z]).T
	else:
		D = np.array([x**2 + y**2 + z**2, 2*x, 2*y, 2*z]).T

	# Solve the normal system of equations
	v = np.linalg.lstsq(D, np.ones(len(x)), rcond=None)[0]

	if flag == 0:
		A = np.array([[v[0], v[3], v[4], v[6]],
					  [v[3], v[1], v[5], v[7]],
					  [v[4], v[5], v[2], v[8]],
					  [v[6], v[7], v[8], -1]])
		
		center = -np.linalg.inv(A[:3, :3]).dot(v[6:9])
		T = np.eye(4)
		T[3, :3] = center
		R = T.dot(A).dot(T.T)
		evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
		radii = np.sqrt(1. / evals)
	else:
		if flag == 1:
			v = np.concatenate([v[:3], [0, 0, 0], v[3:]])
		elif flag == 2:
			if equals in ['xz', 'zx']:
				v = np.concatenate([[v[0], v[1], v[0]], [0, 0, 0], v[2:]])
			elif equals in ['yz', 'zy']:
				v = np.concatenate([[v[1], v[0], v[0]], [0, 0, 0], v[2:]])
			else:  # xy
				v = np.concatenate([[v[0], v[0], v[1]], [0, 0, 0], v[2:]])
		else:
			v = np.concatenate([[v[0], v[0], v[0]], [0, 0, 0], v[1:]])
		
		center = -v[6:9] / v[:3]
		gam = 1 + (v[6]**2 / v[0] + v[7]**2 / v[1] + v[8]**2 / v[2])
		radii = np.sqrt(gam / v[:3])
		evecs = np.eye(3)

	return center, radii, evecs, evals if 'evals' in locals() else None, v
	
	
def calc_tilt_matrix(pitch, roll):
	"""
	Calculate the tilt matrix based on the pitch and roll angles.

	:param pitch: The pitch angle in degrees.
	:param roll: The roll angle in degrees.
	:return: The tilt matrix.
	"""
	sinpp = np.sin(np.radians(pitch))
	cospp = np.cos(np.radians(pitch))
	sinrr = np.sin(np.radians(roll))
	cosrr = np.cos(np.radians(roll))

	m = np.array([
		[cospp, -sinpp * sinrr, -cosrr * sinpp],
		[0, cosrr, -sinrr],
		[sinpp, sinrr * cospp, cospp * cosrr]
	])

	return m
	
	
def calc_heading(hxhyhz_sensor, pitch, roll, orientation):
	"""
	Calculate the heading based on sensor data, pitch, roll, and orientation.

	:param hxhyhz_sensor: The sensor data as a list or numpy array [Hx, Hy, Hz].
	:param pitch: The pitch angle in degrees.
	:param roll: The roll angle in degrees.
	:param orientation: The orientation of the instrument (0 for upwards, otherwise downwards).
	:return: The calculated heading in degrees.
	"""
	# Heading calculation
	m_tilt = calc_tilt_matrix(pitch, roll)

	# Check the orientation and adjust sensor data accordingly
	if orientation == 0:
		# Upwards looking instrument
		hxhyhz_inst = np.array(hxhyhz_sensor)
	else:
		# Downwards looking instrument (rotation around x-axis)
		hxhyhz_inst = np.array(hxhyhz_sensor)
		hxhyhz_inst[1:] = -hxhyhz_inst[1:]

	# Calculate the magnetic vector in the Earth aligned coordinate system
	hxhyhz_earth = np.dot(m_tilt, hxhyhz_inst)

	# Calculate the heading
	heading = np.arctan2(hxhyhz_earth[1], hxhyhz_earth[0]) * (180 / np.pi)
	if heading < 0:
		heading += 360

	return heading


def correct_ad2cp_heading(time, head, pitch, roll, pressure, x, y, z):
	head = np.array(head).T
	pitch = np.array(pitch).T
	roll = np.array(roll).T
	pressure = np.array(pressure).T
	x = np.array(x).T
	y = np.array(y).T
	z = np.array(z).T
	time = np.array(time).T

	xyz_original = np.column_stack((x, y, z))
	
	pitch_ranges = np.arange(-20, 21, 1)
	for k in range(len(pitch_ranges) - 1):
		mask = (pitch > pitch_ranges[k]) & (pitch < pitch_ranges[k + 1])
		indices = np.where(mask)
		if len(indices[0]) > 0:
			xyz1 = np.column_stack((x[indices], y[indices], z[indices]))
			offset, *_ = ellipsoid_fit(xyz1)

			x1 = x[indices] - offset[0]
			y1 = y[indices] - offset[1]
			z1 = z[indices] - offset[2]

			new_center, *_ = ellipsoid_fit(np.column_stack((x1, y1, z1)))
			if abs(new_center[0]) > 150 or abs(new_center[1]) > 150:
				x1 = x[indices]
				y1 = y[indices]
				z1 = z[indices]

			x[indices] = x1
			y[indices] = y1
			z[indices] = z1

	xyz_final = np.column_stack((x, y, z))

	headingf = np.empty_like(head) * np.nan  # ensure headingf has the same shape as head
	headingo = np.copy(headingf)
	for k in range(len(head)):  # use head since it's your original 1D array
		headingf[k] = calc_heading(xyz_final[k, :], pitch[k], roll[k], 1)
		headingo[k] = calc_heading(xyz_original[k, :], pitch[k], roll[k], 1)
	x_o=xyz_original[:,0]
	y_o=xyz_original[:,1]
	return headingf,x_o,y_o,x,y  # return the corrected heading
	
	
from scipy.interpolate import interp1d

def grid_ad2cp_beams(ds, Vrange):
	# Preallocate interpolated velocity matrices with NaN
	InterpVelocityBeam1 = np.full((len(Vrange), len(ds['time'])), np.nan)
	InterpVelocityBeam2 = np.full_like(InterpVelocityBeam1, np.nan)
	InterpVelocityBeam3 = np.full_like(InterpVelocityBeam1, np.nan)
	InterpVelocityBeam4 = np.full_like(InterpVelocityBeam1, np.nan)
	
	# Loop through each time (ping) and interpolate beam velocity onto the regular grid
	for x in range(len(ds['time'])):
		InterpVelocityBeam1[:, x] = interp1d(ds['TrueDepthBeam1'][:, x], ds['VelocityBeam1'][:, x],
											 bounds_error=False, fill_value=np.nan)(Vrange)
		InterpVelocityBeam2[:, x] = interp1d(ds['TrueDepthBeam2'][:, x], ds['VelocityBeam2'][:, x],
											 bounds_error=False, fill_value=np.nan)(Vrange)
		InterpVelocityBeam3[:, x] = interp1d(ds['TrueDepthBeam3'][:, x], ds['VelocityBeam3'][:, x],
											 bounds_error=False, fill_value=np.nan)(Vrange)
		InterpVelocityBeam4[:, x] = interp1d(ds['TrueDepthBeam4'][:, x], ds['VelocityBeam4'][:, x],
											 bounds_error=False, fill_value=np.nan)(Vrange)
											 
	# Ensure the dimension names align with those in the existing dataset
	velocity_range_dim = 'VelocityRange'  # or whatever the exact name is in your dataset
	# Add the interpolated data back into the structure, ensuring consistent dimension names
	ds['InterpVelocityBeam1'] = xr.DataArray(InterpVelocityBeam1, dims=[velocity_range_dim, 'time'], coords={velocity_range_dim: ds[velocity_range_dim], 'time': ds['time']})
	ds['InterpVelocityBeam2'] = xr.DataArray(InterpVelocityBeam2, dims=[velocity_range_dim, 'time'], coords={velocity_range_dim: ds[velocity_range_dim], 'time': ds['time']})
	ds['InterpVelocityBeam3'] = xr.DataArray(InterpVelocityBeam3, dims=[velocity_range_dim, 'time'], coords={velocity_range_dim: ds[velocity_range_dim], 'time': ds['time']})
	ds['InterpVelocityBeam4'] = xr.DataArray(InterpVelocityBeam4, dims=[velocity_range_dim, 'time'], coords={velocity_range_dim: ds[velocity_range_dim], 'time': ds['time']})

	# Create meshgrid for TrueDepth
	time_dim = 'time'  # or whatever the exact name is in your dataset
	# Create meshgrid for TrueDepth and assign it with proper dimensions
	true_depth = np.meshgrid(ds[time_dim], Vrange, indexing='ij')[1]
	ds['TrueDepth'] = xr.DataArray(true_depth, dims=[time_dim, velocity_range_dim], coords={time_dim: ds[time_dim], velocity_range_dim: Vrange})

	
	return ds


def calcAHRS(headingVal, rollVal, pitchVal):
	RotMatrix = np.full((9,len(pitchVal)), np.nan)
	
	for k in range(len(pitchVal)):

		hh = np.deg2rad(headingVal[k]-90)
		pp = np.deg2rad(pitchVal[k])
		rr = np.deg2rad(rollVal[k])
		
		# Make heading matrix
		H = np.array([
			[np.cos(hh), np.sin(hh), 0],
			[-np.sin(hh), np.cos(hh), 0],
			[0, 0, 1]
		])
		
		# Make tilt matrix
		P = np.array([
			[np.cos(pp), -np.sin(pp)*np.sin(rr), -np.cos(rr)*np.sin(pp)],
			[0, np.cos(rr), -np.sin(rr)],
			[np.sin(pp), np.sin(rr)*np.cos(pp), np.cos(pp)*np.cos(rr)]
		])
		
		# Make resulting transformation matrix
		R = np.dot(H, P)
		RotMatrix[:,k] = R.reshape(-1)
	
	return RotMatrix
	
	
def beam2enu(ds):
	## 01/21/2022     jgradone@marine.rutgers.edu     Initial

	## This function transforms velocity data from beam coordinates to XYZ to ENU. Beam coordinates
	## are defined as the velocity measured along the three beams of the instrument.
	## ENU coordinates are defined in an earth coordinate system, where E represents the East-West
	## component, N represents the North-South component and U represents the Up-Down component.
	## This function was created for a Nortek AD2CP mounted looking downward on a Slocum glider.

	#############################################################################################################
	## Per Nortek:                                                                                             ##
	## https://support.nortekgroup.com/hc/en-us/articles/360029820971-How-is-a-coordinate-transformation-done- ##
	#############################################################################################################

	## "Each instrument has its own unique transformation matrix, based on the transducer geometry.
	## This matrix can be found, as previously mentioned, in the .hdr file generated when performing
	## a binary data conversion in the software. Each row of the matrix represents a component in the
	## instrument’s XYZ coordinate system, starting with X at the top row. Each column represents a beam.
	## The third and fourth rows of the Vectrino or Signature transformation matrix represent the two
	## estimates of vertical velocity (Z1 and Z2) produced by the instrument. XYZ coordinates are
	## defined relative to the instrument, so they do not take into account heading, pitch and roll.
	## ENU utilizes the attitude measurements to provide an Earth-relative coordinate system."

	## These are the transformation matricies for up and down cast.
	## Beam 1: Forward
	## Beam 2: Port
	## Beam 3: Aft
	## Beam 4: Starboard

	##################################
	## Transformation matrix layout ##
	##################################
	# Beam: 1.   2.   3.   4.
	# X     1X.  2X.  3X.  4X.
	# Y     1Y.  2Y.  3Y.  4Y.
	# Z1    1Z1. 2Z1. 3Z1. 4Z1.
	# Z2    1Z2. 2Z2. 3Z2. 4Z2.
	##################################

	################# Input Variables #################
	## beam1vel     = single ping of velocity from beam 1
	## beam2vel     = single ping of velocity from beam 2
	## beam3vel     = single ping of velocity from beam 3
	## beam4vel     = single ping of velocity from beam 4
	## beam2xyz_mat = Static transformation matrix from beam to XYZ taking from AD2CP config
	## ahrs_rot_mat = Dynamic transformation matrix from XYZ to beam, changes depending on heading, pitch, and roll
	## pitch        = pitch in degrees

	############################################################################################################
	## First from beam to XYZ
	## If downcast, grab just beams 124 and correction transformation matrix

	if 'burst_beam2xyz' in ds.attrs:
		beam2xyz = ds.attrs['burst_beam2xyz']
	elif 'beam2xyz' in ds.attrs:
		beam2xyz = ds.attrs['beam2xyz']  # you might want to correct the attribute you're checking for here
	else:
		print('No beam transformation matrix info found')
	
	beam2xyz = beam2xyz.reshape(4,4)  # Because we know this configuration is a 4 beam AD2CP

	ds = ds.assign(UVelocity=ds["InterpVelocityBeam1"] *np.NaN)
	ds = ds.assign(VVelocity=ds["InterpVelocityBeam1"] *np.NaN)
	ds = ds.assign(WVelocity=ds["InterpVelocityBeam1"] *np.NaN)
	
	InterpVelocityBeam1 = ds.InterpVelocityBeam1.values
	InterpVelocityBeam2 = ds.InterpVelocityBeam2.values
	InterpVelocityBeam3 = ds.InterpVelocityBeam3.values
	InterpVelocityBeam4 = ds.InterpVelocityBeam4.values
	
	## Preallocate interpolated velocity outside of master xarray dataset for easy looping
	UVelocity = np.empty((len(ds.VelocityRange),len(ds.time)))
	VVelocity = np.empty((len(ds.VelocityRange),len(ds.time)))
	WVelocity = np.empty((len(ds.VelocityRange),len(ds.time)))
	
	## Set the empty variables = NaN
	UVelocity[:] = np.NaN
	VVelocity[:] = np.NaN
	WVelocity[:] = np.NaN
	
	## Pull these out of xarray out of loop
	AHRSRotationMatrix = ds.AHRSRotationMatrix.values
	print(AHRSRotationMatrix.shape)
	
	for x in np.arange(0,len(ds.time)):
		if ds.Pitch[x] < 0:
			tot_vel = np.column_stack((InterpVelocityBeam1[:, x], InterpVelocityBeam2[:, x],InterpVelocityBeam4[:, x]))
			beam2xyz_mat = beam2xyz[0:3, [0, 1, 3]]
		## If upcast, grab just beams 234 and correction transformation matrix
		elif ds.Pitch[x] > 0:
			tot_vel = np.column_stack((InterpVelocityBeam2[:, x], InterpVelocityBeam3[:, x],InterpVelocityBeam4[:, x]))
			beam2xyz_mat = beam2xyz[0:3, 1:4]
		## Not really sure what to do here, seems unlikely the pitch will be exactly equal to zero
		## but I already had it happen once in testing. Just going with the upcast solution.
		elif ds.Pitch[x] == 0:
			tot_vel = np.column_stack((InterpVelocityBeam2[:, x], InterpVelocityBeam3[:, x],InterpVelocityBeam4[:, x]))
			beam2xyz_mat = beam2xyz[0:3, 1:4]

		## If instrument is pointing down, bit 0 in status is equal to 1, rows 2 and 3 must change sign.
		## Hard coding this because of glider configuration which is pointing down.
		beam2xyz_mat[1,:] = -beam2xyz_mat[1,:]
		beam2xyz_mat[2,:] = -beam2xyz_mat[2,:]

		## Now convert to XYZ
		#print(beam2xyz_mat.shape,tot_vel.shape)
		
		xyz = np.dot(beam2xyz_mat,tot_vel.T)

		## Grab AHRS rotation matrix for this ping
		xyz2enuAHRS = AHRSRotationMatrix[:,x].reshape(3,3)

		## Now convert XYZ velocities to ENU, where enu[0,:] is U, enu[1,:] is V, and enu[2,:] is W velocities.
		enu = np.array(np.dot(xyz2enuAHRS,xyz))
		UVelocity[:,x] = enu[0,:].ravel()
		VVelocity[:,x] = enu[1,:].ravel()
		WVelocity[:,x] = enu[2,:].ravel()
	
	ds['UVelocity'].values = UVelocity
	ds['VVelocity'].values = VVelocity
	ds['WVelocity'].values = WVelocity
	
	return(ds)


def inversion(U,V,dz,u_daverage,v_daverage,bins,depth, wDAC, wSmoothness):
	global O_ls, G_ls, bin_new

	## Feb-2021 jgradone@marine.rutgers.edu Initial
	## Jul-2021 jgradone@marine.rutgers.edu Updates for constraints
	## Jun-2022 jgradone@marine.rutgers.edu Corrected dimensions and indexing of G matrix
	## Jun-2022 jgradone@marine.rutgers.edu Added curvature minimizing constraint and constraint weights

	## Purpose: Take velocity measurements from glider mounted ADCP and compute
	# shear profiles

	## Outputs:
	# O_ls is the ocean velocity profile
	# G_ls is the glider velocity profile
	# bin_new are the bin centers for the point in the profiles
	# obs_per_bin is the number of good velocity observations per final profile bin

	## Inputs:
	# dz is desired vertical resolution, should not be smaller than bin length
	# U is measured east-west velocities from ADCP
	# V is measured north-south velocities from ADCP
	# bins is the bin depths for the U and V measurements
	# uv_daverage is depth averaged velocity (Set to 0 for real-time)
	# depth is the depth of the glider measured by the ADCP
	# wDAC is the weight of the DAC constraint (5 per Todd et al. 2017)
	# wSmoothness is the weight of the curvature minimizing contraint (1 per Todd et al. 2017)


	#########################################################################
	## These steps filter for NAN rows and columns so they are technically QAQC
	## but I think the best place to put them is inthe inversion function because
	## if there are NaNs still present in the data here, it will throw everything off
	## These steps are HUGE for efficiency because it reduces the size of the G
	## matrix as much as possible.

	## This determines the rows (bins) where all the columns are NaN
	nanind = np.where( (np.sum(np.isnan(U),axis=1)/U.shape[1]) == 1)[0]
	if len(nanind) > 0:
		U = np.delete(U,nanind,axis=0)
		V = np.delete(V,nanind,axis=0)
		bins = np.delete(bins,nanind)

	## Do the same thing with individual ensembles. Note: need to remove the corresponding
	## ensemble pressure reading to ensure correction dimensions and values.
	nanind = np.where((np.sum(np.isnan(U),axis=0)/U.shape[0]) == 1)[0]
	if len(nanind) > 0:
		U = np.delete(U,nanind,axis=1)
		V = np.delete(V,nanind,axis=1)
		depth = np.delete(depth,nanind)
	##########################################################################


	##########################################################################
	# Take difference between bin lengths for bin size [m]
	bin_size = np.diff(bins)[0]
	bin_num = len(bins)
	# This creates a grid of the ACTUAL depths of the ADCP bins by adding the
	# depths of the ADCP bins to the actual depth of the instrument
	[bdepth,bbins]=np.meshgrid(depth,bins)
	bin_depth = bdepth+bbins
	Z = bin_depth
	# Calculate the maximum depth of glider which is different than maximum ADCP bin depth
	ZmM = np.nanmax(depth)
	##########################################################################


	##########################################################################
	# Set knowns from Equations 19 from Visbeck (2002) page 800
	# Maximum number of observations (nd) is given by the number of velocity
	# estimates per ping (nbin) times the number of profiles per cast (nt)
	nbin = U.shape[0]  # number of programmed ADCP bins per individual profile
	nt   = U.shape[1]  # number of individual velocity profiles
	nd   = nbin*nt      # G dimension (1)

	# Define the edges of the bins
	bin_edges = np.arange(0,math.floor(np.max(bin_depth)),dz).tolist()

	# Check that each bin has data in it
	bin_count = np.empty(len(bin_edges)-1) # Preallocate memory
	bin_count[:] = np.NaN

	for k in np.arange(len(bin_edges))[:-1]:
		# Create index of depth values that fall inside the bin edges
		ii = np.where((bin_depth > bin_edges[k]) & (bin_depth < bin_edges[k+1]))
		bin_count[k] = len(bin_depth[ii])
		ii = []

	# Create list of bin centers
	bin_new = [x+dz/2 for x in bin_edges[:-1]]

	# Calculate which FINAL solution bin is deeper than the maximum depth of the glider
	# This is done so that the depth averaged velocity constraint is only applied to bins shallower than this depth
	depth_ind = len(np.where(bin_new>ZmM)[0])
	# Chop off the top of profile if no data
	ind = np.argmax(bin_count > 0) # Stops at first index greater than 0
	bin_new = bin_new[ind:]        # Removes all bins above first with data
	z1 = bin_new[0]                # Depth of center of first bin with data
	##########################################################################


	##########################################################################
	# Create and populate G
	nz = len(bin_new)  # number of ocean velocities desired in output profile
	nm = nt + nz       # G dimension (2), number of unknowns
	# Let's build the corresponding coefficient matrix G
	G = scipy.sparse.lil_matrix((nd, nm), dtype=float)

	# Indexing of the G matrix was taken from Todd et al. 2012
	for ii in np.arange(0,nt):           # Number of ADCP ensembles per segment
		for jj in np.arange(0,nbin):     # Number of measured bins per ensemble

			# Uctd part of matrix
			G[(nbin*(ii))+jj,ii] = -1
			# This will fill in the Uocean part of the matrix. It loops through
			# all Z members and places them in the proper location in the G matrix
			# Find the difference between all bin centers and the current Z value
			dx = abs(bin_new-Z[jj,ii])
			# Find the minimum of these differences
			minx = np.nanmin(dx)
			# Finds bin_new index of the first match of Z and bin_new
			idx = np.argmin(dx-minx)

			# Uocean part of matrix
			G[(nbin*(ii))+jj,(nt)+idx] = 1

			del dx, minx, idx

	##########################################################################
	# Reshape U and V into the format of the d column vector (order='F')
	# Based on how G is made, d needs to be ensembles stacked on one another vertically
	d_u = U.flatten(order='F')
	d_v = V.flatten(order='F')

	##########################################################################
	## This chunk of code containts the constraints for depth averaged currents
	# Make sure the constraint is only applied to the final ocean velocity bins that the glider dives through
	# Don't apply it to the first bin and don't apply it to the bins below the gliders dive depth
	constraint = np.concatenate(([np.zeros(nt)], [0], [np.tile(dz,nz-(1+depth_ind))], [np.zeros(depth_ind)]), axis=None)

	# Ensure the L^2 norm of the constraint equation is unity
	constraint_norm = np.linalg.norm(constraint/ZmM)
	C = 1/constraint_norm
	constraint_normalized = (C/ZmM)*constraint ## This is now equal to 1 (unity)
	# Build Gstar and add weight from todd 2017
	## Some smarts would be to calculate signal to noise ratio first
	Gstar = scipy.sparse.vstack((G,wDAC*constraint_normalized), dtype=float)


	# Add the constraint for the depth averaged velocity from Todd et al. (2017)
	du = np.concatenate(([d_u],[wDAC*C*u_daverage]), axis=None)
	dv = np.concatenate(([d_v],[wDAC*C*v_daverage]), axis=None)
	d = np.array(list(map(complex,du, dv)))


	##########################################################################
	#### THIS removes all NaN elements of d AND Gstar so the inversion doesn't blow up with NaNs
	ind2 = np.where(np.isnan(d)==True)[0]
	d = np.delete(d,ind2)

	def delete_rows_csr(mat, indices):
		"""
		Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
		"""
		if not isinstance(mat, scipy.sparse.csr_matrix):
			raise ValueError("works only for CSR format -- use .tocsr() first")
		indices = list(indices)
		mask = np.ones(mat.shape[0], dtype=bool)
		mask[indices] = False
		return mat[mask]

	Gstar = delete_rows_csr(Gstar.tocsr().copy(),ind2)

	#########################################################################
	# Test adding depth for tracking bin location
	# d is ensembles stacked on one another vertically so same for Z (order='F')
	Z_filt = Z.flatten(order='F')
	Z_filt = np.delete(Z_filt,ind2)
	Z_filt = np.concatenate(([Z_filt],[0]), axis=None)

	##########################################################################
	## Calculation the number of observations per bin
	obs_per_bin = np.empty(len(bin_new))
	obs_per_bin[:] = np.NaN

	for x in np.arange(0,nz):
		rows_where_nt_not_equal_zero = np.where(Gstar.tocsr()[0:Z_filt.shape[0],nt+x].toarray() > 0 )[0]
		obs_per_bin[x] = len(rows_where_nt_not_equal_zero)

	## If there is no data in the last bin, drop that from the G matrix, bin_new, and obs_per_bin
	if obs_per_bin[-1] == 0:
		Gstar.tocsr()[:,:-1]
		bin_new = bin_new[:-1]
		obs_per_bin = obs_per_bin[:-1]
		## Update nz and nt
		nz = len(bin_new)
		nt = Gstar.shape[1]-nz

	##########################################################################
	## Smoothness constraint
	## Only do this is the smoothness constraint is set
	if wSmoothness > 0:
		## Add a vector of zerosm the length of nz, twice to the bottom of the data column vector
		d = np.concatenate(([d],[np.zeros(nz)],[np.zeros(nz)]), axis=None)
		## Constraint on smoothing Uocean side of matrix
		smoothing_matrix_Uocean = scipy.sparse.diags([[-1],[2],[-1]], [0,1,2], shape=(nz,nz))
		smoothing_matrix1 = scipy.sparse.hstack((np.zeros((nz,nt)),smoothing_matrix_Uocean), dtype=float)
		## Constraint on smoothing Uglider side of matrix
		smoothing_matrix_Uglider = scipy.sparse.diags([[-1],[2],[-1]], [0,1,2], shape=(nz,nt))
		smoothing_matrix2 = scipy.sparse.hstack((smoothing_matrix_Uglider,np.zeros((nz,nz))), dtype=float)
		Gstar = scipy.sparse.vstack((Gstar,wSmoothness*smoothing_matrix1,wSmoothness*smoothing_matrix2), dtype=float)


	##########################################################################
	## Run the Least-Squares Inversion!
	x = lsqr(Gstar, d)[0]

	O_ls = x[nt:]
	G_ls = x[0:nt]
	##########################################################################

	return(O_ls, G_ls, bin_new, obs_per_bin)
