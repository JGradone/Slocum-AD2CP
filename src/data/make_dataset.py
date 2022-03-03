## Don't know if this actually needs to be here or just in the main script
import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
import scipy.interpolate as interp
import math


##################################################################################################

def check_max_beam_range(beam,bins):
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.NaN
    elif len(ind1) > 0:
        ind2 = np.max(ind1[:,0])
        beam_range = bins[ind2]
    return(beam_range)

def check_max_beam_range_bins(beam,bins):
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.NaN
    elif len(ind1) > 0:
        beam_range = np.max(ind1[:,0])
    return(beam_range)



##################################################################################################

def check_mean_beam_range(beam,bins):
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.NaN
    elif len(ind1) > 0:
        ind2 = round(np.nanmean(ind1[:,0]))
        beam_range = bins[ind2]
    return(beam_range)

def check_mean_beam_range_bins(beam,bins):
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.NaN
    elif len(ind1) > 0:
        beam_range = np.nanmean(ind1[:,0])
    return(beam_range)







##################################################################################################

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

    beam2xyz = ds.attrs['burst_beam2xyz']
    beam2xyz = beam2xyz.reshape(4,4)  # Because we know this configuration is a 4 beam AD2CP

    #     ds['UVelocity'] = np.empty((ds.VelocityBeam1.shape[0],ds.VelocityBeam1.shape[1]))
    #     ds['VVelocity'] = np.empty((ds.VelocityBeam1.shape[0],ds.VelocityBeam1.shape[1]))
    #     ds['WVelocity'] = np.empty((ds.VelocityBeam1.shape[0],ds.VelocityBeam1.shape[1]))
    #     ds['UVelocity'][:] = np.NaN()
    #     ds['VVelocity'][:] = np.NaN()
    #     ds['WVelocity'][:] = np.NaN()


    ds = ds.assign(UVelocity=ds["VelocityBeam1"] *np.NaN)
    ds = ds.assign(VVelocity=ds["VelocityBeam1"] *np.NaN)
    ds = ds.assign(WVelocity=ds["VelocityBeam1"] *np.NaN)



    for x in np.arange(0,len(ds.time)):
        if ds.Pitch[x] < 0:
            tot_vel = np.matrix([ds.VelocityBeam1[:,x], ds.VelocityBeam2[:,x], ds.VelocityBeam4[:,x]])
            beam2xyz_mat = beam2xyz[0:3,(0,1,3)]
        ## If upcast, grab just beams 234 and correction transformation matrix
        elif ds.Pitch[x] > 0:
            tot_vel = np.matrix([ds.VelocityBeam2[:,x], ds.VelocityBeam3[:,x], ds.VelocityBeam4[:,x]])
            beam2xyz_mat = beam2xyz[0:3,1:4]
        elif ds.Pitch[x] == 0: ## Not really sure what to do here, seems unlikely the pitch will be exactly equal to zero
                         ## but I already had it happen once in testing. Just going with the upcast solution.
            tot_vel = np.matrix([ds.VelocityBeam1[:,x], ds.VelocityBeam2[:,x], ds.VelocityBeam4[:,x]])
            beam2xyz_mat = beam2xyz[0:3,1:4]

        ## If instrument is pointing down, bit 0 in status is equal to 1, rows 2 and 3 must change sign.
        ## Hard coding this because of glider configuration.
        beam2xyz_mat[1,:] = -beam2xyz_mat[1,:]
        beam2xyz_mat[2,:] = -beam2xyz_mat[2,:]

        ## Now convert to XYZ
        xyz     = beam2xyz_mat*tot_vel

        ## Grab AHRS rotation matrix for this ping
        xyz2enuAHRS = ds.AHRSRotationMatrix[:,x].values.reshape(3,3)

        ## Now convert XYZ velocities to ENU, where enu[0,:] is U, enu[1,:] is V, and enu[2,:] is W velocities.
        enu = np.array(xyz2enuAHRS*xyz)
        ds['UVelocity'][:,x] = enu[0,:].ravel()
        ds['VVelocity'][:,x] = enu[1,:].ravel()
        ds['WVelocity'][:,x] = enu[2,:].ravel()
    
    return(ds)








##################################################################################################

def binmap_adcp(ds):
    ## bins = bin depths output from ADCP
    ## true_depth = Actual bin depths calculated with function cell_vert based on pitch and roll    
    ## Comment this out better!      

    for i in np.arange(0,len(ds.time)):
        TrueDepthBeam1 = cell_vert(ds['Pitch'][i], ds['Roll'][i], ds['Velocity Range'], beam_number=1)
        ds.VelocityBeam1.values[:,i] = interp.griddata(TrueDepthBeam1, ds['VelocityBeam1'][:,i], ds['Velocity Range'],method='nearest')
        #ds['TrueDepthBeam1'][i,:] = TrueDepthBeam1

        TrueDepthBeam2 = cell_vert(ds['Pitch'][i], ds['Roll'][i], ds['Velocity Range'], beam_number=2)
        ds.VelocityBeam2.values[:,i] = interp.griddata(TrueDepthBeam2, ds['VelocityBeam2'][:,i], ds['Velocity Range'],method='nearest')
        #ds['TrueDepthBeam2'][i,:] = TrueDepthBeam2

        TrueDepthBeam3 = cell_vert(ds['Pitch'][i], ds['Roll'][i], ds['Velocity Range'], beam_number=3)
        ds.VelocityBeam3.values[:,i] = interp.griddata(TrueDepthBeam3, ds['VelocityBeam3'][:,i], ds['Velocity Range'],method='nearest')
        #ds['TrueDepthBeam3'][i,:] = TrueDepthBeam3

        TrueDepthBeam4 = cell_vert(ds['Pitch'][i], ds['Roll'][i], ds['Velocity Range'], beam_number=4)
        ds.VelocityBeam4.values[:,i] = interp.griddata(TrueDepthBeam4, ds['VelocityBeam4'][:,i], ds['Velocity Range'],method='nearest')
        #ds['TrueDepthBeam4'][i,:] = TrueDepthBeam4
        
    return ds









##################################################################################################

def cell_vert(pitch, roll, velocity_range, beam_number):
    ## Calculate a vertical displacement below instrument for
    ## each adcp bin adjusting for pitch and roll (in degrees)
    ## Positive roll: Port wing up
    ## Positive pitch: Pitch up  
    
    ## Beam 1: Forward   (47.5 degrees off horizontal)
    ## Beam 2: Port      (25 degrees off horizontal)
    ## Beam 3: Aft       (47.5 degrees off horizontal)
    ## Beam 4: Starboard (25 degrees off horizontal)
    
    ## Beam angle is only incorporated in pitch for Beams 1 & 3 and
    ## in roll for Beams 2 & 4

    if beam_number == 1:
        beam_angle = 47.5
        pitch_adjusted = velocity_range * np.sin(np.deg2rad(90 + beam_angle + pitch))
        z = (pitch_adjusted * np.sin(np.deg2rad(90 - roll)))
    
    elif beam_number == 2:
        beam_angle = 25
        pitch_adjusted = velocity_range * np.sin(np.deg2rad(90 - pitch))
        z = (pitch_adjusted * np.sin(np.deg2rad(90 + roll + beam_angle)))
    
    elif beam_number == 3:
        beam_angle = 47.5
        pitch_adjusted = velocity_range * np.sin(np.deg2rad(90 - beam_angle + pitch))
        z = (pitch_adjusted * np.sin(np.deg2rad(90 - roll)))
        
    elif beam_number == 4:
        beam_angle = 25
        pitch_adjusted = velocity_range * np.sin(np.deg2rad(90 - pitch))
        z = (pitch_adjusted * np.sin(np.deg2rad(90 - roll + beam_angle)))
    
    else:
        print("Must specify beam number")
        exit(1)
    
    return z.transpose()







##################################################################################################

def correct_sound_speed(ds):
    default_speedofsound = 1500
    ds["VelocityBeam1"] = ds.VelocityBeam1*(ds.SpeedOfSound/default_speedofsound)
    ds["VelocityBeam2"] = ds.VelocityBeam2*(ds.SpeedOfSound/default_speedofsound)
    ds["VelocityBeam3"] = ds.VelocityBeam3*(ds.SpeedOfSound/default_speedofsound)
    ds["VelocityBeam4"] = ds.VelocityBeam4*(ds.SpeedOfSound/default_speedofsound)
    return ds




##################################################################################################

def qaqc_pre_coord_transform(ds):
#     # For determining upcast vs downcast
#     pitch_threshold = 0
    
#     for i in np.arange(0,len(ds.time)):
#         if ds.Pitch[i] > pitch_threshold: # Upcast so use beams 234
#             # If the return is above the threshold, flag the data
#             amp2_ind = ds.AmplitudeBeam2.values[:,i] > max_amplitude 
#             amp3_ind = ds.AmplitudeBeam3.values[:,i] > max_amplitude 
#             amp4_ind = ds.AmplitudeBeam4.values[:,i] > max_amplitude
#             amp_ind  = []
#             amp_ind  = amp2_ind + amp3_ind + amp4_ind
#             ds.VelocityBeam2.values[amp_ind,i] = np.NaN
#             ds.VelocityBeam3.values[amp_ind,i] = np.NaN
#             ds.VelocityBeam4.values[amp_ind,i] = np.NaN
#         elif ds.Pitch[i] < pitch_threshold: # Downcast so use beams 124
#             # If the return is above the threshold, flag the data
#             amp1_ind = ds.AmplitudeBeam1.values[:,i] > max_amplitude 
#             amp2_ind = ds.AmplitudeBeam2.values[:,i] > max_amplitude 
#             amp4_ind = ds.AmplitudeBeam4.values[:,i] > max_amplitude
#             amp_ind  = []
#             amp_ind  = amp1_ind + amp2_ind + amp4_ind
#             ds.VelocityBeam1.values[amp_ind,i] = np.NaN
#             ds.VelocityBeam2.values[amp_ind,i] = np.NaN
#             ds.VelocityBeam4.values[amp_ind,i] = np.NaN

    # Set low correlation threshold
    corr_threshold = 50
    ## Need the .values here because xarray is funky
    ds.VelocityBeam1.values[ds.CorrelationBeam1.values < corr_threshold] = np.NaN
    ds.VelocityBeam2.values[ds.CorrelationBeam2.values < corr_threshold] = np.NaN
    ds.VelocityBeam3.values[ds.CorrelationBeam3.values < corr_threshold] = np.NaN
    ds.VelocityBeam4.values[ds.CorrelationBeam4.values < corr_threshold] = np.NaN
    
    # Set extreme amplitude threshold
    max_amplitude = 75 # [dB]
    ## Need the .values here because xarray is funky
    ds.VelocityBeam1.values[ds.AmplitudeBeam1.values > max_amplitude] = np.NaN
    ds.VelocityBeam2.values[ds.AmplitudeBeam2.values > max_amplitude] = np.NaN
    ds.VelocityBeam3.values[ds.AmplitudeBeam3.values > max_amplitude] = np.NaN
    ds.VelocityBeam4.values[ds.AmplitudeBeam4.values > max_amplitude] = np.NaN
    
    return(ds)







##################################################################################################
                     
def inversion(U,V,H,dz,u_daverage,v_daverage,bins,depth):
    global O_ls, G_ls, bin_new    
    
    ## Feb-2021 jgradone@marine.rutgers.edu Initial
    ## Jul-2021 jgradone@marine.rutgers.edu Updates for constraints
    
    ## Purpose: Take velocity measurements from glider mounted ADCP and compute
    # shear profiles
    
    ## Outputs:
    # O_ls is the ocean velocity profile
    # G_ls is the glider velocity profile
    # bin_new are the bin centers for the point in the profiles
    # C is the constant used in the constraint equation (Not applicable for
    # real-time processing)
    
    ## Inputs:
    # dz is desired vertical resolution, should not be smaller than bin length
    # H is the max depth of the water column
    # U is measured east-west velocities from ADCP
    # V is measured north-south velocities from ADCP
    # Z is the measurement depths of U and V
    # uv_daverage is depth averaged velocity (Set to 0 for real-time)
    
    ##########################################################################        
    # Take difference between bin lengths for bin size [m]
    bin_size = np.diff(bins)[0]
    bin_num = len(bins)

    # This creates a grid of the ACTUAL depths of the ADCP bins by adding the
    # depths of the ADCP bins to the actual depth of the instrument
    [bdepth,bbins]=np.meshgrid(depth,bins)
    bin_depth = bdepth+bbins  
    Z = bin_depth

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

    # Chop off the top of profile if no data
    ind = np.argmax(bin_count > 0) # Stops at first index greater than 0
    bin_new = bin_new[ind:]        # Removes all bins above first with data
    z1 = bin_new[0]                # Depth of center of first bin with data

    # Create and populate G
    nz = len(bin_new)  # number of ocean velocities desired in output profile
    nm = nz + nt       # G dimension (2), number of unknowns
    # Let's build the corresponding coefficient matrix G 
    G = np.zeros((nd,nm))

    # Indexing of the G matrix was taken from Todd et al. 2012
    for ii in np.arange(0,nt):           # Number of ADCP profiles per cast
        for jj in np.arange(0,nbin):     # Number of measured bins per profile 
            
            # Uctd part of matrix
            G[(nbin*(ii-1))+jj,ii] = 1

            # This will fill in the Uocean part of the matrix. It loops through
            # all Z members and places them in the proper location in the G matrix

            # Find the difference between all bin centers and the current Z value        
            dx = abs(bin_new-Z[ii,jj])

            # Find the minimum of these differences
            minx = np.nanmin(dx)

            # Finds bin_new index of the first match of Z and bin_new    
            idx = np.argmin(dx-minx)

            G[(nbin*(ii-1))+jj,nt+idx] = 1
            del dx, minx, idx


    # Reshape U and V into the format of the d column vector
    d_u = np.flip(U.transpose(),axis=0)
    d_u = d_u.flatten()
    d_v = np.flip(V.transpose(),axis=0)
    d_v = d_v.flatten()


    ##########################################################################
    ## This chunk of code containts the constraints for depth averaged currents
    ## which we likely won't be using for the real-time processing

    # Need to calculate C (Todd et al. 2017) based on our inputs 
    # This creates a row that has the same # of columns as G. The elements
    # of the row follow the trapezoid rule which is used because of the
    # extension of the first bin with data to the surface. The last entry of
    # the row corresponds to the max depth reached by the glider, any bins
    # below that should have already been removed.

    constraint = np.concatenate(([np.zeros(nt)], [z1/2], [z1/2+dz/2], [[dz]*(nz-3)], [dz/2]), axis=None)

    # To find C, we use the equation of the norm and set norm=1 because we
    # desire unity. The equation requires we take the sum of the squares of the
    # entries in constraint.

    sqr_constraint = constraint*constraint
    sum_sqr_constraint = np.sum(sqr_constraint)

    # Then we can solve for the value of C needed to maintain unity 

    C = H*(1/np.sqrt(sum_sqr_constraint))

    # This is where you would add the constraint for the depth averaged
    # velocity from Todd et al., (2011/2017)

    # OG
    du = np.concatenate(([d_u],[C*u_daverage]), axis=None)
    dv = np.concatenate(([d_v],[C*v_daverage]), axis=None)

    # Build Gstar
    # Keep this out because not using depth averaged currents
    Gstar = np.vstack((G, (C/H)*constraint))

    ##########################################################################

    # Build the d matrix
    d = list(map(complex,du, dv))

    ##### Inversion!
    ## If want to do with a sparse matrix sol'n, look at scipy
    #Gs = scipy.sparse(Gstar)
    Gs = Gstar

    ms = np.linalg.solve(np.dot(Gs.conj().transpose(),Gs),Gs.conj().transpose())

    ## This is a little clunky but I think the dot product fails because of
    ## NaN's in the d vector. So, this code will replace NaN's with 0's just
    ## for that calculation    
    sol = np.dot(ms,np.where(np.isnan(d),0,d))

    O_ls = sol[nt:]   # Ocean velocity
    G_ls = sol[0:nt]  # Glider velocity
    return(O_ls, G_ls, bin_new)



