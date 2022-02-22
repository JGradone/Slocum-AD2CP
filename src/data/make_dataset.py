## Don't know if this actually needs to be here or just in the main script
import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
import scipy.interpolate as interp



##################################################################################################

def ad2cp_nc_to_xarray(filename):

    ds = nc.Dataset(filename)

    tm                 = np.array(ds['Data']['Burst'].variables['time'])
    tm                 = pd.to_datetime(tm, unit='s').values
    pressure           = np.array(ds['Data']['Burst'].variables['Pressure'])
    heading            = np.array(ds['Data']['Burst'].variables['Heading'])
    pitch              = np.array(ds['Data']['Burst'].variables['Pitch'])
    roll               = np.array(ds['Data']['Burst'].variables['Roll'])
    roll               = roll-180 # Note: the AD2CP baseline roll is at 180 so must subtract 180 from roll first
    bins               = np.array(ds['Data']['Burst'].variables['Velocity Range'])
    bins               = np.round(bins,2) ## Doing this because there are some weird 3.999's and I want a regular 0.5 m grid
    speedofsound       = np.array(ds['Data']['Burst'].variables['SpeedOfSound'])
    error              = np.array(ds['Data']['Burst'].variables['Error'])
    status             = np.array(ds['Data']['Burst'].variables['Status'])
    cellsize           = np.array(ds['Data']['Burst'].variables['CellSize'])
    numberofcells      = np.array(ds['Data']['Burst'].variables['NumberofCells'])
    blanking           = np.array(ds['Data']['Burst'].variables['Blanking'])
    nominalcorrelation = np.array(ds['Data']['Burst'].variables['NominalCorrelation'])
    ambiguity          = np.array(ds['Data']['Burst'].variables['Ambiguity'])
    ensemblecount      = np.array(ds['Data']['Burst'].variables['EnsembleCount'])
    unshiftedroll      = np.array(ds['Data']['Burst'].variables['UnshiftedRoll'])
    beam2xyz           = np.array(ds['Config'].burst_beam2xyz)
    beam2xyz           = beam2xyz.reshape(4,4)   # Because we know this configuration is a 4 beam AD2CP
    ahrs_rot_mat       = np.array(ds['Data']['Burst'].variables['AHRSRotationMatrix'])

    magnetometerx = np.array(ds['Data']['Burst'].variables['MagnetometerX'])
    magnetometery = np.array(ds['Data']['Burst'].variables['MagnetometerY'])
    magnetometerz = np.array(ds['Data']['Burst'].variables['MagnetometerZ'])

    accelerometerx = np.array(ds['Data']['Burst'].variables['AccelerometerX'])
    accelerometery = np.array(ds['Data']['Burst'].variables['AccelerometerY'])
    accelerometerz = np.array(ds['Data']['Burst'].variables['AccelerometerZ'])

    beam1vel = np.array(ds['Data']['Burst'].variables['VelocityBeam1']).transpose()
    beam2vel = np.array(ds['Data']['Burst'].variables['VelocityBeam2']).transpose()
    beam3vel = np.array(ds['Data']['Burst'].variables['VelocityBeam3']).transpose()
    beam4vel = np.array(ds['Data']['Burst'].variables['VelocityBeam4']).transpose()

    beam1cor = np.array(ds['Data']['Burst'].variables['CorrelationBeam1']).transpose()
    beam2cor = np.array(ds['Data']['Burst'].variables['CorrelationBeam2']).transpose()
    beam3cor = np.array(ds['Data']['Burst'].variables['CorrelationBeam3']).transpose()
    beam4cor = np.array(ds['Data']['Burst'].variables['CorrelationBeam4']).transpose()

    beam1amp = np.array(ds['Data']['Burst'].variables['AmplitudeBeam1']).transpose()
    beam2amp = np.array(ds['Data']['Burst'].variables['AmplitudeBeam2']).transpose()
    beam3amp = np.array(ds['Data']['Burst'].variables['AmplitudeBeam3']).transpose()
    beam4amp = np.array(ds['Data']['Burst'].variables['AmplitudeBeam4']).transpose()


    df = xr.Dataset(
        {"heading" : (("time"), heading),
         "depth" : (("time"), pressure),
         "pitch" : (("time"), pitch),
         "roll" : (("time"), roll),
         "speedofsound" : (("time"), speedofsound),
         "error" : (("time"), error),
         "status" : (("time"), status),
         "cellsize" : (("time"), cellsize),
         "numberofcells" : (("time"), numberofcells),
         "blanking" : (("time"), blanking),
         "nominalcorrelation" : (("time"), nominalcorrelation),
         "ambiguity" : (("time"), ambiguity),
         "ensemblecount" : (("time"), ensemblecount),
         "unshiftedroll" : (("time"), unshiftedroll),

         "beam2xyz" : (("beams1","beams2"), beam2xyz),
         "ahrs_rot_mat" : (("time","ahrs_rot_mat_dim"),ahrs_rot_mat),


         "magnetometerx" : (("time"), magnetometerx),
         "magnetometery" : (("time"), magnetometery),
         "magnetometerz" : (("time"), magnetometerz),
         "accelerometerx" : (("time"), accelerometerx),
         "accelerometery" : (("time"), accelerometery),
         "accelerometerz" : (("time"), accelerometerz),     

         "beam1vel" : (("bins","time"),beam1vel),
         "beam2vel" : (("bins","time"),beam2vel),
         "beam3vel" : (("bins","time"),beam3vel),
         "beam4vel" : (("bins","time"),beam4vel),
         "beam1cor" : (("bins","time"),beam1cor),
         "beam2cor" : (("bins","time"),beam2cor),
         "beam3cor" : (("bins","time"),beam3cor),
         "beam4cor" : (("bins","time"),beam4cor),
         "beam1amp" : (("bins","time"),beam1amp),
         "beam2amp" : (("bins","time"),beam2amp),
         "beam3amp" : (("bins","time"),beam3amp),
         "beam4amp" : (("bins","time"),beam4amp)},    
                 coords   = {"bins":bins,"time":tm,"beams1":np.arange(1,5),"beams2":np.arange(1,5),"ahrs_rot_mat_dim":np.arange(0,9)}
                           )
    ds.close()
    return df






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

def beam2enu(beam1vel,beam2vel,beam3vel,beam4vel,beam2xyz,ahrs_rot_mat,pitch):
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
    if pitch < 0:
        tot_vel = np.matrix([beam1vel, beam2vel, beam4vel])
        beam2xyz_mat = beam2xyz[0:3,(0,1,3)]
    ## If upcast, grab just beams 234 and correction transformation matrix
    elif pitch > 0:
        tot_vel = np.matrix([beam2vel, beam3vel, beam4vel])
        beam2xyz_mat = beam2xyz[0:3,1:4]
    elif pitch == 0: ## Not really sure what to do here, seems unlikely the pitch will be exactly equal to zero
                     ## but I already had it happen once in testing.
        tot_vel = np.matrix([beam2vel, beam3vel, beam4vel])
        beam2xyz_mat = beam2xyz[0:3,1:4]
    
    ## If instrument is pointing down, bit 0 in status is equal to 1, rows 2 and 3 must change sign.
    ## Hard coding this because of glider configuration.
    beam2xyz_mat[1,:] = -beam2xyz_mat[1,:]
    beam2xyz_mat[2,:] = -beam2xyz_mat[2,:]
    
    ## Now convert to XYZ
    xyz     = beam2xyz_mat*tot_vel
     
    ## Grab AHRS rotation matrix for this ping
    xyz2enuAHRS = ahrs_rot_mat.reshape(3,3)
    
    ## Now convert XYZ velocities to ENU, where enu[0,:] is U, enu[1,:] is V, and enu[2,:] is W velocities.
    enu = xyz2enuAHRS*xyz
    
    return(enu)









##################################################################################################

#def binmap_adcp(beamvelocity, pitch, roll, bins):
def binmap_adcp(beamvelocity, pitch, roll, bins, rng_cells, blanking_dist, beam_number):
    ## bins = bin depths output from ADCP
    ## true_depth = Actual bin depths calculated with function cell_vert based on pitch and roll
    
    ## Comment this out better!
    
    #true_depth = cell_vert(pitch, roll, bins)
    
    true_depth = cell_vert(pitch, roll, rng_cells, blanking_dist, beam_number)

    gridvel = interp.griddata(true_depth, beamvelocity, bins, method='nearest')
    
    return gridvel










# ##################################################################################################
# ## Realized that the bins is really the projection along the vertical axis and it takes into account
# ## the transducer geometry already. For this function, just need to correct for vehicle pitch and roll.

# def cell_vert(pitch, roll, bins):
#     ## Calculate a vertical displacement below instrument for
#     ## each adcp bin adjusting for pitch and roll (in degrees)
#     ## Positive roll: Port wing up
#     ## Positive pitch: Pitch up  
    
#     pitch_adjusted = bins * np.sin(np.deg2rad(90-pitch))
#     z = pitch_adjusted * np.sin(np.deg2rad(90-roll))    
#     return z



def cell_vert(pitch, roll, rng_cells, blanking_dist, beam_number):
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
        pitch_adjusted = rng_cells * np.sin(np.deg2rad(90 + beam_angle + pitch))
        z = blanking_dist + (pitch_adjusted * np.sin(np.deg2rad(90 - roll)))
    
    elif beam_number == 2:
        beam_angle = 25
        pitch_adjusted = rng_cells * np.sin(np.deg2rad(90 - pitch))
        z = blanking_dist + (pitch_adjusted * np.sin(np.deg2rad(90 + roll + beam_angle)))
    
    elif beam_number == 3:
        beam_angle = 47.5
        pitch_adjusted = rng_cells * np.sin(np.deg2rad(90 - beam_angle + pitch))
        z = blanking_dist + (pitch_adjusted * np.sin(np.deg2rad(90 - roll)))
        
    elif beam_number == 4:
        beam_angle = 25
        pitch_adjusted = rng_cells * np.sin(np.deg2rad(90 - pitch))
        z = blanking_dist + (pitch_adjusted * np.sin(np.deg2rad(90 - roll + beam_angle)))
    
    else:
        print("Must specify beam number")
        exit(1)
    
    return z




##################################################################################################

def correct_sound_speed(beamvelocity, real_speedofsound, default_speedofsound):
    vel = beamvelocity*(real_speedofsound/default_speedofsound)
    return vel








##################################################################################################

def qaqc_pre_coord_transform(beam1vel,beam2vel,beam3vel,beam4vel,beam1amp,beam2amp,beam3amp,beam4amp,beam1cor,beam2cor,beam3cor,beam4cor,pitch):
    pitch_threshold = 0
    
    # Set extreme amplitude threshold
    max_amplitude = 75 # [dB]
    
    for i in np.arange(0,beam1vel.shape[1]):
        if pitch[i] > pitch_threshold: # Upcast so use beams 234
            # If the return is above the threshold, flag the data
            amp2_ind = beam2amp[:,i] > max_amplitude 
            amp3_ind = beam3amp[:,i] > max_amplitude 
            amp4_ind = beam4amp[:,i] > max_amplitude
            amp_ind  = []
            amp_ind  = amp2_ind + amp3_ind + amp4_ind
            beam2vel[amp_ind,i] = np.NaN
            beam3vel[amp_ind,i] = np.NaN
            beam4vel[amp_ind,i] = np.NaN
        elif pitch[i] < pitch_threshold: # Downcast so use beams 124
            # If the return is above the threshold, flag the data
            amp1_ind = beam1amp[:,i] > max_amplitude 
            amp2_ind = beam2amp[:,i] > max_amplitude 
            amp4_ind = beam4amp[:,i] > max_amplitude
            amp_ind  = []
            amp_ind  = amp1_ind + amp2_ind + amp4_ind
            beam1vel[amp_ind,i] = np.NaN
            beam2vel[amp_ind,i] = np.NaN
            beam4vel[amp_ind,i] = np.NaN

    # Set threshold
    corr_threshold = 50
    ## Need the .values here because xarray is funky
    beam1vel.values[beam1cor.values < corr_threshold] = np.NaN
    beam2vel.values[beam2cor.values < corr_threshold] = np.NaN
    beam3vel.values[beam3cor.values < corr_threshold] = np.NaN
    beam4vel.values[beam4cor.values < corr_threshold] = np.NaN
    
    return(beam1vel,beam2vel,beam3vel,beam4vel)









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



