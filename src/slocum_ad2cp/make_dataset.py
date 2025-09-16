import numpy as np
import math
from scipy.sparse.linalg import lsqr
import scipy
import xarray as xr
import gsw




##################################################################################################

def check_max_beam_range(beam,bins):
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.nan
    elif len(ind1) > 0:
        ind2 = np.max(ind1[:,0])
        beam_range = bins[ind2]
    return(beam_range)

##################################################################################################

def check_max_beam_range_bins(beam,bins):
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.nan
    elif len(ind1) > 0:
        beam_range = np.max(ind1[:,0])
    return(beam_range)



##################################################################################################

def check_mean_beam_range(beam,bins):
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.nan
    elif len(ind1) > 0:
        ind2 = round(np.nanmean(ind1[:,0]))
        beam_range = bins[ind2]
    return(beam_range)

##################################################################################################

def check_mean_beam_range_bins(beam,bins):
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.nan
    elif len(ind1) > 0:
        beam_range = np.nanmean(ind1[:,0])
    return(beam_range)


##################################################################################################

def beam_true_depth(ds):
    ## Create true-depth variables in master xarray dataset
    ds = ds.assign(TrueDepthBeam1=ds["VelocityBeam1"] *np.nan)
    ds = ds.assign(TrueDepthBeam2=ds["VelocityBeam2"] *np.nan)
    ds = ds.assign(TrueDepthBeam3=ds["VelocityBeam3"] *np.nan)
    ds = ds.assign(TrueDepthBeam4=ds["VelocityBeam4"] *np.nan)
    ## Now just for UVW bins
    ds = ds.assign(TrueDepth=ds["VelocityBeam1"] *np.nan)
    
    ## Preallocate variables outside of master xarray dataset for easy looping
    TrueDepthBeam1 = np.empty((len(ds.VelocityRange),len(ds.time)))
    TrueDepthBeam2 = np.empty((len(ds.VelocityRange),len(ds.time)))
    TrueDepthBeam3 = np.empty((len(ds.VelocityRange),len(ds.time)))
    TrueDepthBeam4 = np.empty((len(ds.VelocityRange),len(ds.time)))
    ## Set the empty variables = nan
    TrueDepthBeam1[:] = np.nan
    TrueDepthBeam2[:] = np.nan
    TrueDepthBeam3[:] = np.nan
    TrueDepthBeam4[:] = np.nan
    
    Pitch  = ds['Pitch'].values
    Roll   = ds['Roll'].values
    Vrange = ds.VelocityRange.values
    Depth = ds['Depth'].values
    
    
    ## Loop through each time (ping) and find the correct depth for each beam based on transducer geometry, pitch, and roll.
    for i in np.arange(0,len(ds.time)):
        TrueDepthBeam1[:,i] = cell_vert(Pitch[i], Roll[i], Vrange, beam_number=1)
        TrueDepthBeam2[:,i] = cell_vert(Pitch[i], Roll[i], Vrange, beam_number=2)
        TrueDepthBeam3[:,i] = cell_vert(Pitch[i], Roll[i], Vrange, beam_number=3)
        TrueDepthBeam4[:,i] = cell_vert(Pitch[i], Roll[i], Vrange, beam_number=4)
    
    ## True depth of
    [bdepth,bbins]=np.meshgrid(Depth,Vrange)
    true_depth =  bdepth+bbins    
    
    ## Now put the output back into the master xarray dataset
    ds['TrueDepthBeam1'].values = TrueDepthBeam1
    ds['TrueDepthBeam2'].values = TrueDepthBeam2
    ds['TrueDepthBeam3'].values = TrueDepthBeam3
    ds['TrueDepthBeam4'].values = TrueDepthBeam4
    ds['TrueDepth'].values = true_depth
    return ds

##################################################################################################

def binmap_adcp(ds):
    ## Depth bins to interp onto
    Vrange = np.array(ds.VelocityRange.values)
    
    ## Apparently xarray kind of sucks and it is faster to pull out these variables as objects, perform calculations, and stuff back in
    TrueDepthBeam1 = ds.TrueDepthBeam1.values
    TrueDepthBeam2 = ds.TrueDepthBeam2.values
    TrueDepthBeam3 = ds.TrueDepthBeam3.values
    TrueDepthBeam4 = ds.TrueDepthBeam4.values
    VelocityBeam1 = ds.VelocityBeam1.values
    VelocityBeam2 = ds.VelocityBeam2.values
    VelocityBeam3 = ds.VelocityBeam3.values    
    VelocityBeam4 = ds.VelocityBeam4.values
    
    ## Preallocate interpolated velocity outside of master xarray dataset for easy looping
    InterpVelocityBeam1 = np.empty((len(Vrange),len(ds.time)))
    InterpVelocityBeam2 = np.empty((len(Vrange),len(ds.time)))
    InterpVelocityBeam3 = np.empty((len(Vrange),len(ds.time)))
    InterpVelocityBeam4 = np.empty((len(Vrange),len(ds.time)))
    ## Set the empty variables = nan
    InterpVelocityBeam1[:] = np.nan
    InterpVelocityBeam2[:] = np.nan
    InterpVelocityBeam3[:] = np.nan
    InterpVelocityBeam4[:] = np.nan
    
    ## Create true-interp velocity variables in master xarray dataset
    ds = ds.assign(InterpVelocityBeam1=ds["VelocityBeam1"] *np.nan)
    ds = ds.assign(InterpVelocityBeam2=ds["VelocityBeam2"] *np.nan)
    ds = ds.assign(InterpVelocityBeam3=ds["VelocityBeam3"] *np.nan)
    ds = ds.assign(InterpVelocityBeam4=ds["VelocityBeam4"] *np.nan)

    ## Loop through each time (ping) and interpolate beam velocity onto the regular grid defined in the initial sensor config
    for x in np.arange(0,len(ds.time)):
        InterpVelocityBeam1[:,x] = np.interp(Vrange,TrueDepthBeam1[:,x],VelocityBeam1[:,x],right=np.nan)
        InterpVelocityBeam2[:,x] = np.interp(Vrange,TrueDepthBeam2[:,x],VelocityBeam2[:,x],right=np.nan)
        InterpVelocityBeam3[:,x] = np.interp(Vrange,TrueDepthBeam3[:,x],VelocityBeam3[:,x],right=np.nan)
        InterpVelocityBeam4[:,x] = np.interp(Vrange,TrueDepthBeam4[:,x],VelocityBeam4[:,x],right=np.nan)


    ## Now put the output back into the master xarray dataset
    ds['InterpVelocityBeam1'].values = InterpVelocityBeam1
    ds['InterpVelocityBeam2'].values = InterpVelocityBeam2
    ds['InterpVelocityBeam3'].values = InterpVelocityBeam3
    ds['InterpVelocityBeam4'].values = InterpVelocityBeam4
    
    return(ds)



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

def qaqc_pre_coord_transform(ds, corr_threshold, max_amplitude):
    ## This sucks but much faster than working through xarray
    VelocityBeam1    = ds.VelocityBeam1.values
    VelocityBeam2    = ds.VelocityBeam2.values
    VelocityBeam3    = ds.VelocityBeam3.values
    VelocityBeam4    = ds.VelocityBeam4.values
    CorrelationBeam1 = ds.CorrelationBeam1.values
    CorrelationBeam2 = ds.CorrelationBeam2.values
    CorrelationBeam3 = ds.CorrelationBeam3.values
    CorrelationBeam4 = ds.CorrelationBeam4.values
    AmplitudeBeam1   = ds.AmplitudeBeam1.values
    AmplitudeBeam2   = ds.AmplitudeBeam2.values
    AmplitudeBeam3   = ds.AmplitudeBeam3.values
    AmplitudeBeam4   = ds.AmplitudeBeam4.values

    # Filter for low correlation
    VelocityBeam1[np.where(CorrelationBeam1 < corr_threshold)] = np.nan
    VelocityBeam2[np.where(CorrelationBeam2 < corr_threshold)] = np.nan
    VelocityBeam3[np.where(CorrelationBeam3 < corr_threshold)] = np.nan
    VelocityBeam4[np.where(CorrelationBeam4 < corr_threshold)] = np.nan

    # Filter for high amplitude
    VelocityBeam1[np.where(AmplitudeBeam1 > max_amplitude)] = np.nan
    VelocityBeam2[np.where(AmplitudeBeam2 > max_amplitude)] = np.nan
    VelocityBeam3[np.where(AmplitudeBeam3 > max_amplitude)] = np.nan
    VelocityBeam4[np.where(AmplitudeBeam4 > max_amplitude)] = np.nan

    # Now stuff back into xarray ds
    ds.VelocityBeam1.values = VelocityBeam1
    ds.VelocityBeam2.values = VelocityBeam2
    ds.VelocityBeam3.values = VelocityBeam3
    ds.VelocityBeam4.values = VelocityBeam4
    return(ds)


##################################################################################################

def qaqc_post_coord_transform(ds, high_velocity_threshold, surface_depth_to_filter):
    ## This does three thingss:
    ## 1) Filters out high velocities relative to glider
    ## 2) Filters out the first bin below the glider (contaminated for vehicle motion)
    ## 3) Filters out velocity data if the GLIDER'S depth is x meters or shallower
    
    ## This sucks but much faster than working through xarray
    UVelocity    =  ds.UVelocity.values
    VVelocity    =  ds.VVelocity.values
    WVelocity    =  ds.WVelocity.values
    depth        =  ds.Depth.values
    
    ## Filter out high velocities relative to glider
    UVelocity[np.abs(UVelocity) > high_velocity_threshold] = np.nan
    VVelocity[np.abs(VVelocity) > high_velocity_threshold] = np.nan
    WVelocity[np.abs(WVelocity) > high_velocity_threshold] = np.nan
    
    ## Filter out first bin below glider
    UVelocity[0,:] = np.nan
    VVelocity[0,:] = np.nan
    WVelocity[0,:] = np.nan
    
    ## Filter out velocity if true depth is 5 meters or shallower
    depthind = np.where(depth <= surface_depth_to_filter)
    UVelocity[:,depthind] = np.nan
    VVelocity[:,depthind] = np.nan
    WVelocity[:,depthind] = np.nan
    
    ## Now stuff back into xarray ds
    ds.UVelocity.values = UVelocity
    ds.VVelocity.values = VVelocity
    ds.WVelocity.values = WVelocity
    return(ds)
    


##################################################################################################

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
    ## if there are nans still present in the data here, it will throw everything off
    ## These steps are HUGE for efficiency because it reduces the size of the G
    ## matrix as much as possible.

    ## This determines the rows (bins) where all the columns are nan
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
    bin_count[:] = np.nan

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
    #### THIS removes all nan elements of d AND Gstar so the inversion doesn't blow up with nans
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
    obs_per_bin[:] = np.nan

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


##################################################################################################

def shear_method(U,V,W,vx,vy,bins,depth,dz):
    ########################################################################  
    # These steps filter for NAN rows and columns so they are technically QAQC
    # but I think the best place to put them is inthe inversion function because
    # if there are nans still present in the data here, it will throw everything off
    # These steps are HUGE for efficiency because it reduces the size of the G
    # matrix as much as possible.

    ## This determines the rows (bins) where all the columns are nan
    nanind = np.where( (np.sum(np.isnan(U),axis=1)/U.shape[1]) == 1)[0]
    if len(nanind) > 0:
        U = np.delete(U,nanind,axis=0)
        V = np.delete(V,nanind,axis=0)
        W = np.delete(W,nanind,axis=0)
        bins = np.delete(bins,nanind)

    ## Do the same thing with individual ensembles. Note: need to remove the corresponding
    ## ensemble pressure reading to ensure correction dimensions and values.
    nanind = np.where((np.sum(np.isnan(U),axis=0)/U.shape[0]) == 1)[0]
    if len(nanind) > 0:
        U = np.delete(U,nanind,axis=1)
        V = np.delete(V,nanind,axis=1)
        W = np.delete(W,nanind,axis=1)
        depth = np.delete(depth,nanind)
    ##########################################################################        


    ##########################################################################        
    # Take difference between bin lengths for bin size [m]
    bin_size = np.diff(bins)[0]
    bin_num = len(bins)
    # This creates a grid of the ACTUAL depths of the ADCP bins by adding the
    # depths of the ADCP bins to the actual depth of the instrument


    [bdepth,bbins]=np.meshgrid(depth,bins[0:-1])
    bin_depth = bdepth+bbins  

    Z = bin_depth
    # Calculate the maximum depth of glider which is different than maximum ADCP bin depth
    ZmM = np.nanmax(depth)

    ## Calculate shear per ensemble
    ensemble_shear_U = calc_ensemble_shear(U,bins)
    ensemble_shear_V = calc_ensemble_shear(V,bins)
    ensemble_shear_W = calc_ensemble_shear(W,bins)

    ## Create velocity dataframes for shear
    flatu = ensemble_shear_U.flatten(order='F')
    flatv = ensemble_shear_V.flatten(order='F')
    flatw = ensemble_shear_W.flatten(order='F')
    flatz = -Z.flatten(order='F')
    flat_df = np.column_stack((flatu,flatv,flatw,flatz))
    flat_df = flat_df[flat_df[:, 3].argsort()[::-1]]

    ## Shear binning
    shear_binned, shear_binned_std, shear_cell_center, vels_in_bin = bin_attr(shear_v= flat_df[:,0:3], shear_z=flat_df[:,3], bin_size=dz, Hmax=np.nanmin(flat_df[:,3]))
    ## Shear to absolute
    vel, vel_referenced, bin_centers = shear_to_vel(shear_binned, shear_cell_center, ref_vel=[vx,vy,0])
    
    ## First define uncertainty of a single ping according to the ADCP configuration
    std_ping = 0.03     #[m/s] (in average mode)
    vel_referenced_std = np.sqrt(std_ping**2 + shear_binned_std**2)
    
    return(vel_referenced, bin_centers, vel_referenced_std)


##################################################################################################

def mag_var_correction(heading,u_dac,v_dac,mag_var):
    heading_corrected = heading - mag_var ## Corrected heading in degrees
    mag_var_rad = np.deg2rad(mag_var)
    heading_rad = np.deg2rad(heading)
    u_dac_corrected = u_dac*np.cos(mag_var_rad) - v_dac*np.sin(mag_var_rad)
    v_dac_corrected = u_dac*np.sin(mag_var_rad) + v_dac*np.cos(mag_var_rad)
    
    return heading_corrected, u_dac_corrected, v_dac_corrected





def mag_var_correction_ad2cp_ds(ds, heading_var="CorrectedHeading", mag_var_arr=0):
    """
    Apply magnetic variation correction to AD2CP heading directly in the dataset.
    Drops any reference to u_dac and v_dac (not used here).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing heading and magnetic variation.
    heading_var : str
        Name of heading variable to correct.
    mag_var : str
        Name of magnetic variation variable (degrees).

    Returns
    -------
    ds : xarray.Dataset
        Dataset with corrected heading assigned to 'CorrectedHeading_MagVar'.
    """
    # Extract arrays
    heading = ds[heading_var].values

    # Correct heading
    heading_corrected = heading - mag_var_arr

    # Assign corrected heading to new variable
    ds = ds.assign({"CorrectedHeading_MagVar": ("time", heading_corrected)})

    return ds



##################################################################################################

def calcAHRS(ds, heading_var="CorrectedHeading_MagVar", roll_var="Roll", pitch_var="Pitch"):
    """
    Compute AHRS rotation matrix for each time step and attach it to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain variables for heading, roll, and pitch.
    heading_var : str
        Name of heading variable in ds.
    roll_var : str
        Name of roll variable in ds.
    pitch_var : str
        Name of pitch variable in ds.

    Returns
    -------
    ds_out : xarray.Dataset
        Original dataset with added variable 'AHRSRotationMatrix' of shape (9, time)
    """
    headingVal = np.array(ds[heading_var])
    rollVal = np.array(ds[roll_var])
    pitchVal = np.array(ds[pitch_var])

    RotMatrix = np.full((9, len(pitchVal)), np.nan)

    for k in range(len(pitchVal)):
        hh = np.deg2rad(headingVal[k] - 90)
        pp = np.deg2rad(pitchVal[k])
        rr = np.deg2rad(rollVal[k])

        # Heading matrix
        H = np.array([
            [np.cos(hh), np.sin(hh), 0],
            [-np.sin(hh), np.cos(hh), 0],
            [0, 0, 1]
        ])

        # Tilt matrix
        P = np.array([
            [np.cos(pp), -np.sin(pp)*np.sin(rr), -np.cos(rr)*np.sin(pp)],
            [0, np.cos(rr), -np.sin(rr)],
            [np.sin(pp), np.sin(rr)*np.cos(pp), np.cos(pp)*np.cos(rr)]
        ])

        # Combined rotation matrix
        R = H @ P
        RotMatrix[:, k] = R.reshape(-1, order='F')



    # Attach to dataset
    ds_out = ds.copy()
    ds_out = ds_out.assign(AHRSRotationMatrix=(("x", "time"), RotMatrix))

    return ds_out



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
    if 'burst_beam2xyz' in ds.attrs:
        beam2xyz = ds.attrs['burst_beam2xyz']
    elif 'beam2xyz' in ds.attrs:
        beam2xyz = ds.attrs['beam2xyz']
    elif 'avg_beam2xyz' in ds.attrs:
        beam2xyz = ds.attrs['avg_beam2xyz']
    else:
        print('No beam transformation matrix info found')

    beam2xyz = beam2xyz.reshape(4,4)  # Because we know this configuration is a 4 beam AD2CP

    ds = ds.assign(UVelocity=ds["InterpVelocityBeam1"] *np.nan)
    ds = ds.assign(VVelocity=ds["InterpVelocityBeam1"] *np.nan)
    ds = ds.assign(WVelocity=ds["InterpVelocityBeam1"] *np.nan)

    InterpVelocityBeam1 = ds.InterpVelocityBeam1.values
    InterpVelocityBeam2 = ds.InterpVelocityBeam2.values
    InterpVelocityBeam3 = ds.InterpVelocityBeam3.values
    InterpVelocityBeam4 = ds.InterpVelocityBeam4.values

    ## Preallocate interpolated velocity outside of master xarray dataset for easy looping
    UVelocity = np.empty((len(ds.VelocityRange),len(ds.time)))
    VVelocity = np.empty((len(ds.VelocityRange),len(ds.time)))
    WVelocity = np.empty((len(ds.VelocityRange),len(ds.time)))

    ## Set the empty variables = nan
    UVelocity[:] = np.nan
    VVelocity[:] = np.nan
    WVelocity[:] = np.nan

    ## Pull these out of xarray out of loop
    AHRSRotationMatrix = ds.AHRSRotationMatrix.values

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
        # beam2xyz_mat[1,:] = -beam2xyz_mat[1,:]
        # beam2xyz_mat[2,:] = -beam2xyz_mat[2,:]
        beam2xyz_mat = beam2xyz[0:3, 1:4].copy()
        # then apply sign correction
        beam2xyz_mat[1,:] *= -1
        beam2xyz_mat[2,:] *= -1

        ## Now convert to XYZ        
        xyz = np.dot(beam2xyz_mat,tot_vel.T)

        ## Grab AHRS rotation matrix for this ping
        xyz2enuAHRS = AHRSRotationMatrix[:,x].reshape(3,3, order='C')

        ## Now convert XYZ velocities to ENU, where enu[0,:] is U, enu[1,:] is V, and enu[2,:] is W velocities.
        enu = np.array(np.dot(xyz2enuAHRS,xyz))
        UVelocity[:,x] = enu[0,:].ravel()
        VVelocity[:,x] = enu[1,:].ravel()
        WVelocity[:,x] = enu[2,:].ravel()

    ds['UVelocity'].values = UVelocity
    ds['VVelocity'].values = VVelocity
    ds['WVelocity'].values = WVelocity

    return(ds)



##################################################################################################

def load_ad2cp(ncfile, mean_lat=45):
    """
    Load Nortek AD2CP data from one or more NetCDF files.
    
    Tries 'Data/Average/' first, then 'Data/Burst/' if no data in Average.
    Converts Pressure to Depth and drops Pressure.
    
    Parameters
    ----------
    ncfile : str or list of str
        Path to a single NetCDF file or list of files.
    mean_lat : float
        Latitude for converting Pressure to Depth.
    
    Returns
    -------
    ds : xarray.Dataset
        Combined dataset with Depth variable.
    group : str
        Group that was loaded ('Average' or 'Burst').
    """
    # Normalize input into list
    if isinstance(ncfile, str):
        files = [ncfile]
    elif isinstance(ncfile, (list, tuple, np.ndarray)):
        files = list(ncfile)
    else:
        raise TypeError("ncfile must be a string or list of strings")

    group = None
    ds = None

    # --- Try Average group ---
    try:
        if len(files) == 1:
            ds = xr.open_dataset(files[0], group="Data/Average/", engine="netcdf4")
        else:
            ds = xr.open_mfdataset(
                files,
                group="Data/Average/",
                concat_dim="time",
                combine="nested",
                engine="netcdf4"
            )
        if ds.time.size > 0:
            group = "Average"
    except Exception:
        pass

    # --- Fallback to Burst group ---
    if group is None:
        try:
            if len(files) == 1:
                ds = xr.open_dataset(files[0], group="Data/Burst/", engine="netcdf4")
            else:
                ds = xr.open_mfdataset(
                    files,
                    group="Data/Burst/",
                    concat_dim="time",
                    combine="nested",
                    engine="netcdf4"
                )
            if ds.time.size > 0:
                group = "Burst"
        except Exception:
            raise ValueError("Neither 'Average' nor 'Burst' groups contain data")

    # Sort by time
    ds = ds.sortby("time")

    # Attach attributes from Config group of the FIRST file
    config = xr.open_dataset(files[0], group="Config", engine="netcdf4")
    ds = ds.assign_attrs(config.attrs)

    # Rename variables for consistency
    rename_map = {
        "Velocity Range": "VelocityRange",
        "Correlation Range": "CorrelationRange",
        "Amplitude Range": "AmplitudeRange"
    }
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.variables})

    # Convert Pressure -> Depth
    if "Pressure" in ds.variables:
        ds = ds.assign(Depth=("time", -gsw.z_from_p(ds.Pressure.values, mean_lat)))
        ds = ds.drop_vars("Pressure")

    # Reorder dimensions consistently
    ds = ds.transpose()

    return ds




##################################################################################################

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
	
##################################################################################################
	
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
	
##################################################################################################

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



##################################################################################################

def correct_ad2cp_heading(ds):
    """
    Correct AD2CP heading using magnetometer and orientation data,
    and return a new xarray.Dataset with corrected values.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain variables: 'Heading', 'Pitch', 'Roll', 'Pressure',
        'MagnetometerX', 'MagnetometerY', 'MagnetometerZ', 'time'

    Returns
    -------
    ds_corrected : xarray.Dataset
        Original dataset with added variables:
        - 'CorrectedHeading'
        - 'MagX_corrected'
        - 'MagY_corrected'
        Also retains original magnetometer variables.
    """
    head = np.array(ds['Heading'])
    pitch = np.array(ds['Pitch'])
    roll = np.array(ds['Roll'])
    x = np.array(ds['MagnetometerX'])
    y = np.array(ds['MagnetometerY'])
    z = np.array(ds['MagnetometerZ'])

    xyz_original = np.column_stack((x, y, z))

    pitch_ranges = np.arange(-35, 35, 1)
    for k in range(len(pitch_ranges) - 1):
        mask = (pitch > pitch_ranges[k]) & (pitch < pitch_ranges[k + 1])
        indices = np.where(mask)
        if len(indices[0]) > 9:
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

    CorrectedHeading = np.empty_like(head) * np.nan
    for k in range(len(head)):
        CorrectedHeading[k] = calc_heading(xyz_final[k, :], pitch[k], roll[k], 1)

    # Create a new dataset with corrected variables
    ds_corrected = ds.copy()
    ds_corrected = ds_corrected.assign(
        CorrectedHeading=("time", CorrectedHeading),
        MagX_corrected=("time", x),
        MagY_corrected=("time", y),
    )

    return ds_corrected






__all__ = [
    "check_max_beam_range",
    "check_max_beam_range_bins",
    "check_mean_beam_range",
    "check_mean_beam_range_bins",
    "beam2enu",
    "beam_true_depth",
    "binmap_adcp",
    "cell_vert",
    "correct_sound_speed",
    "qaqc_pre_coord_transform",
    "qaqc_post_coord_transform",
    "inversion",
    "mag_var_correction",
    "shear_method",
    "calcAHRS",
    "load_ad2cp",
    "correct_ad2cp_heading",
    "mag_var_correction_ad2cp_ds"
]
