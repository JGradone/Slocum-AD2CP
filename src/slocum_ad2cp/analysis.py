from erddapy import ERDDAP
import numpy as np
import pandas as pd
import gsw
from math import sin, cos, sqrt, atan2, radians



## Calculate distance in meters from 2 lat and lon points
def dist_from_lat_lon(lat1,lon1,lat2,lon2):

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c *1000 # in meters

    return(distance)




# Calculate rho from salinity, temperature, pressure, lon, lat
def gsw_rho(SP, T, P, lon, lat):
        # Calculates absolute salinity (g/kg) from PSU
        SA = gsw.SA_from_SP(SP, P, lon, lat)
        # in-situ density
        rho = gsw.density.rho_t_exact(SA, T, P)
        return rho

    
    

def profile_mld(mld_var='density', zvar='depth', qi_threshold=0.5):
    """
    Written by Sam Coakley and Lori Garzio, Jan 2022
    Calculates the Mixed Layer Depth (MLD) for a single profile as the depth of max Brunt‐Vaisala frequency squared
    (N**2) from Carvalho et al 2016
    :param mld_var: the name of the variable for which MLD is calculated, default is 'density'
    :param zvar: the name of the depth variable in the dataframe, default is 'pressure'
    :param qi_threshold: quality index threshold for determining well-mixed water, default is 0.5
    :return: the depth of the mixed layer in the units of zvar
    """
    pN2 = np.sqrt(9.81 / np.nanmean(mld_var) * np.diff(mld_var) / np.diff(zvar)) ** 2
    if np.sum(~np.isnan(pN2)) == 0:
        mld = np.nan
    else:
        mld_idx = np.where(pN2 == np.nanmax(pN2))[0][0]
        mld = np.nanmean([zvar[mld_idx], zvar[mld_idx + 1]])

        if mld_idx == 0:
            # if the code finds the first data point as the MLD, return nan
            mld = np.nan
        elif mld < 2:
            # if MLD is <2, return nan
            mld = np.nan
        else:
            if qi_threshold:
                # find MLD  1.5
                mld15 = mld * 1.5
                mld15_idx = np.argmin(np.abs(zvar - mld15))

                # Calculate Quality index (QI) from Lorbacher et al, 2006
                surface_mld_values = mld_var[0:mld_idx]  # values from the surface to MLD
                surface_mld15_values = mld_var[0:mld15_idx]  # values from the surface to MLD * 1.5

                qi = 1 - (np.std(surface_mld_values - np.nanmean(surface_mld_values)) /
                          np.std(surface_mld15_values - np.nanmean(surface_mld15_values)))

                if qi < qi_threshold:
                    # if the Quality Index is < the threshold, this indicates well-mixed water so don't return MLD
                    mld = np.nan

    return mld


def grid_glider_data(df, varname, delta_z=.3):
    """
    Written by aristizabal. Returns a gridded glider dataset by depth and time
    """
    df.dropna(inplace=True)
    #df.dropna() # Changed to work with ru29 2020 datatset by JG
    df.drop(df[df['depth'] < .1].index, inplace=True)  # drop rows where depth is <1
    df.drop(df[df[varname] == 0].index, inplace=True)  # drop rows where the variable equals zero
    df.sort_values(by=['time', 'depth'], inplace=True)

    # find unique times and coordinates
    timeg, ind = np.unique(df.time.values, return_index=True)
    latg = df['latitude'].values[ind]
    long = df['longitude'].values[ind]
    dg = df['depth'].values
    vg = df[varname].values
    zn = np.int(np.max(np.diff(np.hstack([ind, len(dg)]))))

    depthg = np.empty((zn, len(timeg)))
    depthg[:] = np.nan
    varg = np.empty((zn, len(timeg)))
    varg[:] = np.nan

    for i, ii in enumerate(ind):
        if i < len(timeg) - 1:
            i_f = ind[i + 1]
        else:
            i_f = len(dg)
        depthi = dg[ind[i]:i_f]
        vari = vg[ind[i]:i_f]
        depthg[0:len(dg[ind[i]:i_f]), i] = depthi
        varg[0:len(vg[ind[i]:i_f]), i] = vari

    # sort time variable
    okt = np.argsort(timeg)
    timegg = timeg[okt]
    depthgg = depthg[:, okt]
    vargg = varg[:, okt]

    # Grid variables
    depthg_gridded = np.arange(0, np.nanmax(depthgg), delta_z)
    varg_gridded = np.empty((len(depthg_gridded), len(timegg)))
    varg_gridded[:] = np.nan

    for t, tt in enumerate(timegg):
        depthu, oku = np.unique(depthgg[:, t], return_index=True)
        varu = vargg[oku, t]
        okdd = np.isfinite(depthu)
        depthf = depthu[okdd]
        varf = varu[okdd]
        ok = np.asarray(np.isfinite(varf))
        if np.sum(ok) < 3:
            varg_gridded[:, t] = np.nan
        else:
            okd = np.logical_and(depthg_gridded >= np.min(depthf[ok]), depthg_gridded < np.max(depthf[ok]))
            varg_gridded[okd, t] = np.interp(depthg_gridded[okd], depthf[ok], varf[ok])

    return timegg, long, latg, depthg_gridded, varg_gridded



def get_erddap_dataset(ds_id, server, variables=None, constraints=None, filetype=None):
    ## Written by Mike Smith
    """
    Returns a netcdf dataset for a specified dataset ID (or dataframe if dataset cannot be converted to xarray)
    :param ds_id: dataset ID e.g. ng314-20200806T2040
    :param variables: optional list of variables
    :param constraints: optional list of constraints
    :param filetype: optional filetype to return, 'nc' (default) or 'dataframe'
    :return: netcdf dataset
    """
    variables = variables or None
    constraints = constraints or None
    filetype = filetype or 'nc'
    #ioos_url = 'https://data.ioos.us/gliders/erddap'


    e = ERDDAP(server,
               protocol='tabledap',
               response='nc')
    e.dataset_id = ds_id
    if constraints:
        e.constraints = constraints
    if variables:
        e.variables = variables
    if filetype == 'nc':
        try:
            ds = e.to_xarray()
            ds = ds.sortby(ds.time)
        except OSError:
            print('No dataset available for specified constraints: {}'.format(ds_id))
            ds = []
        except TypeError:
            print('Cannot convert to xarray, providing dataframe: {}'.format(ds_id))
            ds = e.to_pandas().dropna()
    elif filetype == 'dataframe':
        #ds = e.to_pandas().dropna()
        ds = e.to_pandas().dropna(how='all')
    else:
        print('Unrecognized filetype: {}. Needs to  be "nc" or "dataframe"'.format(filetype))

    return ds









__all__ = [
    "get_erddap_dataset",
    "grid_glider_data",
    "profile_mld",
    "gsw_rho",
    "dist_from_lat_lon",
]