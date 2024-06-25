"""
Script to process a month of the Extracted Radar Columns and In-Situ Sensors (RadCLss)

Written: Joe O'Brien <obrienj@anl.gov> - 13 June 2024
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import glob
import time
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

from dask.distributed import Client, LocalCluster

import pyart
import act

#-----------------
# Define Functions
#-----------------

def subset_points(file, **kwargs):
    """
    Subset a radar file for a set of latitudes and longitudes
    utilizing Py-ART's column-vertical-profile functionality.

    Parameters
    ----------
    file : str
        Path to the radar file to extract columns from
    nsonde : list
        List containing file paths to the desired sonde file to merge

    Calls
    -----
    radar_start_time
    merge_sonde

    Returns
    -------
    ds : xarray DataSet
        Xarray Dataset containing the radar column above a give set of locations
    
    """

    ds = None
    
    # Define the splash locations [lon,lat]
    kettle_ponds = [-106.9731488, 38.9415427]
    avery_point = [-106.9965928, 38.9705885]
    pumphouse_site = [-106.9502476, 38.9226741]
    M1 = [-106.987, 38.956158]
    snodgrass = [-106.978929, 38.926572]

    sites = ["M1", "kettle_ponds", "avery_point", "pumphouse_site", "snodgrass"]

    # Zip these together!
    lons, lats = list(zip(M1,
                          kettle_ponds,
                          avery_point,
                          pumphouse_site,
                          snodgrass))
    try:
        # Read in the file
        radar = pyart.io.read(file)
    except:
        radar = None

    if radar:
        # Easier to map the nearest sonde file to radar gates before extraction
        if 'sonde' in kwargs:

            # variables to discard when reading in the sonde file
            exclude_sonde = ['base_time', 'time_offset', 'lat', 'lon', 'qc_pres',
                             'qc_tdry', 'qc_dp', 'qc_wspd', 'qc_deg', 'qc_rh',
                             'qc_u_wind', 'qc_v_wind', 'qc_asc']
        
            # find the nearest sonde file to the radar start time
            radar_start = datetime.datetime.strptime(file.split('/')[-1].split('.')[-2], '%Y%m%d-%H%M%S')
            sonde_start = [datetime.datetime.strptime(xfile.split('/')[-1].split('.')[2] + 
                                                      '-' + 
                                                      xfile.split('/')[-1].split('.')[3], 
                                                      '%Y%m%d-%H%M%S') for xfile in kwargs['sonde']
                          ]
            # difference in time between radar file and each sonde file
            start_diff = [radar_start - sonde for sonde in sonde_start]

            # merge the sonde file into the radar object
            ds_sonde = act.io.read_arm_netcdf(nsonde[start_diff.index(min(start_diff))], 
                                              cleanup_qc=True, 
                                              drop_variables=exclude_sonde)
   
            # create list of variables within sonde dataset to add to the radar file
            for var in list(ds_sonde.keys()):
                if var != "alt":
                    z_dict, sonde_dict = pyart.retrieve.map_profile_to_gates(ds_sonde.variables[var],
                                                                             ds_sonde.variables['alt'],
                                                                             radar)
                # add the field to the radar file
                radar.add_field_like('DBZ', "sonde_" + var,  sonde_dict['data'], replace_existing=True)
                radar.fields["sonde_" + var]["units"] = sonde_dict["units"]
                radar.fields["sonde_" + var]["long_name"] = sonde_dict["long_name"]
                radar.fields["sonde_" + var]["standard_name"] = sonde_dict["standard_name"]
                radar.fields["sonde_" + var]["datastream"] = ds_sonde.datastream

            del radar_start, sonde_start, ds_sonde
            del z_dict, sonde_dict
        
        column_list = []
        for lat, lon in zip(lats, lons):
            # Make sure we are interpolating from the radar's location above sea level
            # NOTE: interpolating throughout Troposphere to match sonde to in the future
            #da = pyart.util.columnsect.get_field_location(radar, lat, lon).interp(height=np.arange(np.round(radar.altitude['data'][0]), 10100, 100))
            #da = pyart.util.columnsect.column_vertical_profile(radar, lat, lon).interp(height=np.arange(np.round(radar.altitude['data'][0]), 10100, 100))
            da = pyart.util.columnsect.column_vertical_profile(radar, lat, lon).interp(height=np.arange(3150, 10050, 50))
            # Add the latitude and longitude of the extracted column
            da["latitude"], da["longitude"] = lat, lon
            # Time is based off the start of the radar volume
            dt = pd.to_datetime(radar.time["data"], unit='s')[-1]
            da["time"] = [dt]
            column_list.append(da)
        
        # Concatenate the extracted radar columns for this scan across all sites    
        ds = xr.concat(column_list, dim='site')
        ds["site"] = sites
        # Add attributes for Time, Latitude, Longitude, and Sites
        ds.time.attrs.update(long_name=('Time in Seconds that Cooresponds to the Start'
                                        + " of each Individual Radar Volume Scan before"
                                        + " Concatenation"),
                             description=('Time in Seconds that Cooresponds to the Minimum'
                                          + ' Height Gate'))
        ds.site.attrs.update(long_name="SAIL/SPLASH In-Situ Ground Observation Site Identifers")
        ds.latitude.attrs.update(long_name='Latitude of SAIL Ground Observation Site',
                                 units='Degrees North')
        ds.longitude.attrs.update(long_name='Longitude of SAIL Ground Observation Site',
                                 units='Degrees East')
        # delete the radar to free up memory
        del radar, column_list, da
    return ds

def match_datasets_act(column, ground, site, discard, resample='sum', DataSet=False):
    """
    Time synchronization of a Ground Instrumentation Dataset to 
    a Radar Column for Specific Locations using the ARM ACT package
    
    Parameters
    ----------
    column : Xarray DataSet
        Xarray DataSet containing the extracted radar column above multiple locations.
        Dimensions should include Time, Height, Site
             
    ground : str; Xarray DataSet
        String containing the path of the ground instrumentation file that is desired
        to be included within the extracted radar column dataset. 
        If DataSet is set to True, ground is Xarray Dataset and will skip I/O. 
             
    site : str
        Location of the ground instrument. Should be included within the filename. 
        
    discard : list
        List containing the desired input ground instrumentation variables to be 
        removed from the xarray DataSet. 
    
    resample : str
        Mathematical operational for resampling ground instrumentation to the radar time.
        Default is to sum the data across the resampling period. Checks for 'mean' or 
        to 'skip' altogether. 
    
    DataSet : boolean
        Boolean flag to determine if ground input is an Xarray Dataset.
        Set to True if ground input is Xarray DataSet. 
             
    Returns
    -------
    ds : Xarray DataSet
        Xarray Dataset containing the time-synced in-situ ground observations with
        the inputed radar column 
    """
    # Check to see if input is xarray DataSet or a file path
    if DataSet == True:
        grd_ds = ground
    else:
        # Read in the file using ACT
        grd_ds = act.io.read_arm_netcdf(ground, cleanup_qc=True, drop_variables=discard)
        # Default are Lazy Arrays; convert for matching with column
        grd_ds = grd_ds.compute()
        # Check to see if file is the RWP, 
        if 'rwp' in ground[0].split('/')[-1]:
            # adjust the RWP heights above ground level
            grd_ds['height'] = grd_ds.height.data + grd_ds.alt.data
        if 'ceil' in ground[0].split('/')[-1]:
            # correct ceilometer backscatter 
            grd_ds = act.corrections.correct_ceil(grd_ds, var_name='backscatter')
            # Rename the range dimension and apply altitude 
            grd_ds = grd_ds.rename({'range' : 'height'})
            grd_ds['height'] = grd_ds.height.data + grd_ds.alt.data
        
    # Remove Base_Time before Resampling Data since you can't force 1 datapoint to 5 min sum
    if 'base_time' in grd_ds.data_vars:
        del grd_ds['base_time']
        
    # Check to see if height is a dimension within the ground instrumentation. 
    # If so, first interpolate heights to match radar, before interpolating time.
    if 'height' in grd_ds.dims:
        grd_ds = grd_ds.interp(height=np.arange(3150, 10050, 50), method='linear')
        
    # Resample the ground data to 5 min and interpolate to the CSU X-Band time. 
    # Keep data variable attributes to help distingish between instruments/locations
    if resample.split('=')[-1] == 'mean':
        matched = grd_ds.resample(time='5Min', 
                                  closed='right').mean(keep_attrs=True).interp(time=column.time, 
                                                                               method='linear')
    elif resample.split('=')[-1] == 'skip':
        matched = grd_ds.interp(time=column.time, method='linear')
    else:
        matched = grd_ds.resample(time='5Min', 
                                  closed='right').sum(keep_attrs=True).interp(time=column.time, 
                                                                              method='linear')
    
    # Add SAIL site location as a dimension for the Pluvio data
    matched = matched.assign_coords(coords=dict(site=site))
    matched = matched.expand_dims('site')
   
    # Remove Lat/Lon Data variables as it is included within the Matched Dataset with Site Identfiers
    if 'lat' in matched.data_vars:
        del matched['lat']
    if 'lon' in matched.data_vars:
        del matched['lon']
    if 'alt' in matched.data_vars:
        del matched['alt']
        
    # Update the individual Variables to Hold Global Attributes
    # global attributes will be lost on merging into the matched dataset.
    # Need to keep as many references and descriptors as possible
    for var in matched.data_vars:
        matched[var].attrs.update(source=matched.datastream)
        
    # Merge the two DataSets
    column = xr.merge([column, matched])
   
    return column


def main(args):
    print("process start time: ", time.strftime("%H:%M:%S"))
    # Define directories
    ndate = args.date
     # Define the directory where the CSU-X Band CMAC2.0 files are located.
    RADAR_DIR = '/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradarcmacS2.c1/ppi/%s/*.nc' % ndate
    out_path = '/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradclssS2.c2/%s/' % ndate

    # Define an output directory for downloaded ground instrumentation
    PLUVIO_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/gucwbpluvio2M1.a1/*%s*.nc' % ndate
    MET_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/gucmetM1.b1/*%s*.nc' % ndate
    LD_M1_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/gucldM1.b1/*%s*.nc' % ndate
    LD_S2_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/gucldS2.b1/*%s*.nc' % ndate
    SONDE_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/gucsondewnpnM1.b1/*%s*.nc' % ndate
    RWP_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/guc915rwpprecipmeanlowM1.a1/*%s*.nc' % ndate
    CEIL_DIR = "/gpfs/wolf2/arm/atm124/proj-shared/gucceilM1.b1/*%s*.nc" % ndate

    # define the number of days within the month
    d0 = datetime.datetime(year=int(ndate[0:4]), month=int(ndate[4:7]), day=1)
    d1 = datetime.datetime(year=int(ndate[0:4]), month=int(ndate[4:7])+1, day=1)
    volumes = {'date': [], 'radar' : [], 'pluvio' : [], 'met' : [], 'ld_m1' : [], 
               'ld_s2' : [], 'sonde' : [], 'rwp' : [], 'ceil' : []}
    # iterate through files and collect together
    for i in range((d1-d0).days):
        if i < 9:
            #volumes['radar'].append(sorted(glob.glob(RADAR_DIR + 'gucxprecipradarcmacM1.c1.' + ndate + '0' + str(i+1) + '*.nc')))
            volumes['pluvio'].append(sorted(glob.glob(PLUVIO_DIR + '*gucwbpluvio2M1.a1.' + DATE.replace('-','') + '*')))
        else:
            #volumes['radar'].append(sorted(glob.glob(RADAR_DIR + 'gucxprecipradarcmacM1.c1.' + ndate + str(i+1) + '*.nc')))
  
    if args.serial is True:
        granule(volumes[0])
        granule(volumes[1])
        granule(volumes[2])
        print("processing finished: ", time.strftime("%H:%M:%S"))
    else:
        print("starting dask cluster...")
        cluster = LocalCluster(n_workers=32,  threads_per_worker=1)
        print(cluster)
        with Client(cluster) as c:
            results = c.map(granule, volumes)
            wait(results)
        print("processing finished: ", time.strftime("%H:%M:%S"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Matched Radar Columns and In-Situ Sensors (RadCLss) Processing." +
            "Extracts Radar columns above a given site and collocates with in-situ sensors")

    parser.add_argument("--date",
                        default="202203",
                        dest='date',
                        type=str,
                        help="Month to process in YYYYMM format"
    )
    parser.add_argument("--serial",
                        default=False,
                        dest='serial',
                        type=bool,
                        help="Process in Serial for testing"
    )
    args = parser.parse_args()

    main(args)