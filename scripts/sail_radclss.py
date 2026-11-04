import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import glob
import time
import datetime
import argparse
import logging
import dask

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

from dask.distributed import Client, LocalCluster, wait, as_completed, fire_and_forget
dask.config.set({'logging.distributed': 'error'})

from matplotlib.dates import DateFormatter
from matplotlib import colors

import pyart
import act

#-----------------
# Define Constants
#-----------------

# Height gates the radar columns and the height-resolved in-situ sensors
# are interpolated onto. Defined once so the column extraction and the
# ground-instrument matching can never drift apart.
RADCLSS_HEIGHTS = np.arange(3150, 10050, 50)

# Datastream name. Used for both the `datastream` global attribute and the
# output filename, so the two can never disagree with each other or with the
# names the shipped files already carry.
DATASTREAM = "gucxprecipradarradclssS2.c2"

# Missing value code declared by the RadCLss DOD (xprecipradarradclss-c2).
# Every variable is written with this as both `_FillValue` and `missing_value`
# so the declared value always matches what is actually stored in the file.
MISSING_VALUE = -999999.0

# In-situ variables that are a true per-interval accumulation, and so must be
# summed when resampled onto the 5 minute radar volume cadence. Every other
# variable is a state or a rate and must be averaged -- summing those inflates
# them by the number of input samples per window (5x for the 1 minute MET,
# Pluvio and laser disdrometer streams, ~19x for the 16 second ceilometer).
SUM_VARS = frozenset({
    "accum_nrt",
    "accum_rtnrt",
    "intensity_rt",
    "intensity_rtnrt",
    "tbrg_precip_total",
    "tbrg_precip_total_corr",
})

# Attributes for the coordinate variables. The DOD holds only placeholder
# entries for these, so they are set here to match the shipped files.
COORD_ATTRS = {
    'time': {'long_name': 'Time offset from midnight',
             'standard_name': 'time'},
    'height': {'long_name': 'Height of the measurement volume',
               'units': 'm',
               'standard_name': 'height'},
    'station': {'long_name': 'Location of surface in-situ observations'},
    'particle_size': {'long_name': 'Particle class size average',
                      'units': 'mm'},
    'raw_fall_velocity': {'long_name': 'Fall velocity classes observed by Parsivel2',
                          'units': 'm/s'},
}

#-----------------
# Define Functions
#-----------------

def subset_points(nfile, **kwargs):
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
    S2 = [-106.943114, 38.897961]

    sites = ["M1", "kettle_ponds", "avery_point", "pumphouse_site", "snodgrass", "S2"]

    # Zip these together!
    lons, lats = list(zip(M1,
                          kettle_ponds,
                          avery_point,
                          pumphouse_site,
                          snodgrass,
                          S2))
    try:
        # Read in the file
        radar = pyart.io.read(nfile)
        if isinstance(radar.range["meters_between_gates"], str):
            radar.range["meters_between_gates"] = float(radar.range["meters_between_gates"])
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
            radar_start = datetime.datetime.strptime(nfile.split('/')[-1].split('.')[-3] + '.' + nfile.split('/')[-1].split('.')[-2], 
                                                     '%Y%m%d.%H%M%S'
            )
            sonde_start = [datetime.datetime.strptime(xfile.split('/')[-1].split('.')[2] + 
                                                      '-' + 
                                                      xfile.split('/')[-1].split('.')[3], 
                                                      '%Y%m%d-%H%M%S') for xfile in kwargs['sonde']
                          ]
            # difference in time between radar file and each sonde file
            start_diff = [radar_start - sonde for sonde in sonde_start]

            # merge the sonde file into the radar object
            ds_sonde = act.io.read_arm_netcdf(kwargs['sonde'][start_diff.index(min(start_diff))], 
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
        
        # Time is based off the start of the radar volume.
        # Guard against volumes whose stored times cannot be interpreted as
        # milliseconds since the epoch (a handful of glue files carry seconds
        # instead, which decodes to Jan 1970 and drags the file timeline with it).
        dt = volume_time(nfile, radar)
        if dt is None:
            del radar
            return None

        column_list = []
        for lat, lon in zip(lats, lons):
            # Make sure we are interpolating from the radar's location above sea level
            # NOTE: interpolating throughout Troposphere to match sonde to in the future
            #da = pyart.util.columnsect.get_field_location(radar, lat, lon).interp(height=np.arange(np.round(radar.altitude['data'][0]), 10100, 100))
            #da = pyart.util.columnsect.column_vertical_profile(radar, lat, lon).interp(height=np.arange(np.round(radar.altitude['data'][0]), 10100, 100))
            try:
                da = (pyart.util.columnsect.column_vertical_profile(radar, lat, lon)
                      .interp(height=RADCLSS_HEIGHTS))
            except ValueError:
                # The column can contain non-finite or unsorted height gates, which
                # `interp` refuses. Interpolate across whatever gates are valid rather
                # than falling back on a constant-filled profile.
                da = pyart.util.columnsect.column_vertical_profile(radar, lat, lon)
                valid = np.isfinite(da["height"])
                if int(valid.sum()) > 0:
                    da = (da.sel(height=valid)
                            .sortby("height")
                            .interp(height=RADCLSS_HEIGHTS))
                else:
                    target_height = xr.DataArray(RADCLSS_HEIGHTS, dims="height", name="height")
                    da = da.reindex(height=target_height)

            # Add the latitude and longitude of the extracted column
            da["lat"], da["lon"] = lat, lon
            da["time"] = [dt]
            column_list.append(da)

        # Concatenate the extracted radar columns for this scan across all sites
        ds = xr.concat(column_list, dim='station')
        ds["station"] = sites
        # Add attributes for Time, Latitude, Longitude, and Sites
        ds.time.attrs.update(long_name=('Time in Seconds that Cooresponds to the Start'
                                        + " of each Individual Radar Volume Scan before"
                                        + " Concatenation"),
                             description=('Time in Seconds that Cooresponds to the Minimum'
                                          + ' Height Gate'))
        ds.station.attrs.update(long_name="SAIL/SPLASH In-Situ Ground Observation Station Identifers")
        ds.lat.attrs.update(long_name='Latitude of SAIL Ground Observation Site',
                            units='Degrees North')
        ds.lon.attrs.update(long_name='Longitude of SAIL Ground Observation Site',
                            units='Degrees East')
        # delete the radar to free up memory
        del radar, column_list, da
    return ds

def volume_time(nfile, radar):
    """
    Determine the start time of a radar volume, cross-checked against the
    timestamp in the filename.

    The glued .b1 volumes store `time` as milliseconds since the epoch, but a
    small number of them were written with seconds instead. Decoding those as
    milliseconds places the volume in January 1970, which then becomes the
    `units` epoch of the daily RadCLss file and makes the whole file timeline
    span 1970 to the present.

    Parameters
    ----------
    nfile : str
        Path to the radar file. The filename supplies the expected date in
        `.YYYYmmdd.HHMMSS.` form.
    radar : pyart.core.Radar
        Radar object the volume time is read from.

    Returns
    -------
    dt : pandas.Timestamp or None
        Start time of the volume, or None if no sane time could be recovered.
    """
    try:
        expected = datetime.datetime.strptime(
            nfile.split('/')[-1].split('.')[-3] + '.' + nfile.split('/')[-1].split('.')[-2],
            '%Y%m%d.%H%M%S'
        )
    except (ValueError, IndexError):
        expected = None

    try:
        dt = pd.to_datetime(radar.time["data"], unit='ms')[-1]
    except (ValueError, OverflowError, IndexError):
        dt = None

    if expected is None:
        return dt

    # Accept the stored time only when it lands on the day the filename claims.
    if dt is not None and abs(dt - pd.Timestamp(expected)) <= pd.Timedelta(days=1):
        return dt

    print("WARNING: volume time " + str(dt) + " disagrees with filename "
          + nfile.split('/')[-1] + "; using the filename timestamp")
    return pd.Timestamp(expected)

def match_datasets_act(column, ground, site, discard, resample='auto', DataSet=False):
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
        Resampling behaviour. The default, 'auto', resamples each variable
        individually: accumulation variables (see SUM_VARS) are summed across the
        5 minute window and every other variable is averaged. Set to 'skip' to
        interpolate onto the radar time without resampling first.

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
            if len(ground) > 1:
                # adjust the RWP heights above ground level
                grd_ds['height'] = grd_ds.height.data + grd_ds.alt.data[0]
            else:
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
        grd_ds = grd_ds.interp(height=RADCLSS_HEIGHTS, method='linear')

    # Resample the ground data to 5 min and interpolate to the CSU X-Band time.
    # Resample each variable on its own: summing a state variable (pressure,
    # temperature, cloud base height, ...) across the window multiplies it by the
    # number of input samples, so only true accumulations are summed.
    # Keep data variable attributes to help distingish between instruments/locations
    if resample.split('=')[-1] == 'skip':
        matched = grd_ds.interp(time=column.time, method='linear')
    else:
        # Build the matched dataset one variable at a time
        matched = xr.Dataset()
        for var_name, da in grd_ds.data_vars.items():
            # Variables without a time dimension cannot be resampled onto the
            # radar cadence; carry them through untouched
            if 'time' not in da.dims:
                matched[var_name] = da
                continue
            resampled = da.resample(time='5Min', closed='right')
            if var_name in SUM_VARS:
                interpol_da = resampled.sum(keep_attrs=True)
            else:
                interpol_da = resampled.mean(keep_attrs=True)
            interpol_da = interpol_da.interp(time=column.time, method='linear')
            # The referenced QC companions were dropped on read; drop the dangling
            # reference so the output does not point at variables it does not have
            if 'ancillary_variables' in interpol_da.attrs:
                del interpol_da.attrs['ancillary_variables']
            matched[var_name] = interpol_da
        matched.attrs = grd_ds.attrs

    # Add SAIL site location as a dimension for the Pluvio data
    matched = matched.assign_coords(coords=dict(station=site))
    matched = matched.expand_dims('station')
   
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
        matched[var].attrs.update(source=grd_ds.datastream)

    # Merge the two DataSets
    column = xr.merge([column, matched])

    # Free up some memory
    del grd_ds, matched
   
    return column

def adjust_radclss_dod(radclss, dod):
    """
    Bring a RadCLss DataSet up to the RadCLss Data Object Description.

    The RadCLss dataset is the base: every variable that was actually extracted
    or matched keeps its own data and attributes. The DOD only supplies the
    variables that are absent (because an input datastream was missing for the
    day), the variable ordering, and the global attributes.

    Variables sourced from the DOD are filled with NaN rather than the DOD's
    own placeholder values, so that on write they take the single declared
    missing value code instead of a second, undeclared one.

    Parameters
    ----------
    radclss : Xarray DataSet
        extracted columns and in-situ sensor file
    dod : Xarray DataSet
        expected datastreams and data standards for RadCLss

    Returns
    -------
    radclss : Xarray DataSet
        Corrected RadCLss that has all expected parmeters and attributes
    """
    # Supplied DOD has correct data attributes and all required parameters.
    # Add any parameter the RadCLss dataset is missing.
    print("\n Variables not within RadCLss ds:")
    for var in dod.variables:
        # Make sure the variable isn't a dimension
        if var in dod.dims or var in radclss.variables:
            continue
        print(var)
        # Size each dimension from RadCLss where it has one, the DOD otherwise
        new_size = [radclss.sizes[dim] if dim in radclss.sizes else dod.sizes[dim]
                    for dim in dod[var].dims]
        # Fill with NaN so the write encodes the declared missing value code
        new_array = xr.DataArray(np.full(new_size, np.nan), dims=dod[var].dims)
        new_array.attrs = dod[var].attrs
        radclss[var] = new_array

        del new_size, new_array

    # `alt` is a per-station constant; the DOD carries a redundant time
    # dimension on it that the shipped files do not
    if 'alt' in radclss.variables and 'time' in radclss['alt'].dims:
        attrs = radclss['alt'].attrs
        radclss['alt'] = radclss['alt'].isel(time=0, drop=True)
        radclss['alt'].attrs = attrs

    # The DOD is authoritative for data variable attributes. Taking the input
    # datastream's attributes instead would carry over valid_min/valid_max/
    # valid_delta that no longer describe the 5 minute resampled values.
    # Coordinates are excluded: the DOD holds placeholder entries for them.
    for var in radclss.data_vars:
        if var not in dod.variables:
            continue
        source = radclss[var].attrs.get('source')
        radclss[var].attrs = dict(dod[var].attrs)
        if source is not None:
            radclss[var].attrs.setdefault('source', source)

    # Coordinate attributes, matching the shipped files
    for var, attrs in COORD_ATTRS.items():
        if var in radclss.variables:
            radclss[var].attrs = dict(attrs)

    # Stale time units would otherwise fight the encoding applied on write
    for var in ('time', 'time_offset', 'base_time'):
        if var in radclss.variables and 'units' in radclss[var].attrs:
            del radclss[var].attrs['units']

    # Anything the DOD does not define is not part of this datastream
    extra = [var for var in radclss.data_vars if var not in dod.variables]
    if extra:
        print(" Dropping variables not defined by the DOD: " + ", ".join(extra))
        radclss = radclss.drop_vars(extra)

    # Reorder the DataArrays to match the ARM Data Object Description, which
    # puts the station location variables last as lat/lon/alt. The shipped
    # files carry alt/lat/lon, an artefact of a post-hoc correction; the DOD
    # order is the approved one and is what is written here.
    dod_order = [var for var in dod.variables
                 if var in radclss.data_vars and var not in radclss.dims]
    radclss = radclss[dod_order]

    # Update the global attributes
    radclss.attrs = dict(dod.attrs)
    radclss.attrs['command_line'] = "python sail_radclss.py --array True"
    radclss.attrs['dod_version'] = "xprecipradarradclss-c2-1.4"
    radclss.attrs['site_id'] = "guc"
    radclss.attrs['platform_id'] = "xprecipradarradclss"
    radclss.attrs['facility_id'] = "S2"
    radclss.attrs['data_level'] = "c2"
    radclss.attrs['location_description'] = ("Surface Atmosphere Integrated Field Laboratory "
                                             "(SAIL), Crested Butte, Colorado")
    radclss.attrs['datastream'] = DATASTREAM
    # Single delimited string, matching how `field_names` lists its values in
    # this same DOD. The shipped files store this as a vector-valued attribute
    # with one element per datastream.
    radclss.attrs['input_datastreams'] = ", ".join(['gucxprecipradarcmacppiS2.c1',
                                                    'gucwbpluvio2M1.a1',
                                                    'gucmetM1.b1',
                                                    'gucldM1.b1',
                                                    'gucldS2.b1',
                                                    'guc915rwpprecipmeanlowM1.a1',
                                                    'gucceilM1.b1',
                                                    'gucsondewnpnM1.b1'])
    radclss.attrs['history'] = ("created by user jrobrien on machine cumulus.ccs.ornl.gov at "
                                + str(datetime.datetime.now())
                                + " using Py-ART and ACT")
    # `vap_name` is not a permitted attribute for this datastream
    radclss.attrs.pop('vap_name', None)
    print("\n")
    # return radclss, close DOD file
    del dod

    return radclss

def create_radclss_figure(radclss, height=3500, outdir=None):
    """
    With the RadCLss product, generate a timeseries of radar reflectivity factor, particle size distribution and cumuluative precipitaiton 
    for the ARM SAIL M1 Site. 

    This timeseries quick is to serve as a means for evaluating the RadCLss product.

    Input
    -----
    nfile : str
        Filepath to the RadCLss file.
    height : int
        Column height to compare against in-situ sensors for precipitation accumulation. 
    outdir : str
        Path to desired output directory. If not supplied, assumes current working directory.

    Output
    ------
    timeseries : png
        Saved image of the RadCLss timeseris
    """
    # Define a status
    status = None
    # Define the date of the file
    DATE = radclss.time.data[0].astype(str).split('T')[0].replace('-', '')
    
    # Calculate the daily accumulated precipitation for the laser disdrometer and weighing bucket 
    try:
        ld_accum = act.utils.accumulate_precip(radclss.sel(station="M1"), "precip_rate").precip_rate_accumulated.compute()
    except:
        ld_accum = None
    try:
        pluvio_accum = act.utils.accumulate_precip(radclss.sel(station="M1"), "intensity_rtnrt").intensity_rtnrt_accumulated.compute()
    except:
        pluvio_accum = None

    # Resample RadCLss to 1-min temporal frequency for precipiation accumulation calculations
    # Assume min height is 3500 meters
    ds_resampled = radclss.resample(time="1min").mean().sel(height=height).sel(station="M1")
    
    # Define the Z-S relationships used with CMAC-SQUIRE-RadCLss
    zs_fields = {"Wolf_and_Snider": {"A": 110, "B": 2, "name": 'snow_rate_ws2012'},
                 "WSR_88D_Intermountain_West": {"A": 40, "B": 2, "name": 'snow_rate_ws88diw'},
                 "Matrosov et al.(2009) Braham(1990) 1": {"A": 67, "B": 1.28, "name": 'snow_rate_m2009_1'},
                 "Matrosov et al.(2009) Braham(1990) 2": {"A": 114, "B": 1.39, "name": 'snow_rate_m2009_2'},
    }

    # Calculate accumulated precipitation from the Z-S relationships
    for field in zs_fields:
        ds_resampled[zs_fields[field]["name"]].attrs["units"] = "mm/hr"
        ds_resampled = act.utils.accumulate_precip(ds_resampled, zs_fields[field]["name"])

    # Create the figure and subaxes 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(20,12))

    # Define the DateFormatter
    date_form = DateFormatter('%Y-%m-%d \n %H:%M:%S')

    #--------------------------------
    # Plot A - M1 Column Reflectivity
    #--------------------------------
    dbz_plot = radclss.sel(station="M1").corrected_reflectivity.plot(cmap='pyart_HomeyerRainbow',
                                                                     vmin=-20,
                                                                     vmax=40,
                                                                     ax=ax1,
                                                                     add_colorbar=False,
                                                                     figure=fig,
                                                                     y='height')
    ax1.set_ylim(3500, 5000)
    ax1.set_ylabel("Height Above Ground \n (m)", fontsize=14)
    ax1.set_xlabel("")
    ax1.set_title(f'Horizontal Reflectivity at ARM AMF Site', fontsize=20)
    ax1.xaxis.set_major_formatter(date_form)
    ax1.set_xlim(radclss.time.data[0], radclss.time.data[-1])
    ax1.tick_params(axis='both', which='major', labelsize=14)

    #--------------------------------
    # Plot B - Drop Size Distribution
    #--------------------------------

    # Drop Size Distribution
    ds_vmin = np.ma.masked_invalid(radclss.number_density_drops.values).min()+1
    ds_vmax = np.ma.masked_invalid(radclss.number_density_drops.values).max()+2
    if ds_vmax < 0:
        norm = colors.LogNorm(vmin=1,
                              vmax=10)
    else:
        norm = colors.LogNorm(vmin=1,
                              vmax=ds_vmax)

    dsd_plot = radclss.sel(station="M1").number_density_drops.plot(x="time",
                                                                   y="particle_size",
                                                                   norm=norm,
                                                                   cmap="pyart_HomeyerRainbow",
                                                                   add_colorbar=False,
                                                                   ax=ax2,
        )
    ax2.set_ylim(0, 15)
    ax2.set_ylabel("Particle Size \n (mm)", fontsize=14)
    ax2.set_xlabel("")
    ax2.set_title(f'Particle Size Distribution from AMF Laser Disdrometer', fontsize=20)
    ax2.set_xlim(radclss.time.data[0], radclss.time.data[-1])
    ax2.tick_params(axis='both', which='major', labelsize=14)

    #-----------------------------
    # Plot C - Total Accumulation
    #-----------------------------
    for field in zs_fields:
        relationship_name = field.replace("_", " ")
        a_coefficeint = zs_fields[field]["A"]
        b_coefficeint = zs_fields[field]["B"]
        relationship_equation = f"$Z = {a_coefficeint}S^{b_coefficeint}$"
        field_name = zs_fields[field]["name"] + "_accumulated"

        (ds_resampled[field_name]).plot(label=f'{relationship_name} ({relationship_equation})',
                                        ax=ax3
        )
    if ld_accum is not None:
        ld_accum.plot(ax=ax3,label=f"Laser Disdrometer (M1)")
    if pluvio_accum is not None:
        pluvio_accum.plot(ax=ax3,label=f"Pluvio Sensor (M1)")
    ax3.set_title(f"Cumulative Precipitation Comparison", fontsize=20)
    ax3.set_xlim(radclss.time.data[0], radclss.time.data[-1])
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_ylabel("Total Precipitation \n Since 0000 UTC \n (mm)", fontsize=14)
    if pluvio_accum is not None:
        ax3.set_ylim(0, np.max(pluvio_accum)+10)
    else:
        ax3.set_ylim(0, 5)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.xaxis.set_major_formatter(date_form)
    ax3.set_xlabel("Time [UTC]", fontsize=14)

    # ----------
    # Colorbars
    # ----------
    # Reflectivity colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.69, 0.02, 0.165])
    cbar = fig.colorbar(dbz_plot, orientation="vertical", ax=ax1, cax=cbar_ax)
    cbar.set_ticklabels(np.arange(-10, 50, 5), fontsize=14)
    cbar.set_label(label='Horizontal \n Reflectivity \n Factor ($Z_{H}$) \n (dBZ)', fontsize=16)

    # Particle Size Distribution colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax2 = fig.add_axes([0.9, 0.41, 0.02, 0.165])
    cbar2 = fig.colorbar(dsd_plot, orientation="vertical", ax=ax2, cax=cbar_ax2)
    cbar2.set_ticklabels(cbar2.get_ticks(), fontsize=14)
    cbar2.set_label(label='Number Density \n Per Unit \n Volume \n ($m^{-3}$ $mm$)', fontsize=16)

    try:
        if outdir:
            plt.savefig(outdir + "xprecipradar.radclss.timeseries." + DATE + ".png", bbox_inches="tight", dpi=300)
        else:
            plt.savefig("xprecipradar.radclss.timeseries." + DATE + ".png", bbox_inches="tight", dpi=300)
        status = "SUCCESS: " + DATE + " timeseries plot"
    except:
        status = "FAILURE: " + DATE + " timeseries plot"
        
    # Free up memory and delete files read into memory
    del ld_accum, pluvio_accum, ds_resampled, radclss
    del fig, dbz_plot, dsd_plot

    return status


def radclss(volumes, serial=True, outdir=None, postprocess=True, dod_file=None):
    """
    Extracted Radar Columns and In-Situ Sensors

    Utilizing Py-ART and ACT, extract radar columns above various sites and 
    collocate with in-situ ground based sensors.

    Within this verison of RadCLss, supported sensors are:
        - Pluvio Weighing Bucket Rain Gauge
        - Surface Meteorological Sensors (MET)
        - Laser Disdrometer (mutliple sites)
        - Radar Wind Profiler
        - Interpolated Radiosonde
        - Ceilometer

    Calls
    -----
    subset_points
    match_datasets_act

    Parameters
    ----------
    volumes : Dictionary
        Dictionary contianing files for each of the instruments, including
        all CMAC processed radar files per day. 
    
    Keywords
    --------
    serial : Boolean, Default = False
        Option to denote serial processing; used to start dask cluster for
        subsetting columns
    
    outdir : str, Default = None
        Option of specifying Location of where to write RadCLss files
    
    postprocess : Boolean, Default = True
        Option of creating timeseries plot for the processed RadCLss file. 

    dod_file : str, Default = None
        Option to supply a Data Object Description file to verify standards

    Returns
    -------
    ds : Xarray Dataset
        Daily time-series of extracted columns saved into ARM formatted netCDF files. 
    """
    discard_var = {'LD' : ['base_time', 'time_offset', 'equivalent_radar_reflectivity_ott',
                           'laserband_amplitude', 'sensor_temperature', 
                           'heating_current', 'sensor_voltage', 
                           'moment1', 'moment2', 'moment3', 'moment4',
                           'moment5', 'moment6', 'lat', 'lon', 'alt',
                           'qc_precip_rate', 'qc_weather_code', 'qc_equivalent_radar_reflectivity_ott',
                           'qc_number_detected_particles', 'qc_mor_visibility', 
                           'qc_snow_depth_intensity', 'qc_laserband_amplitude', 
                           'qc_heating_current', 'qc_sensor_voltage'
                   ],
                   'Pluvio' : ['base_time', 'time_offset', 'load_cell_temp', 
                               'heater_status', 'elec_unit_temp', 'supply_volts', 
                               'orifice_temp', 'volt_min', 'ptemp', 'lat', 'lon', 
                               'alt', 'maintenance_flag', 'reset_flag', 'qc_rh_mean', 'pluvio_status'
                   ],
                   'Met' : ['base_time', 'time_offset', 'time_bounds', 'logger_volt',
                            'logger_temp', 'qc_logger_temp', 'lat', 'lon', 'alt', 
                            'qc_temp_mean', 'qc_rh_mean', 'qc_vapor_pressure_mean', 
                            'qc_wspd_arith_mean', 'qc_wspd_vec_mean', 'qc_wdir_vec_mean', 
                            'qc_pwd_mean_vis_1min', 'qc_pwd_mean_vis_10min', 'qc_pwd_pw_code_inst',
                            'qc_pwd_pw_code_15min', 'qc_pwd_pw_code_1hr', 
                            'qc_pwd_precip_rate_mean_1min', 'qc_pwd_cumul_rain', 
                            'qc_pwd_cumul_snow', 'qc_org_precip_rate_mean', 'qc_tbrg_precip_total',
                            'qc_tbrg_precip_total_corr', 'qc_logger_volt', 
                            'qc_logger_temp', 'qc_atmos_pressure', 
                            'pwd_pw_code_inst', 'pwd_pw_code_15min', 'pwd_pw_code_1hr', 
                            'pwd_err_code',
                   ],
                   'RWP' : ['base_time', 'time_offset', 'time_bounds', 
                            'height_bounds', 'vertical_wind_speed_count',
                            'vertical_wind_speed_quality_flag', 'lat', 'lon'
                   ],
                   'ceil': ['base_time', 'time_offset', 'time_bounds', 
                            'range_bounds', 'detection_status','status_flag', 
                            'qc_first_cbh', 'qc_vertical_visibility', 
                            'qc_second_cbh', 'qc_alt_highest_signal', 'qc_third_cbh', 
                            'laser_pulse_energy', 'qc_laser_pulse_energy', 
                            'laser_temperature', 'qc_laser_temperature','window_transmission', 
                            'qc_window_transmission', 'tilt_angle', 
                            'qc_tilt_angle', 'background_light', 'qc_background_light', 
                            'sum_backscatter', 'qc_sum_backscatter', 
                            'measurement_parameters', 'status_string','alt_highest_signal', 
                            'lat', 'lon'
                   ]
              }

    print(volumes['date'] + " start subset-points: ", time.strftime("%H:%M:%S"))
    
    # Call Subset Points
    columns = []
    if serial == False:
        if volumes['sonde']:
            with LocalCluster(n_workers=4, processes=True, threads_per_worker=1, silence_logs=logging.ERROR,
                              ) as cluster, Client(cluster) as client:
                results = client.map(subset_points, volumes["radar"], sonde=volumes['sonde'])
                for done_work in as_completed(results, with_results=False):
                    try:
                        columns.append(done_work.result())
                    except Exception as error:
                        log.exception(error)
        else:
            with LocalCluster(n_workers=4, processes=True, threads_per_worker=1, silence_logs=logging.ERROR,
                              ) as cluster, Client(cluster) as client:
                results = client.map(subset_points, volumes["radar"])
                for done_work in as_completed(results, with_results=False):
                    try:
                        columns.append(done_work.result())
                    except Exception as error:
                        log.exception(error)
    else:
        if volumes['sonde']:
            for rad in volumes['radar']:
                columns.append(subset_points(rad, sonde=volumes['sonde']))
        else:
            for rad in volumes['radar']:
                columns.append(subset_points(rad))
    try:
        ds = xr.concat([data for data in columns if data], dim="time")
    except ValueError:
        ds = None
    # Free up Memory
    del columns

    # If successful column extraction, apply in-situ
    if ds:
        # Remove Global Attributes from the Column Extraction
        # Attributes make sense for single location, but not collection of sites.
        ds.attrs = {}
        # Remove the Base_Time variable from extracted column
        del ds['base_time']
        # Depending on how Dask is behaving, may be to resort time
        ds = ds.sortby("time")
        # Drop any volume that still falls outside the day being processed, so a
        # single mis-timed input cannot stretch the file timeline across years.
        # A day of margin keeps the volume that straddles midnight.
        day = pd.Timestamp(volumes['date'])
        in_day = ((ds.time >= day - pd.Timedelta(days=1))
                  & (ds.time < day + pd.Timedelta(days=2)))
        if not bool(in_day.all()):
            print("WARNING: dropping " + str(int((~in_day).sum()))
                  + " volume(s) outside " + volumes['date'])
            ds = ds.sel(time=in_day)
        print(volumes['date'] + " finish subset-points: ", time.strftime("%H:%M:%S"))
    
        # Pluvio Weighing Bucket Rain Gauge
        if volumes['pluvio']:
            pluv_site = volumes['pluvio'][0].split('gucwbpluvio2')[-1].split('.')[0]
            # Call Match Datasets ACT
            ds = match_datasets_act(ds, volumes['pluvio'][0], pluv_site, discard=discard_var['Pluvio'])
            del pluv_site

        if volumes['met']:
            # Surface Meteorological Station
            met_site = volumes['met'][0].split('gucmet')[-1].split('.')[0]
            # Call Match Datasets ACT
            ds = match_datasets_act(ds, volumes['met'][0], met_site, discard=discard_var['Met'])
            del met_site

        if volumes['ld_m1']:
            # Laser Disdrometer - Main Site
            ld_m1 = volumes['ld_m1'][0].split('gucld')[-1].split('.')[0]
            ds = match_datasets_act(ds, volumes['ld_m1'][0], ld_m1, discard=discard_var['LD'])
            del ld_m1

        if volumes['ld_s2']:
            # Laser Disdrometer - Supplemental Site
            ld_s2 = volumes['ld_s2'][0].split('gucld')[-1].split('.')[0]
            ds = match_datasets_act(ds, volumes['ld_s2'][0], ld_s2, discard=discard_var['LD'])
            del ld_s2

        if volumes['rwp']:
            # Radar Wind Profiler - Precipitation Mode (Mean)
            ds = match_datasets_act(ds, volumes['rwp'], 'M1', resample='skip', discard=discard_var['RWP'])

        if volumes['ceil']:
            # Ceilometer 10m resolution
            ds = match_datasets_act(ds, volumes['ceil'], 'M1', discard=discard_var['ceil'])
        print(volumes['date'] + " finish in-situ match: ", time.strftime("%H:%M:%S"))

        # Transform the matched dataset for consistent dimensions
        if volumes['ld_m1'] or volumes['ld_s2']:
            ds = ds.transpose('time', 'height', 'station', 'particle_size', 'raw_fall_velocity')
        else:
            # The laser disdrometer is the only source of the particle size and
            # fall velocity dimensions; supply the instrument's bin centres so the
            # DOD-sourced spectra still land on real coordinates
            default_particle = [0.062,  0.187,  0.312,  0.437,  0.562,  0.687,
                                0.812,  0.937,  1.062,  1.187,  1.375,  1.625,
                                1.875,  2.125,  2.375,  2.75,   3.25,   3.75,
                                4.25,   4.75,   5.5,    6.5,    7.5,    8.5,
                                9.5,    11.,    13.,    15.,    17.,    19.,
                                21.5,   24.]
            default_velocity = [0.05,  0.15,  0.25,  0.35,  0.45,  0.55,  0.65,
                                0.75,  0.85,  0.95,  1.1,   1.3,   1.5,   1.7,
                                1.9,   2.2,   2.6,   3.,    3.4,   3.8,   4.4,
                                5.2,   6.,    6.8,   7.6,   8.8,  10.4,   12.,
                                13.6,  15.2,  17.6,  20.8 ]
            ds = ds.transpose("time", "height", "station")
            ds = ds.assign_coords(
                particle_size=np.array(default_particle, dtype='float32'),
                raw_fall_velocity=np.array(default_velocity, dtype='float32'))

        # ---------------------------------------------------------------
        # Cumulus cannot access the DOD API, Requires locally stored file
        # ---------------------------------------------------------------
        if dod_file:
            try:
                dod = xr.open_dataset(dod_file)
                ds = adjust_radclss_dod(ds, dod)
            except (ValueError, KeyError, OSError) as e:
                print(f"Error: {e}")
                print(f"Error type: {type(e).__name__}")
                print("WARNING: Unable to Verify DOD")

        # ------------
        # Save to File
        # ------------
        # Coordinates and time variables carry no missing value code
        keys_to_skip = {'time', 'height', 'station', 'particle_size',
                        'raw_fall_velocity', 'base_time', 'time_offset'}
        # Declare the same missing value code that is actually written, as both
        # the encoded fill and the `missing_value` attribute, for every variable
        encoding_dict = {}
        for key in ds.data_vars:
            if key in keys_to_skip:
                continue
            ds[key].attrs['missing_value'] = MISSING_VALUE
            encoding_dict[key] = {'dtype': 'float64', '_FillValue': MISSING_VALUE}
        # Coordinates keep their own types; pin them so a day without the laser
        # disdrometer writes the same types as a day with it
        for key in ('particle_size', 'raw_fall_velocity'):
            if key in ds.coords:
                encoding_dict[key] = {'dtype': 'float32', '_FillValue': np.float32(np.nan)}
        encoding_dict['time'] = {
            'dtype': 'int64',
            'calendar': 'proleptic_gregorian',
            'units': 'milliseconds since ' + pd.Timestamp(ds.time.values[0]).isoformat(),
        }

        try:
            if outdir:
                out_path = outdir + DATASTREAM + '.' + volumes['date'] + '.000000.nc'
            else:
                out_path = DATASTREAM + '.' + volumes['date'] + '.000000.nc'
            # Convert DataSet to ACT DataSet for writing
            ds_out = act.io.WriteDataset(ds)
            # FillValue=False applies the encodings above rather than plain NaNs
            ds_out.write_netcdf(path=out_path,
                                FillValue=False,
                                encoding=encoding_dict)
            del ds_out
            status = ": RadCLss SUCCESS: " + volumes['date']
        except (ValueError, OSError) as e:
            print(f"Error: {e}")
            print(f"Error type: {type(e).__name__}")
            status = ": RadCLss FAILURE: " + volumes['date']

        # create timeseries plot
        if postprocess == True:
            try:
                plot_status = create_radclss_figure(ds, outdir=outdir)
                print(plot_status)
            except:
                print("PLOT FAILURE: " + volumes['date'])

        # free up memory
        del ds, encoding_dict

    else:
        # There is no column extraction
        status = ": RadCLss FAILURE (All Columns Failed to Extract): "
        del ds

    return status

def main(args):
    print("process start time: ", time.strftime("%H:%M:%S"))
    # Define directories
    ndate = args.date
    # Define the directory where the CSU-X Band CMAC2.0 files are located.
    ##RADAR_DIR = '/Users/jrobrien/ANL/Instruments/CSU-XPrecipRadar/cmac_v3_with_cals/%s/' % ndate
    RADAR_DIR = '/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradarcmacS2.c1/ppi/%s/' % ndate
    out_path = args.outdir + '/%s/' % ndate
    print("RADAR DIR: ", RADAR_DIR)
    print("OUTPATH: ", out_path)
    print("VERBOSE: ", args.verbose)

    # Define an output directory for downloaded ground instrumentation
    PLUVIO_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/gucwbpluvio2M1.a1/'
    MET_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/gucmetM1.b1/'
    LD_M1_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/gucldM1.b1/'
    LD_S2_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/gucldS2.b1/'
    SONDE_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/gucsondewnpnM1.b1/'
    RWP_DIR = '/gpfs/wolf2/arm/atm124/proj-shared/guc915rwpprecipmeanlowM1.a1/'
    CEIL_DIR = "/gpfs/wolf2/arm/atm124/proj-shared/gucceilM1.b1/"
    ##PLUVIO_DIR = '/Users/jrobrien/ARM/active/'
    ##MET_DIR = '/Users/jrobrien/ARM/active/'
    ##LD_M1_DIR = '/Users/jrobrien/ARM/active/'
    ##LD_S2_DIR = '/Users/jrobrien/ARM/active/'
    ##SONDE_DIR = '/Users/jrobrien/ARM/active/'
    ##RWP_DIR = '/Users/jrobrien/ARM/active/'
    ##CEIL_DIR = '/Users/jrobrien/ARM/active/'

    # define the number of days within the month
    if int(ndate[4:7]) == 12:
        d0 = datetime.datetime(year=int(ndate[0:4]), month=int(ndate[4:7]), day=1)
        d1 = datetime.datetime(year=int(ndate[0:4])+1, month=1, day=1)
    else:
        d0 = datetime.datetime(year=int(ndate[0:4]), month=int(ndate[4:7]), day=1)
        d1 = datetime.datetime(year=int(ndate[0:4]), month=int(ndate[4:7])+1, day=1)
    volumes = {'date': [], 'radar' : [], 'pluvio' : [], 'met' : [], 'ld_m1' : [], 
               'ld_s2' : [], 'sonde' : [], 'rwp' : [], 'ceil' : []}
    
    # Subset dictionary for desired indice 
    def ith_val_subdict(input_dict, i):
        return {k: v[i] for k, v in input_dict.items()}
    
    # iterate through files and collect together
    if args.array is True:
        day_of_month = ndate + args.day
        print("day of month: ", day_of_month)
        volumes['date'].append(day_of_month)
        volumes['pluvio'].append(sorted(glob.glob(PLUVIO_DIR + 'gucwbpluvio2M1.a1.' + day_of_month + '*.nc')))
        volumes['radar'].append(sorted(glob.glob(RADAR_DIR + 'gucxprecipradarcmacppiS2.c1.' + day_of_month + '*')))
        volumes['met'].append(sorted(glob.glob(MET_DIR + 'gucmetM1.b1.' + day_of_month + '*.cdf')))
        volumes['ld_m1'].append(sorted(glob.glob(LD_M1_DIR + 'gucldM1.b1.' + day_of_month + '*.cdf')))
        volumes['ld_s2'].append(sorted(glob.glob(LD_S2_DIR + 'gucldS2.b1.' + day_of_month + '*.cdf')))
        volumes['rwp'].append(sorted(glob.glob(RWP_DIR + 'guc915rwpprecipmeanlowM1.a1.' + day_of_month + '*.nc')))
        volumes['ceil'].append(sorted(glob.glob(CEIL_DIR + 'gucceilM1.b1.' + day_of_month + '*.nc')))
        volumes['sonde'].append(sorted(glob.glob(SONDE_DIR + 'gucsondewnpnM1.b1.' + day_of_month + '*.cdf')))
    else:
        for i in range((d1-d0).days):
            if i < 9:
                day_of_month = ndate + '0' + str(i+1)
                volumes['date'].append(day_of_month)
                volumes['pluvio'].append(sorted(glob.glob(PLUVIO_DIR + 'gucwbpluvio2M1.a1.' + day_of_month + '*.nc')))
                volumes['radar'].append(sorted(glob.glob(RADAR_DIR + 'gucxprecipradarcmacppiS2.c1.' + day_of_month + '*')))
                volumes['met'].append(sorted(glob.glob(MET_DIR + 'gucmetM1.b1.' + day_of_month + '*.cdf')))
                volumes['ld_m1'].append(sorted(glob.glob(LD_M1_DIR + 'gucldM1.b1.' + day_of_month + '*.cdf')))
                volumes['ld_s2'].append(sorted(glob.glob(LD_S2_DIR + 'gucldS2.b1.' + day_of_month + '*.cdf')))
                volumes['rwp'].append(sorted(glob.glob(RWP_DIR + 'guc915rwpprecipmeanlowM1.a1.' + day_of_month + '*.nc')))
                volumes['ceil'].append(sorted(glob.glob(CEIL_DIR + 'gucceilM1.b1.' + day_of_month + '*.nc')))
                volumes['sonde'].append(sorted(glob.glob(SONDE_DIR + 'gucsondewnpnM1.b1.' + day_of_month + '*.cdf')))
            else:
                day_of_month = ndate + str(i+1)
                volumes['date'].append(day_of_month)
                volumes['pluvio'].append(sorted(glob.glob(PLUVIO_DIR + 'gucwbpluvio2M1.a1.' + day_of_month + '*.nc')))
                volumes['radar'].append(sorted(glob.glob(RADAR_DIR + 'gucxprecipradarcmacppiS2.c1.' + day_of_month + '*')))
                volumes['met'].append(sorted(glob.glob(MET_DIR + 'gucmetM1.b1.' + day_of_month + '*.cdf')))
                volumes['ld_m1'].append(sorted(glob.glob(LD_M1_DIR + 'gucldM1.b1.' + day_of_month + '*.cdf')))
                volumes['ld_s2'].append(sorted(glob.glob(LD_S2_DIR + 'gucldS2.b1.' + day_of_month + '*.cdf')))
                volumes['rwp'].append(sorted(glob.glob(RWP_DIR + 'guc915rwpprecipmeanlowM1.a1.' + day_of_month + '*.nc')))
                volumes['ceil'].append(sorted(glob.glob(CEIL_DIR + 'gucceilM1.b1.' + day_of_month + '*.nc')))
                volumes['sonde'].append(sorted(glob.glob(SONDE_DIR + 'gucsondewnpnM1.b1.' + day_of_month + '*.cdf')))
 
    # Send volume to RadClss for processing
    for i in range(len(volumes['date'])):
        print(volumes['date'][i])
        print(volumes['radar'][i])
        nvol = ith_val_subdict(volumes, i)
        if nvol["radar"]:
            if args.verbose:
                print("serial - ", args.serial)
                print(volumes['date'][i], nvol["radar"])
            status = radclss(nvol,
                             outdir=out_path,
                             serial=args.serial,
                             postprocess=args.postproc,
                             dod_file=args.dod_file)
            print(status)
 
    print("processing finished: ", time.strftime("%H:%M:%S"))
    # free up memory
    del volumes, nvol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Matched Radar Columns and In-Situ Sensors (RadCLss) Processing." +
            "Extracts Radar columns above a given site and collocates with in-situ sensors")

    parser.add_argument("--date",
                        default="202203",
                        dest='date',
                        type=str,
                        help="[str|YYYMM format] Specific Month to Process"
    )
    parser.add_argument("--array",
                        default=False,
                        dest="array",
                        type=bool,
                        help="[bool|default=False] If Set, check for specific days to process")
    parser.add_argument("--day",
                        default="01",
                        dest='day',
                        type=str,
                        help="[str|DD format] Specific Day to Process. Checks for `array` first"
                        )
    parser.add_argument("--serial",
                        default=True,
                        dest='serial',
                        type=bool,
                        help="[bool|default=False] Process in Serial for testing"
    )
    parser.add_argument("--outdir",
                        default='/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradclssS2.c2',
                        dest='outdir',
                        type=str,
                        help="[str] Specific directory to write RadCLss to"
    )
    parser.add_argument("--postprocessing",
                        default=True,
                        dest="postproc",
                        type=bool,
                        help="[bool|default=True] Create timeseries figures using generated RadClss files"
    )
    parser.add_argument("--verbose",
                        default=False,
                        dest="verbose",
                        type=bool,
                        help="[bool|default=False] Display file paths"
    )
    parser.add_argument("--dod_file",
                        default='/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradclssS2.c2/dod/radclss_dod.c2.v1.4.nc',
                        dest="dod_file",
                        type=str,
                        help="[str] RadCLss DOD file"
    )
    args = parser.parse_args()

    main(args)
