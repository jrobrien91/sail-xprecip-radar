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
import argparse

from dask.distributed import Client, LocalCluster, wait
from matplotlib.dates import DateFormatter
from matplotlib import colors

import pyart
import act


def create_radclss_figure(nfile, height=3500, outdir=None):
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
    # Read in the RadCLss file
    try:
        radclss = xr.open_dataset(nfile)
        print('SUCCESS: Read of RadCLss file: ', nfile)
    except:
        print('FAILURE: Unable to read RadCLss file: ', nfile)
        radclss = None

    if radclss:
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
        norm = colors.LogNorm(vmin=np.ma.masked_invalid(radclss.number_density_drops.values).min()+1,
                              vmax=np.ma.masked_invalid(radclss.number_density_drops.values).max()+2)

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
        print(status)

    return status

def main(args):
    print("process start time: ", time.strftime("%H:%M:%S"))
    # Define directories
    ndate = args.date
    
    # Define the directory where the CSU-X Band CMAC2.0 files are located.
    RADCLSS_DIR = '/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradclssS2.c2/%s/' % ndate
    if len(args.outdir) > 2:
        OUT_DIR = args.outdir
    else:
        OUT_DIR = RADCLSS_DIR + "plots/"

    # Search for all RadCLss files within the directory
    nrad = sorted(glob.glob(RADCLSS_DIR + '*.nc'))

    if args.serial is True:
        print("OUT DIRECTORY: ", OUT_DIR)
        print('going into create figure')
        status = create_radclss_figure(nrad[0], outdir=OUT_DIR)
        print(status)

        print("processing finished: ", time.strftime("%H:%M:%S"))
    else:
        print("starting dask cluster...")
        cluster = LocalCluster(n_workers=10,  threads_per_worker=1)
        with Client(cluster) as c:
            results = c.map(create_radclss_figure, nrad)
            wait(results)
        print("processing finished: ", time.strftime("%H:%M:%S"))
        # close the cluster
        del cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Postprocessing for the RADCLss product that creates daily timeseries")

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
    parser.add_argument("--outdir",
                        default='./',
                        dest='outdir',
                        type=str,
                        help="Directory to output RadCLss files to"
    )
    args = parser.parse_args()
    print("about to enter main")
    main(args)
