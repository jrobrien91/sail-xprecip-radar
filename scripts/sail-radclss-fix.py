import xarray as xr
import glob

outpath = "/Users/jrobrien/ANL/Instruments/CSU-XPrecipRadar/radclss/v2/gucxprecipradarradclss_fixed/"
inpath = "/Users/jrobrien/ANL/Instruments/CSU-XPrecipRadar/radclss/v2/gucxprecipradarradclss.c2/"

def fix_v2_radclss(nfile, outpath):
    status = "SUCCESS"
    ds = None
    try:
        ds = xr.open_dataset(nfile)
    except:
        status = "FAILURE"
    if ds:
        ds['alt'] = ds['alt'].isel(time=0, drop=True)
        ds.rename_vars({'sonde_deg' : 'sonde_wdeg'})
        del ds.attrs['vap_name']
        ds.to_netcdf(outpath + nfile.split('/')[-1])

        del ds

    return status

file_list = sorted(glob.glob(inpath + "*"))

for nfile in file_list:
    nstatus = fix_v2_radclss(nfile, outpath)
    print(nfile, nstatus)