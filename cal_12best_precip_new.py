# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:58:39 2022

@author: SuperiorDrax
"""

import numpy as np
import xarray as xr
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
import os

shortname={'precip':'tp','tmp2m':'t2m'}

def load_test_data(path, var, years=slice('2017', '2018')):
    """
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window
    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    # ds = xr.open_mfdataset(path, combine='by_coords')[var]
    #
    # return ds.sel(time=years)
    with xr.open_mfdataset(path, combine='by_coords') as ds:
        da=ds[var]
    return da.sel(time=years)


def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
    Returns:
        rmse: Latitude weighted root mean squared error
    """

    t = np.intersect1d(da_fc.time, da_true.time)
    da_fc = da_fc.sel(time=t)
    da_true = da_true.sel(time=t)

    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.latitude))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error) ** 2 * weights_lat).mean(mean_dims))
    return rmse


def compute_weighted_acc(da_fc, da_true):
    clim = da_true.mean('time')
    t = np.intersect1d(da_fc.time, da_true.time)

    fa = da_fc.sel(time=t) - clim
    a = da_true.sel(time=t) - clim

    weights_lat = np.cos(np.deg2rad(da_true.latitude))
    weights_lat /= weights_lat.mean()
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = (
            np.sum(w * fa_prime * a_prime) /
            np.sqrt(
                np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
            )
    )
    return acc

if __name__=="__main__":
    # var='tmp2m'
    realfile='./data/dataframes/precip.nc'
    # st_date='20220701'
    # ed_date='20220831'
    with xr.open_dataset(realfile) as realds:
        realdata = realds['tp']

#==========pred===========

    rmse_pred = []
    acc_pred = []
    for lead_time in range(1,30):
        predfile='./Forecast_lead_2024/prate/precip_pred_'+str(lead_time)+'.nc'
        with xr.open_dataset(predfile) as predds:
            preddata = predds['tp']
        tscore1=compute_weighted_rmse(preddata,realdata).load().data
        tscore2=compute_weighted_acc(preddata,realdata).load().data
        rmse_pred.append(tscore1)
        acc_pred.append(tscore2)

    np.save("./rmse_acc_2024/precip_pred_rmse.npy", rmse_pred)
    np.save("./rmse_acc_2024/precip_pred_acc.npy", acc_pred)

# #==========ave===========
#
rmse_ave = []
acc_ave = []
for lead_time in range(1, 30):
    predfile = './Forecast_lead_2024/prate/precip_ave_' + str(lead_time) + '.nc'
    with xr.open_dataset(predfile) as predds:
        preddata = predds['tp']
    tscore1 = compute_weighted_rmse(preddata, realdata).load().data
    tscore2 = compute_weighted_acc(preddata, realdata).load().data
    rmse_ave.append(tscore1)
    acc_ave.append(tscore2)

np.save("./rmse_acc_2024/precip_ave_rmse.npy", rmse_ave)
np.save("./rmse_acc_2024/precip_ave_acc.npy", acc_ave)

# #==========cfs===========
#
rmse_cfs = []
acc_cfs = []
for lead_time in range(1, 30):
    predfile = './Forecast_lead_2024/prate/precip_cfs_' + str(lead_time) + '.nc'
    with xr.open_dataset(predfile) as predds:
        preddata = predds['tp']
    tscore1 = compute_weighted_rmse(preddata, realdata).load().data
    tscore2 = compute_weighted_acc(preddata, realdata).load().data
    rmse_cfs.append(tscore1)
    acc_cfs.append(tscore2)

np.save("./rmse_acc_2024/precip_cfs_rmse.npy", rmse_cfs)
np.save("./rmse_acc_2024/precip_cfs_acc.npy", acc_cfs)

# #==========ecmf===========
#
rmse_ecmf = []
acc_ecmf = []
for lead_time in range(1, 30):
    predfile = './Forecast_lead_2024/prate/precip_ecmf_' + str(lead_time) + '.nc'
    with xr.open_dataset(predfile) as predds:
        preddata = predds['tp']
    tscore1 = compute_weighted_rmse(preddata, realdata).load().data
    tscore2 = compute_weighted_acc(preddata, realdata).load().data
    rmse_ecmf.append(tscore1)
    acc_ecmf.append(tscore2)

np.save("./rmse_acc_2024/precip_ecmf_rmse.npy", rmse_ecmf)
np.save("./rmse_acc_2024/precip_ecmf_acc.npy", acc_ecmf)

# # #==========p0===========
rmse_p0 = []
acc_p0 = []
for lead_time in range(1, 30):
    predfile = './Forecast_lead_2024/prate/precip_p0_' + str(lead_time) + '.nc'
    with xr.open_dataset(predfile) as predds:
        preddata = predds['tp']
    tscore1 = compute_weighted_rmse(preddata, realdata).load().data
    tscore2 = compute_weighted_acc(preddata, realdata).load().data
    rmse_p0.append(tscore1)
    acc_p0.append(tscore2)
    print(tscore1)
np.save("./rmse_acc_2024/precip_p0_rmse.npy", rmse_p0)
np.save("./rmse_acc_2024/precip_p0_acc.npy", acc_p0)
# #
# # #==========p1===========
rmse_p1 = []
acc_p1 = []
for lead_time in range(1, 30):
    predfile = './Forecast_lead_2024/prate/precip_p1_' + str(lead_time) + '.nc'
    with xr.open_dataset(predfile) as predds:
        preddata = predds['tp']
    tscore1 = compute_weighted_rmse(preddata, realdata).load().data
    tscore2 = compute_weighted_acc(preddata, realdata).load().data
    rmse_p1.append(tscore1)
    acc_p1.append(tscore2)

np.save("./rmse_acc_2024/precip_p1_rmse.npy", rmse_p1)
np.save("./rmse_acc_2024/precip_p1_acc.npy", acc_p1)
# #
# # #==========p2===========
rmse_p2 = []
acc_p2 = []
for lead_time in range(1, 30):
    predfile = './Forecast_lead_2024/prate/precip_p2_' + str(lead_time) + '.nc'
    with xr.open_dataset(predfile) as predds:
        preddata = predds['tp']
    tscore1 = compute_weighted_rmse(preddata, realdata).load().data
    tscore2 = compute_weighted_acc(preddata, realdata).load().data
    rmse_p2.append(tscore1)
    acc_p2.append(tscore2)

np.save("./rmse_acc_2024/precip_p2_rmse.npy", rmse_p2)
np.save("./rmse_acc_2024/precip_p2_acc.npy", acc_p2)

# # #==========p3===========
rmse_p3 = []
acc_p3 = []
for lead_time in range(1, 30):
    predfile = './Forecast_lead_2024/prate/precip_p3_' + str(lead_time) + '.nc'
    with xr.open_dataset(predfile) as predds:
        preddata = predds['tp']
    tscore1 = compute_weighted_rmse(preddata, realdata).load().data
    tscore2 = compute_weighted_acc(preddata, realdata).load().data
    rmse_p3.append(tscore1)
    acc_p3.append(tscore2)
np.save("./rmse_acc_2024/precip_p3_rmse.npy", rmse_p3)
np.save("./rmse_acc_2024/precip_p3_acc.npy", acc_p3)
