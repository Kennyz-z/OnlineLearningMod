# import numpy as np
#
# weight = np.load('./precip_weight/precip_weight_dorm_mean_g2.npy')
# print(weight)
import numpy as np
import xarray as xr
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
import os

shortname = {'precip': 'tp', 'tmp2m': 't2m'}


def load_test_data(path, var, years=slice('2017', '2018')):
    """
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window
    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    ds = xr.open_mfdataset(path, combine='by_coords')[var]

    return ds.sel(time=years)


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


if __name__ == "__main__":
    var = 'precip'
    realfile = './data/dataframes/precip.nc'
    st_date = '20220701'
    ed_date = '20220915'
    realdata = load_test_data(realfile, shortname[var], years=slice(st_date, ed_date))

    # ==========pred===========
    # rmse = []
    # acc = []
    # for lead_time in range(1, 60):
    lead_time=2
    predtfile = './Forecast_lead/prate/precip_pred_' + str(lead_time) + '.nc'
    preddata = load_test_data(predtfile, 'tp', years=slice(st_date, ed_date))
    tscore1 = compute_weighted_rmse(preddata, realdata).load().data
    tscore2 = compute_weighted_acc(preddata, realdata).load().data

    print(preddata.values)
    print(realdata.values)
    # rmse.append(tscore1.tolist())
    # acc.append(tscore2)
