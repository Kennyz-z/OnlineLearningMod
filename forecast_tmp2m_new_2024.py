import numpy as np
from datetime import datetime,timedelta
import pandas as pd
import xarray as xr
import string

def model_last_date(date,model):
    dates_start=datetime.strftime(date-timedelta(days=7),'%Y%m%d')
    dates_end=datetime.strftime(date,'%Y%m%d')
    dates=get_issuance_dates(startdate=dates_start,enddate=dates_end,model=model)
    return last_issuance_date(dates,date)

def get_issuance_dates(startdate='20200101',enddate='20201231',model='BABJ'):
    if model=='ecmf':
        issuance_dates1=pd.date_range(start=startdate,end=enddate,freq='W-MON')
        issuance_dates2=pd.date_range(start=startdate,end=enddate,freq='W-THU')
        issuance_dates=issuance_dates1.union(issuance_dates2)
        return issuance_dates
    elif model=='p' or 'cfs':
        issuance_dates=pd.date_range(start=startdate,end=enddate,freq='D')
        return issuance_dates

def last_issuance_date(issuance_dates,this_date):
    # print(issuance_dates)
    # print(this_date)
    for i,issuance_date in enumerate(issuance_dates):
        if issuance_date<this_date:
            continue
        elif issuance_date==this_date:
            return issuance_date
        elif issuance_date>this_date:
            return issuance_dates[i-1]
    if issuance_dates[-1]<this_date:
        return issuance_dates[-1]

#----main------
models=[]
models_all=[]
with open('./model_list.txt','r') as f:
    for line in f.readlines():
        mod_temp=line.rstrip()
        models_all.append(mod_temp)
        models.append(mod_temp.rstrip(string.digits))

data_test = xr.open_dataset("./data/dataframes/CPSset/p0/tmp2m/20220106.nc")
t2m_test=data_test['t2m']
data_test.close()

latitude = data_test['latitude']
longitude = data_test['longitude']
nlat = len(latitude)
nlon = len(longitude)

start_date = "20230101"
end_date   = "20230430"
start_datetime = datetime.strptime(start_date, "%Y%m%d")
end_datetime = datetime.strptime(end_date, "%Y%m%d")
delta = end_datetime - start_datetime
ndays = delta.days+1

for i in range(1,30):
    print(i)
    weight=np.load('./tmp2m_weight_2024/tmp2m_weight_dorm_mean_g'+str(i)+'.npy')
    t2m_val=np.zeros((ndays,nlat,nlon))

    k=0
    for pdate in pd.date_range(start=start_date,end=end_date,freq='D'):
        pdatestr=datetime.strftime(pdate,'%Y%m%d')

        t2m_wg1 = np.zeros((nlat,nlon))
        for j in range(0,4):
            model_date_time = model_last_date(pdate, model=models[j])
            model_date = datetime.strftime(model_date_time, '%Y%m%d')
            pdatestr_new=datetime.strftime(pdate+timedelta(days=i), '%Y%m%d')
            with xr.open_dataset("./data/dataframes/CPSset/"+models_all[j]+'/tmp2m/'+model_date+'.nc') as data_p_temp:
                t2m_p = data_p_temp['t2m']
                t2m_wg1 = t2m_wg1+t2m_p.loc[pdatestr_new].values*weight[0,j]
                # t2m_wg1 = np.squeeze(t2m_wg1,axis=0)

        t2m_wg2 = 0
        for j in range(4,54):
            model_date_time = model_last_date(pdate, model=models[j])
            model_date = datetime.strftime(model_date_time, '%Y%m%d')
            with xr.open_dataset("./data/dataframes/CPSset/"+models_all[j]+'/tmp2m/'+model_date+'.nc') as data_ecmf_temp:
                t2m_ecmf = data_ecmf_temp['t2m']
                t2m_wg2 = t2m_wg2 + t2m_ecmf.loc[pdate+timedelta(days=i)].values*weight[0,j]

        t2m_wg3 = 0
        j=54
        model_date_time = model_last_date(pdate, model=models[j])
        model_date = datetime.strftime(model_date_time, '%Y%m%d')
        pdate_new = pdate+timedelta(days=i)
        with xr.open_dataset("./data/dataframes/CPSset/"+models_all[j]+'/tmp2m/'+model_date+'.nc') as data_cfs_temp:
            t2m_cfs = data_cfs_temp['t2m']
            t2m_wg3 = t2m_wg3 + t2m_cfs.loc[pdate_new].values*weight[0,j]
            
        t2m_val[k,:,:]=t2m_wg1+t2m_wg2+t2m_wg3
        k=k+1

    pdate_wr=pd.to_datetime(start_date) + timedelta(days=i)
    pdatestr_wr = datetime.strftime(pdate_wr, '%Y%m%d')
    time=pd.date_range(start=pdatestr_wr,periods=ndays,freq='D')

    t2m_DataArray=xr.DataArray(t2m_val, dims=['time', 'latitude','longitude'], coords=[time,latitude, longitude])
    t2m_DataArray.attrs['definition']='surface_air_temperature'
    t2m_DataArray.attrs['validity']='Daily averaged'
    t2m_DataArray.attrs['unit']='K'
    t2m_DataArray.attrs['long_name']='The temperature near the surface'

    t2m = xr.Dataset({'t2m': t2m_DataArray})
    t2m.to_netcdf("./Forecast_lead_2024/tmp2m"+'/tmp2m_pred_'+str(i)+'.nc')


