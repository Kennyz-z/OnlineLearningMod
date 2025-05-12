# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:06:12 2022

@author: SuperiorDrax

Contact me via Bilibili, uid:272404363(preferred)/14363489

This file will help you prepare data needed for online learning.
Original model forecast data can be downloaded from ecmwf.
https://apps.ecmwf.int/datasets/data/s2s-realtime-daily-averaged-babj/levtype=sfc/type=cf/?
The functions in this file helps you transform original grib files into h5 files used in online learning.
It also transform twice-a-week forecasts into daily forecasts. 
"""

import xarray as xr
import pandas as pd
import numpy as np
import datetime
import calendar
import os

model_factor_num={'BABJ':4,
                  'ECMF':51,
                  'KWBC':16}

model_lead_time={'BABJ':np.arange(1,61),
                 'ECMF':np.arange(1,47),
                 'KWBC':np.arange(2,45)}

def make_env_folder(chosen_path,var=['precip','tmp2m'],lead_times=np.arange(11,41)):
    assert(os.path.exists(chosen_path))
    #os.makedirs(os.path.join(chosen_path,'final'))
    #os.makedirs(os.path.join(chosen_path,'realdata'))
    #os.makedirs(os.path.join(chosen_path,'splited'))
    for model,factor in model_factor_num.items():
        for factor_num in np.arange(1,factor+1):
            for variable in var:
                #os.makedirs(os.path.join(chosen_path,'splited',model+str(factor_num),variable))
                for lead_time in lead_times:
                    os.makedirs(os.path.join(chosen_path,model+str(factor_num),variable,str(lead_time)))

def get_issuance_dates(startdate='20200101',enddate='20201231',model='BABJ'):
    if model=='BABJ':
        issuance_dates1=pd.date_range(start=startdate,end=enddate,freq='W-MON')
        issuance_dates2=pd.date_range(start=startdate,end=enddate,freq='W-THU')
        return issuance_dates1.union(issuance_dates2)
    elif model=='cfs' or model=='p':
        issuance_dates=pd.date_range(start=startdate,end=enddate,freq='D')
        return issuance_dates
    elif model=='ecmf':
        issuance_dates = pd.date_range(start=startdate, end=enddate, freq='W-THU')
        return issuance_dates


def get_issuance_dates_yearly(model='BABJ',year=2020):
    startdate=str(year)+'0101'
    enddate=str(year)+'1231'
    return get_issuance_dates(startdate=startdate,enddate=enddate,model=model)

def get_issuance_dates_monthly(model='BABJ',month=202001):
    _,days_in_month=calendar.monthrange(year=int(str(month)[0:4]),month=int(str(month)[4:6]))
    startdate=str(month)+'01'
    enddate=str(month)+str(days_in_month)
    return get_issuance_dates(startdate=startdate,enddate=enddate,model=model)

def get_issuance_dates_string(startdate='20200101',enddate='20201231',model='BABJ'):
    issuance_dates=get_issuance_dates(startdate=startdate,enddate=enddate,model=model)
    date_set=[]
    for date in issuance_dates:
        date_set.append(date.strftime('%Y-%m-%d'))
    return '/'.join(date_set)

def get_dates_from_filename(folder_path):
    files=os.listdir(folder_path)
    dates=[datetime.datetime.strptime(os.path.splitext(x)[0],'%Y%m%d') for x in files]
    return sorted(dates)

def last_issuance_date(issuance_dates,this_date):
    for i,issuance_date in enumerate(issuance_dates):
        if issuance_date<this_date:
            continue
        elif issuance_date==this_date:
            return issuance_date
        elif issuance_date>this_date:
            return issuance_dates[i-1]
    if this_date>issuance_dates[-1]:
        return issuance_dates[-1]

def cutoff_control(file,model,startdate,enddate,var,outputpath='./splited'):
    issuance_dates=get_issuance_dates(startdate=startdate,enddate=enddate,model=model)
    data=xr.open_dataset(file)

    example_file='./cressman/2015_AvgT_grid.nc'
    example = xr.open_dataarray(example_file)
    lon_grid = np.array(example.lon)
    lat_grid = np.array(example.lat)
    data=data.interp(latitude=lat_grid,longitude=lon_grid)

    outputfolder=os.path.join(outputpath,model+str(model_factor_num[model]),var)
    assert os.path.exists(outputfolder)
    try:
        lon=np.array(data.lon)
        lat=np.array(data.lat)
    except:
        lon=np.array(data.longitude)
        lat=np.array(data.latitude)
    if var=='tmp2m':
        variable=np.array(data.t2m)
        for i,this_issuance_date in enumerate(issuance_dates):
            forecast_start_time=this_issuance_date+datetime.timedelta(days=int(model_lead_time[model][0]))
            this_issuance_date=this_issuance_date.strftime('%Y%m%d')
            times=pd.date_range(start=forecast_start_time,periods=len(model_lead_time[model]),freq='D')
            this_date_var=variable[i,:,:,:].squeeze()
            output=xr.DataArray(this_date_var,dims=['time', 'latitude', 'longitude'],coords={'time':times, 'latitude': lat, 'longitude': lon},name=var)
            outputfile=os.path.join(outputfolder,this_issuance_date+'.nc')
            output.to_netcdf(outputfile)
    elif var=='precip':
        pass

def cutoff_perturbated(file,model,startdate,enddate,var,outputpath='./splited'):
    issuance_dates=get_issuance_dates(startdate=startdate,enddate=enddate,model=model)
    data=xr.open_dataset(file)
    
    example_file='./cressman/2015_AvgT_grid.nc'
    example = xr.open_dataarray(example_file)
    lon_grid = np.array(example.lon)
    lat_grid = np.array(example.lat)
    data=data.interp(latitude=lat_grid,longitude=lon_grid)

    for factor_num in range(1,model_factor_num[model]):
        outputfolder=os.path.join(outputpath,model+str(factor_num),var)
        assert os.path.exists(outputfolder)
        try:
            lon=np.array(data.lon)
            lat=np.array(data.lat)
        except:
            lon=np.array(data.longitude)
            lat=np.array(data.latitude)
        if var=='tmp2m':
            variable=np.array(data.t2m)
            variable=variable[factor_num-1,:,:,:].squeeze()
            for i,this_issuance_date in enumerate(issuance_dates):
                forecast_start_time=this_issuance_date+datetime.timedelta(days=int(model_lead_time[model][0]))
                this_issuance_date=this_issuance_date.strftime('%Y%m%d')
                times=pd.date_range(start=forecast_start_time,periods=len(model_lead_time[model]),freq='D')
                this_date_var=variable[i,:,:,:].squeeze()
                output=xr.DataArray(this_date_var,dims=['time', 'latitude', 'longitude'],coords={'time':times, 'latitude': lat, 'longitude': lon},name=var)
                outputfile=os.path.join(outputfolder,this_issuance_date+'.nc')
                output.to_netcdf(outputfile)
        elif var=='precip':
            pass

def nctodataframe(file,startdate,enddate,var,is_interp=False,outputpath='./realdata'):
    dataset=xr.open_dataset(file)
    try:
        lat=np.array(dataset.latitude,dtype='float64')
        lon=np.array(dataset.longitude,dtype='float64')
    except:
        lat=np.array(dataset.lat,dtype='float64')
        lon=np.array(dataset.lon,dtype='float64')
    if is_interp:
        example_file='./cressman/2015_AvgT_grid.nc'
        example = xr.open_dataarray(example_file)
        lon_grid = np.array(example.lon)
        lat_grid = np.array(example.lat)
        dataset=dataset.interp(latitude=lat_grid,longitude=lon_grid)
        lat=np.array(dataset.latitude,dtype='float64')
        lon=np.array(dataset.longitude,dtype='float64')
    lat=np.around(lat,decimals=2)
    lon=np.around(lon,decimals=2)
    time=pd.date_range(start=startdate,end=enddate,freq='D')
    assert(os.path.exists(outputpath))
    if var=='tmp2m':
        variable=np.array(dataset.data0).squeeze()
    elif var=='precip':
        variable=np.array(dataset.data0).squeeze()
    list_data=[]
    for i,latitude in enumerate(lat):
        for j,longitude in enumerate(lon):
            print(datetime.datetime.now(),latitude,longitude)
            for k,start_date in enumerate(time):
                list_data.append(pd.DataFrame({'lat':latitude,'lon':longitude,'start_date':start_date,var:variable[k,i,j]},index=[0]))
    h5data=pd.concat(list_data)
    h5data=h5data.set_index(['lat','lon','start_date']).squeeze().sort_index()
    outputfile=os.path.join(outputpath,var+'.h5')
    h5data.to_hdf(outputfile,key=var)

def make_daily_h5(nc_file_folder,startdate,enddate,lead_time=None,output_folder='./final'):
    if lead_time==None:
        lead_time=np.arange(11,41)
    model_and_factor=nc_file_folder.split('/')[-2]
    var=nc_file_folder.split('/')[-1]
    exist_dates=get_dates_from_filename(nc_file_folder)
    needed_dates=pd.date_range(start=startdate,end=enddate,freq='D')
    for i,needed_date in enumerate(needed_dates):
        if needed_date<exist_dates[0]:
            continue
        else:
            data_date=last_issuance_date(exist_dates,needed_date)
            data_date=data_date.strftime('%Y%m%d')
            data_file=os.path.join(nc_file_folder,data_date+'.nc')
            data=xr.open_dataset(data_file)
            try:
                lat=np.array(data.latitude,dtype='float64')
                lon=np.array(data.longitude,dtype='float64')
            except:
                lat=np.array(data.lat,dtype='float64')
                lon=np.array(data.lon,dtype='float64')
            lat=np.around(lat,decimals=2)
            lon=np.around(lon,decimals=2)
            for this_lead_time in lead_time:
                forecast_date=datetime.datetime.strftime(needed_date+datetime.timedelta(days=int(this_lead_time)),'%Y%m%d')
                if var=='prate':
                    output_file=os.path.join(output_folder,model_and_factor,'precip',str(this_lead_time),forecast_date+'.h5')
                elif var=='tmp2m':
                    output_file=os.path.join(output_folder,model_and_factor,var,str(this_lead_time),forecast_date+'.h5')
                data_output=data.sel(forecast_time0=slice(forecast_date,forecast_date))
                if var=='tmp2m':
                    variable=np.array(data_output.t2m).squeeze()
                elif var=='prate':
                    variable=np.array(data_output.tp).squeeze()
                list_data=[]
                for i,latitude in enumerate(lat):
                    for j,longitude in enumerate(lon):
                        list_data.append(pd.DataFrame({'lat':latitude,'lon':longitude,'start_date':datetime.datetime.strptime(forecast_date,'%Y%m%d'),'pred':variable[i,j]},index=[0]))
                h5data=pd.concat(list_data)
                h5data=h5data.set_index(['lat','lon','start_date']).squeeze().sort_index()
                h5data=h5data.reset_index()
                h5data.to_hdf(output_file,key=var)

if __name__=='__main__':
    with open('model_list.txt','r') as f:
        models=[]
        for line in f.readlines():
            models.append(line.rstrip('\n'))
    for lead_time in range(11,41):
        for model in models:
            make_daily_h5('/media/climate/新加卷/s2s/'+model+'/tmp2m','20200101','20201231',lead_time=[lead_time],output_folder='/media/climate/新加卷/olfile')
            make_daily_h5('/media/climate/新加卷/s2s/'+model+'/prate','20200101','20201231',lead_time=[lead_time],output_folder='/media/climate/新加卷/olfile')
    # make_daily_h5('/media/climate/新加卷/s2s/BABJ3/tmp2m','20200101','20201231',lead_time=[20],output_folder='/media/climate/新加卷/olfile')
    # cutoff_perturbated('./BABJ-pf.2020.grib','BABJ','20200101','20201231','tmp2m')
    # nctodataframe('./2020_PRE.nc','20200101','20201231','precip',is_interp=False)
    # make_env_folder('/media/climate/新加卷/olfile')
