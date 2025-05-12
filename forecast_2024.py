import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime,timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


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
    
#---------------read path----------------
with open('information.txt','r') as f:
    line=f.readlines()
    setting_name1,dir_CPS=line[0].split('=')#CPSv3原始资料目录
    setting_name2,dir_ecmf=line[1].split('=')#ecmf原始资料目录
    setting_name3,dir_CFS=line[2].split('=')#CFSv2原始资料目录
    setting_name4,dir_sta=line[3].split('=')#station
    setting_name5,dir_clim=line[4].split('=')#climate
    setting_name6,dir_out=line[5].split('=')#output
    setting_name7,tt=line[6].split('=')#订正日期
    dir_CPS=dir_CPS.lstrip().rstrip()
    dir_ecmf=dir_ecmf.lstrip().rstrip()
    dir_CFS=dir_CFS.lstrip().rstrip()
    dir_sta=dir_sta.lstrip().rstrip()
    dir_clim=dir_clim.lstrip().rstrip()
    dir_out=dir_out.lstrip().rstrip()
    tt =tt.lstrip().rstrip()

#----------------read weight-----------------

weight_t2m=np.empty((29,55))
for i in range(1,30):
    weight_t2m[i-1,:]=np.load('./tmp2m_weight_2024/tmp2m_weight_dorm_mean_g'+str(i)+'.npy')

weight_tp=np.empty((30,55))
for i in range(0,30):
    weight_tp[i,:]=np.load('./precip_weight_2024/precip_weight_dorm_mean_g'+str(i)+'.npy')  

#=====read CPSv3 data

ds_p0_t2m=xr.open_dataset(dir_CPS+tt+'/daily_bcccsm_'+tt+'00_TREFHT.nc')
p0_t2m=ds_p0_t2m['TREFHT'].loc[:,20:27.5,103.2:112.8].values
p0_t2m=p0_t2m[0:30,:,:]
    
latitude_p=ds_p0_t2m['latitude'].loc[20:27.5]
longitude_p=ds_p0_t2m['longitude'].loc[103.2:112.8]
nlat_p=len(latitude_p)
nlon_p=len(longitude_p)
    
ds_p1_t2m=xr.open_dataset(dir_CPS+tt+'/daily_bcccsm_'+tt+'00p01_TREFHT.nc')
p1_t2m=ds_p1_t2m['TREFHT'].loc[:,20:27.5,103.2:112.8].values
p1_t2m=p1_t2m[0:30,:,:]
ds_p2_t2m=xr.open_dataset(dir_CPS+tt+'/daily_bcccsm_'+tt+'00p02_TREFHT.nc')
p2_t2m=ds_p2_t2m['TREFHT'].loc[:,20:27.5,103.2:112.8].values
p2_t2m=p2_t2m[0:30,:,:]
ds_p3_t2m=xr.open_dataset(dir_CPS+tt+'/daily_bcccsm_'+tt+'00p03_TREFHT.nc')
p3_t2m=ds_p3_t2m['TREFHT'].loc[:,20:27.5,103.2:112.8].values
p3_t2m=p3_t2m[0:30,:,:]

ds_p0_tp=xr.open_dataset(dir_CPS+tt+'/daily_bcccsm_'+tt+'00_PRECT.nc')
p0_tp=ds_p0_tp['PRECT'].loc[:,20:27.5,103.2:112.8].values
p0_tp=p0_tp[0:30,:,:]*24.0*60.0*60.0*1000.0
ds_p1_tp=xr.open_dataset(dir_CPS+tt+'/daily_bcccsm_'+tt+'00p01_PRECT.nc')
p1_tp=ds_p1_tp['PRECT'].loc[:,20:27.5,103.2:112.8].values
p1_tp=p1_tp[0:30,:,:]*24.0*60.0*60.0*1000.0
ds_p2_tp=xr.open_dataset(dir_CPS+tt+'/daily_bcccsm_'+tt+'00p02_PRECT.nc')
p2_tp=ds_p2_tp['PRECT'].loc[:,20:27.5,103.2:112.8].values
p2_tp=p2_tp[0:30,:,:]*24.0*60.0*60.0*1000.0
ds_p3_tp=xr.open_dataset(dir_CPS+tt+'/daily_bcccsm_'+tt+'00p03_PRECT.nc')
p3_tp=ds_p3_tp['PRECT'].loc[:,20:27.5,103.2:112.8].values
p3_tp=p3_tp[0:30,:,:]*24.0*60.0*60.0*1000.0

#==========read CFSv2 data

ds_cfs=xr.open_dataset(dir_CFS+tt+"/NAFP_TMP2M_FOR_DAY_0P9375_CFSv2_GLB_"+tt+"00.grb2",engine='cfgrib')
cfs_raw1=ds_cfs['t2m'].loc[:,27.5:20,103.2:112.8].values
    
latitude_cfs=ds_cfs['latitude'].loc[27.5:20]
longitude_cfs=ds_cfs['longitude'].loc[103.2:112.8]
nlat_cfs=len(latitude_cfs)
nlon_cfs=len(longitude_cfs)
    
cfs_t2m=np.empty((30,nlat_cfs,nlon_cfs))
for i in range(30):
    cfs_t2m[i,:,:]=cfs_raw1[0+4*i:3+4*i,:,:].mean(axis=0)
    
    
ds_cfs=xr.open_dataset(dir_CFS+tt+"/NAFP_PRATE_FOR_DAY_0P9375_CFSv2_GLB_"+tt+"00.grb2",engine='cfgrib')
cfs_raw2=ds_cfs['prate'].loc[:,27.5:20,103.2:112.8].values
    
cfs_tp=np.empty((30,nlat_cfs,nlon_cfs))
for i in range(30):
    cfs_tp[i,:,:]=cfs_raw2[0+4*i:3+4*i,:,:].sum(axis=0)*21600.0

#============read ecmf data
format_string = "%Y%m%d"
date_tt = datetime.strptime(tt, format_string)
model_date_time = model_last_date(date_tt,'ecmf')
model_date = datetime.strftime(model_date_time, '%Y%m%d')
delta = date_tt - model_date_time
ndays = delta.days
 
ds_ecmf=xr.open_dataset(dir_ecmf+model_date+'/NAFP_2T_FOR_DAY_1P50_S2S_ECMF_GLB_'+model_date+'.grib',filter_by_keys={'dataType':'pf'})
ecmf_raw1 = ds_ecmf['t2m'].loc[:,:,27.5:20,103.2:112.8]
ecmf_t2m = ecmf_raw1[:,0+ndays:30+ndays,:,:].values
    
latitude_ecmf=ds_ecmf['latitude'].loc[27.5:20]
longitude_ecmf=ds_ecmf['longitude'].loc[103.2:112.8]
nlat_ecmf=len(latitude_ecmf)
nlon_ecmf=len(longitude_ecmf)
    
ds_ecmf=xr.open_dataset(dir_ecmf+model_date+'/NAFP_TP_FOR_DAY_1P50_S2S_ECMF_GLB_'+model_date+'.grib',filter_by_keys={'dataType':'pf'})
ecmf_raw2 = ds_ecmf['tp'].loc[:,:,27.5:20,103.2:112.8]
    
ecmf_val=np.ones((50,47,nlat_ecmf,nlon_ecmf))
for i in range(1,51):
    for j in range(nlat_ecmf):
        tp_df=pd.DataFrame(ecmf_raw2.loc[i,:,latitude_ecmf[j],:].values,
                            index=ds_ecmf['step'],
                            columns=longitude_ecmf)
        for k in range(47):
            if k==0:
                ecmf_val[i-1,k,j,:] = tp_df[tp_df.index.days == k].max(axis=0).values
            else:
                ecmf_val[i-1,k,j,:] = tp_df[tp_df.index.days == k].max(axis=0).values-tp_df[tp_df.index.days == k-1].max(axis=0).values
ecmf_tp=ecmf_val[:,0+ndays:30+ndays,:,:]

#---------------interpolate----------------

lat_new=np.linspace(20.45,27.20,28)
lon_new=np.linspace(103.5,112.5,37)
nlat_new=len(lat_new)
nlon_new=len(lon_new)

#==========p
p0_t2m_interp=np.ones((30,nlat_new,nlon_new))
p1_t2m_interp=np.ones((30,nlat_new,nlon_new))
p2_t2m_interp=np.ones((30,nlat_new,nlon_new))
p3_t2m_interp=np.ones((30,nlat_new,nlon_new))

for i in range(0,30):
    interp_p0_t2m = RegularGridInterpolator((latitude_p, longitude_p), p0_t2m[i,:,:],bounds_error=False, fill_value=None)
    lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
    p0_t2m_interp[i,:,:] = interp_p0_t2m((lat_NEW, lon_NEW))

    interp_p1_t2m = RegularGridInterpolator((latitude_p, longitude_p), p1_t2m[i,:,:],bounds_error=False, fill_value=None)
    lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
    p1_t2m_interp[i,:,:] = interp_p1_t2m((lat_NEW, lon_NEW))

    interp_p2_t2m = RegularGridInterpolator((latitude_p, longitude_p), p2_t2m[i,:,:],bounds_error=False, fill_value=None)
    lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
    p2_t2m_interp[i,:,:] = interp_p2_t2m((lat_NEW, lon_NEW))

    interp_p3_t2m = RegularGridInterpolator((latitude_p, longitude_p), p3_t2m[i,:,:],bounds_error=False, fill_value=None)
    lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
    p3_t2m_interp[i,:,:] = interp_p3_t2m((lat_NEW, lon_NEW))

p0_tp_interp=np.ones((30,nlat_new,nlon_new))
p1_tp_interp=np.ones((30,nlat_new,nlon_new))
p2_tp_interp=np.ones((30,nlat_new,nlon_new))
p3_tp_interp=np.ones((30,nlat_new,nlon_new))

for i in range(0,30):
    interp_p0_tp = RegularGridInterpolator((latitude_p, longitude_p), p0_tp[i,:,:],bounds_error=False, fill_value=None)
    lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
    p0_tp_interp[i,:,:] = interp_p0_tp((lat_NEW, lon_NEW))

    interp_p1_tp = RegularGridInterpolator((latitude_p, longitude_p), p1_tp[i,:,:],bounds_error=False, fill_value=None)
    lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
    p1_tp_interp[i,:,:] = interp_p1_tp((lat_NEW, lon_NEW))

    interp_p2_tp = RegularGridInterpolator((latitude_p, longitude_p), p2_tp[i,:,:],bounds_error=False, fill_value=None)
    lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
    p2_tp_interp[i,:,:] = interp_p2_tp((lat_NEW, lon_NEW))

    interp_p3_tp = RegularGridInterpolator((latitude_p, longitude_p), p3_tp[i,:,:],bounds_error=False, fill_value=None)
    lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
    p3_tp_interp[i,:,:] = interp_p3_tp((lat_NEW, lon_NEW))
    
#===========CFSv2
cfs_t2m_interp=np.ones((30,nlat_new,nlon_new))
for i in range(0,30):
    interp_cfs_t2m = RegularGridInterpolator((latitude_cfs, longitude_cfs), cfs_t2m[i,:,:],bounds_error=False, fill_value=None)
    lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
    cfs_t2m_interp[i,:,:] = interp_cfs_t2m((lat_NEW, lon_NEW))

cfs_tp_interp=np.ones((30,nlat_new,nlon_new))
for i in range(0,30):
    interp_cfs_tp = RegularGridInterpolator((latitude_cfs, longitude_cfs), cfs_tp[i,:,:],bounds_error=False, fill_value=None)
    lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
    cfs_tp_interp[i,:,:] = interp_cfs_tp((lat_NEW, lon_NEW))
    
#===========ecmf
ecmf_t2m_interp=np.ones((50,30,nlat_new,nlon_new))
for i in range(50):
    for j in range(0,30):
        interp_ecmf_t2m = RegularGridInterpolator((latitude_ecmf, longitude_ecmf), ecmf_t2m[i,j,:,:],bounds_error=False, fill_value=None)
        lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
        ecmf_t2m_interp[i,j,:,:] = interp_ecmf_t2m((lat_NEW, lon_NEW))

ecmf_tp_interp=np.ones((50,30,nlat_new,nlon_new))
for i in range(50):
    for j in range(0,30):
        interp_ecmf_tp = RegularGridInterpolator((latitude_ecmf, longitude_ecmf), ecmf_tp[i,j,:,:],bounds_error=False, fill_value=None)
        lat_NEW, lon_NEW = np.meshgrid(lat_new, lon_new, indexing='ij')
        ecmf_tp_interp[i,j,:,:] = interp_ecmf_tp((lat_NEW, lon_NEW))
        
#----------------forecast--------------

forecast_t2m_val=np.empty((30,nlat_new,nlon_new))

forecast_t2m_val[0,:,:]=(p0_t2m_interp[0,:,:]+p1_t2m_interp[0,:,:]+p2_t2m_interp[0,:,:]+p3_t2m_interp[0,:,:]+cfs_t2m_interp[0,:,:])/5.0
for i in range(1,30):
    #====p
    wg1_t2m=p0_t2m_interp[i,:,:]*weight_t2m[i-1,0]+p1_t2m_interp[i,:,:]*weight_t2m[i-1,1]+p2_t2m_interp[i,:,:]*weight_t2m[i-1,2]+p3_t2m_interp[i,:,:]*weight_t2m[i-1,3]
    #====ecmf
    wg2_t2m=0
    for j in range(4,54):
        wg2_t2m = wg2_t2m+ecmf_t2m_interp[j-4,i,:,:]*weight_t2m[i-1,j]
    #=====cfs
    wg3_t2m=cfs_t2m_interp[i,:,:]*weight_t2m[i-1,54]
    forecast_t2m_val[i,:,:]=wg1_t2m+wg2_t2m+wg3_t2m
        
forecast_tp_val=np.empty((30,nlat_new,nlon_new))
for i in range(0,30):
    #====p
    wg1_tp=p0_tp_interp[i,:,:]*weight_tp[i,0]+p1_tp_interp[i,:,:]*weight_tp[i,1]+p2_tp_interp[i,:,:]*weight_tp[i,2]+p3_tp_interp[i,:,:]*weight_tp[i,3]
    #====ecmf
    wg2_tp=0
    for j in range(4,54):
        wg2_tp = wg2_tp+ecmf_tp_interp[j-4,i,:,:]*weight_tp[i,j]
    #=====cfs
    wg3_tp=cfs_tp_interp[i,:,:]*weight_tp[i,54]
    forecast_tp_val[i,:,:]=wg1_tp+wg2_tp+wg3_tp

#---------grid to station--------
sta_name = pd.read_csv(dir_sta+'gx90sta.csv')

station_list=sta_name.loc[:,'id'].values
x=xr.DataArray(sta_name.loc[:,'lon'].values,coords=[station_list],dims=['sta'])
y=xr.DataArray(sta_name.loc[:,'lat'].values,coords=[station_list],dims=['sta'])

forecast_t2m_val=forecast_t2m_val-273.15

forecast_t2m_da=xr.DataArray(forecast_t2m_val, dims=['time', 'latitude','longitude'], coords=[pd.date_range(start=tt,periods=30,freq='D'),lat_new, lon_new])
forecast_t2m_da_sta=forecast_t2m_da.interp(longitude=x,latitude=y,method="linear")

forecast_tp_da=xr.DataArray(forecast_tp_val, dims=['time', 'latitude','longitude'], coords=[pd.date_range(start=tt,periods=30,freq='D'),lat_new, lon_new])
forecast_tp_da_sta=forecast_tp_da.interp(longitude=x,latitude=y,method="linear")

#-----------t2m ano & tp ano---------- 
sta_clim = pd.read_table(dir_clim+'gx_day_climate.txt')
sta_clim_t2m=sta_clim['T_AVE'].values.reshape((366,91))
sta_clim_tp =sta_clim['R'].values.reshape((366,91))

clim_md=pd.date_range(start='20200101',end='20201231',freq='D')
clim_md_str = clim_md.strftime('%m%d')
clim_t2m_sta_da=xr.DataArray(sta_clim_t2m,dims=['time','sta'],coords=[clim_md_str,station_list])
clim_tp_sta_da =xr.DataArray(sta_clim_tp ,dims=['time','sta'],coords=[clim_md_str,station_list])

tt_ymd=pd.date_range(start=tt,periods=30,freq='D')
tt_md=tt_ymd.strftime('%m%d')

#------t2m ano-------
t2m_ano=np.ones((91))
for i in range(91):
    t2m_ano[i]=forecast_t2m_da_sta.loc[:,station_list[i]].mean(axis=0).values-clim_t2m_sta_da.loc[tt_md,station_list[i]].mean(axis=0).values

t2m_ano_temp=[]
for i in range(91):
    temp=format(t2m_ano[i].tolist(),"04.1f")
    t2m_ano_temp.append(temp)
t2m_ano_new=np.array(t2m_ano_temp)

#------tp ano--------
tp_ano=np.ones((91))
for i in range(91):
    tp_ano[i]=(forecast_tp_da_sta.loc[:,station_list[i]].mean(axis=0).values-clim_tp_sta_da.loc[tt_md,station_list[i]].mean(axis=0).values)/clim_tp_sta_da.loc[tt_md,station_list[i]].mean(axis=0).values
    tp_ano[i]=tp_ano[i]*100.0
tp_ano_temp=[]
for i in range(91):
    temp=format(int(tp_ano[i].tolist()),"04d")
    tp_ano_temp.append(temp)
tp_ano_new=np.array(tp_ano_temp)

df = pd.DataFrame({
    'MCTF': station_list,
    'BENN': t2m_ano_new,
    tt: tp_ano_new
                  })
df.to_csv('./output.txt', sep=' ', index=False)

