from run import *
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from itertools import product
import os

def onlinelearning(var,lead_times,start_time,end_time,expert_models='./model_list.txt',alg='adahedged',hint='recent_g'):
    #savefolder='./test/'+expert_models
    #save_file='/home/climate/桌面/h5test/71test/factors/'+str(lead_times[0])+'day/acc/precip_weight_'+alg+'_'+hint+'.npy'
    # save_file='/mnt/e/rawpred/factors/'+str(lead_times[0])+'day/tmp2m_weight_'+alg+'_'+hint+'.npy'
    save_file='./precip_weight_2024_new/'+var+'_weight_'+alg+'_'+hint+str(lead_times[0])+'.npy'
    print(save_file)
    #expert_models='./'+expert_models+'.txt'
    print(expert_models)

    if os.path.exists(expert_models):
        models=[]
        with open(expert_models,'r') as f:
            for line in f.readlines():
                models.append(line.rstrip())
    else:
        models = expert_models.split(',')
    w_final=np.zeros([len(lead_times),len(models)])
    for lead_time_num,lead_time in enumerate(lead_times):
        target_dates_start=datetime.strftime((datetime.strptime(start_time,'%Y%m%d')+timedelta(days=lead_time)),"%Y%m%d")
        target_dates=target_dates_start+'-'+end_time
        w_final[lead_time_num,:]=runonline(var,lead_time,target_dates,expert_models=expert_models,alg=alg,hint=hint)
        print(lead_time)

    np.save(save_file,w_final)

if __name__=='__main__':
    algs=['dorm','dormplus','dub']
    hints = ['mean_g', 'prev_g']
    # algs=['dormplus']
    # hints=['prev_g']
    for alg,hint in product(algs,hints):
        for lead_time in range(0,30):
        #for i in range(1):
            #expert_models='/home/climate/桌面/h5test/71test/factors/'+str(lead_time)+'day/'+str(lead_time)+'day-acc.txt'
            # expert_models='/mnt/e/rawpred/factors/'+str(lead_time)+'day-acc.txt'
            # expert_models = './' + str(lead_time) + 'day-acc.txt'
            expert_models = "./model_list.txt"
            onlinelearning('precip',range(lead_time,lead_time+1),'20210107','20210130',expert_models=expert_models,alg=alg,hint=hint)
            #onlinelearning('precip',range(11,41),'20200102','20211231',expert_models='./model_list.txt',alg=alg,hint=hint)
