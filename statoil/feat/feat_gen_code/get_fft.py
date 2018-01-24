
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import io
from datetime import datetime as dt
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import json

pd.options.display.max_columns  = 999
pd.options.display.max_colwidth = 999
pd.options.display.max_rows = 999

train = pd.read_json("../../input/train.json").fillna(-1.0).replace('na', -1.0)
test  = pd.read_json("../../input/test.json").fillna(-1.0).replace('na', -1.0)

def gen_fft(dt, out_file, make_plot):
    #with open(out_file, 'w+') as f:
    res = []
    for j in range(0,dt.shape[0]):            
	#if len(str(dt.iloc[j,:].inc_angle))<=7:
        it = []
        band = 'band_1'
        x = np.array(dt.iloc[j,:][band])
        multiplier = 1.1
        threshold_h = 45.0
        threshold_v = 45.0
        mean_value = {}

        mean_value['h'] = []
        xx = []
        sph = None
        for i in range(75):
            th = np.reshape(x,(75,75))[i,:]
            sph = np.fft.fft(th)
            mnh = np.mean(abs(sph))     
            sph[abs(sph)<mnh*multiplier] = 0.0
            xx.append(abs(np.fft.ifft(sph)))
            mean_value['h'].append(mnh)
        mxh = np.max(mean_value['h'])
        mih = np.min(mean_value['h'])
        mnh = np.mean(mean_value['h'])

        mean_value['v']=[]
        yy = []
        spv = None
        for i in range(75):
            tv = np.reshape(x,(75,75))[:,i]
            spv = np.fft.fft(tv)
            mnv = np.mean(abs(spv))
            spv[abs(spv)<mnv*multiplier] = 0.0
            yy.append(abs(np.fft.ifft(spv)))
            mean_value['v'].append(mnv)

        mxv = np.max(mean_value['v'])
        miv = np.min(mean_value['v'])                
        mnv = np.mean(mean_value['v'])

        estimate_size = sum(mean_value['v'] > mnv*multiplier)*sum(mean_value['h'] > mnh*multiplier)

        yy = np.transpose(yy)
        size_v = sum(mean_value['v'] > mnv*multiplier)
        size_h = sum(mean_value['h'] > mnh*multiplier)
        for i in [mxv,miv,mnv,size_v,mxh,mih,mnh,size_h]:
            it.append(i)
        band = 'band_2'
        x = np.array(dt.iloc[j,:][band])
        multiplier = 1.1
        threshold_h = 45.0
        threshold_v = 45.0
        mean_value = {}

        mean_value['h'] = []
        xx = []
        sph = None
        for i in range(75):
            th = np.reshape(x,(75,75))[i,:]
            sph = np.fft.fft(th)
            mnh = np.mean(abs(sph))     
            sph[abs(sph)<mnh*multiplier] = 0.0
            xx.append(abs(np.fft.ifft(sph)))
            mean_value['h'].append(mnh)
        mxh = np.max(mean_value['h'])
        mih = np.min(mean_value['h'])
        mnh = np.mean(mean_value['h'])

        mean_value['v']=[]
        yy = []
        spv = None
        for i in range(75):
            tv = np.reshape(x,(75,75))[:,i]
            spv = np.fft.fft(tv)
            mnv = np.mean(abs(spv))
            spv[abs(spv)<mnv*multiplier] = 0.0
            yy.append(abs(np.fft.ifft(spv)))
            mean_value['v'].append(mnv)

        mxv = np.max(mean_value['v'])
        miv = np.min(mean_value['v'])                
        mnv = np.mean(mean_value['v'])

        estimate_size = sum(mean_value['v'] > mnv*multiplier)*sum(mean_value['h'] > mnh*multiplier)

        yy = np.transpose(yy)
        size_v = sum(mean_value['v'] > mnv*multiplier)
        size_h = sum(mean_value['h'] > mnh*multiplier)
        for i in [mxv,miv,mnv,size_v,mxh,mih,mnh,size_h]:
            it.append(i)
        res.append(it)
        #fft_data = {'band':band, 'id':dt.iloc[j,:].id, 'inc_angle':dt.iloc[j,:].inc_angle, 'is_iceberg':dt.iloc[j,:].is_iceberg, 'mxv':mxv,'miv':miv,'mnv':mnv,'mean_value_v':mean_value['v'], 'size_v': sum(mean_value['v'] > mnv*multiplier)
#		    ,'mxh':mxh,'mih':mih,'mnh':mnh,'mean_value_h':mean_value['h'], 'size_h': sum(mean_value['h'] > mnh*multiplier) }
#	f.write(str(fft_data)+'\n')
    col = ['fft_band_1_mxv','fft_band_1_miv','fft_band_1_mnv','fft_band_1_sizev','fft_band_1_mxh','fft_band_1_mih','fft_band_1_mnh','fft_band_1_sizeh','fft_band_2_mxv','fft_band_2_miv','fft_band_2_mnv','fft_band_2_sizev','fft_band_2_mxh','fft_band_2_mih','fft_band_2_mnh','fft_band_2_sizeh']
    res = pd.DataFrame(np.array(res),columns=col)
    res['id'] = dt['id']
    res.to_csv('../fft_{}.csv'.format(out_file),index=False)
gen_fft(train, 'train', False)
gen_fft(test, 'test', False)
