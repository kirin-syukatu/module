import os, os.path, re, math, sys, traceback
from datetime import datetime
import importlib, time_deco
importlib.reload(time_deco)
from time_deco import time_deco

from scipy import stats, optimize, special
import numpy as np, matplotlib.pyplot as plt, pandas as pd, seaborn
from moving_average_cy import moving_average_cy as moving_average
from thresholding3_cy import thresholding3_cy as thresholding3
from time_deco import TIME_PRINT

print('my_module'+os.sep+'data_analyzer.py is loaded')

NORM_DIST_CDF = lambda x, mu, sig: 1./2. * (1. + special.erf((x - mu) / np.sqrt(2.) / sig))
NORM_DIST_PDF = lambda x, mu, sig: (1. / np.sqrt((2. * np.pi * sig**2.))) * np.exp(-((x - mu)**2.) / (2. * sig**2.))

def fit_base(_x):
    x = np.copy(_x)
    _guess = [np.mean(x), np.std(x)]
    x.sort()
    _params, _params_covariance = optimize.curve_fit( NORM_DIST_CDF, x, [(i + 1) / len(x) for i in range(len(x))], _guess )
    return _params[0]

@time_deco(TIME_PRINT, __name__)
def subtract_base(t=None, x=None, base_i=None):
	if len(t)<len(x) : x=np.concatenate([x,np.zeros(len(x)-len(t))])
	b_i, b_x, _shift = base_i, [], 20*1000//2 #0.5sec

	for k, _b in enumerate(b_i):
	    b_x.append( fit_base(x[_b-_shift:_b+_shift]) )

	x_base = np.zeros(len(t))
	x_base[0:b_i[0]] = np.ones(b_i[0])*b_x[0]
	for k in range(len(b_i)-1):
	    x_base[ b_i[k] : b_i[k+1] ] = np.linspace(b_x[k], b_x[k+1], b_i[k+1] - b_i[k])
	x_base[ b_i[-1] : len(t) ] = np.ones( len(t)-b_i[-1] ) * b_x[-1]
	x-=x_base

	return t,x

def get_stats(t=None, x=None, _n=None):
    t_var = moving_average(x=np.square(t), n=_n) - np.square(t) # t_ave = t
    x_ave = moving_average(x=x, n=_n)
    E_tx = moving_average(x=x*t, n=_n)
    cov = E_tx - x_ave*t
    v = cov/t_var

    ave_of_sq = moving_average(x=np.square(x), n=_n)
    sq_of_ave = np.square(x_ave)
    x_var = ave_of_sq - sq_of_ave
    x_sd = np.sqrt(x_var*_n/(_n-1)) # unbiased

    _shift = (_n-1)//2
    m1, m2 = np.roll(x_ave, _shift), np.roll(x_ave, -_shift)
    vn1, vn2 = np.roll(x_var, _shift)/_n, np.roll(x_var, -_shift)/_n
    tstat = (m1-m2) / np.sqrt(vn1+vn2)
    tstat[0:_shift], tstat[-_shift:] = np.zeros(_shift), np.zeros(_shift)

    dif_std = np.sqrt(moving_average(x=np.square(np.roll(x,-1)-x)/2, n=_n))

    return v, x_sd, tstat, dif_std

def detect_peak_i(_a=None, _t=None, _x=None):
    second_derivative= (np.roll(_a, 1) -_a) * (_a - np.roll(_a, -1))
    second_derivative[0], second_derivative[-1] = 0, 0
    _peak=np.linspace(0, len(_t)-1, len(_t), dtype='int')[second_derivative<0]

    peak=[]
    for i in _peak:
        _cp = i
        while _x[_cp]-_x[_cp-1] < 0:
            _cp -= 1
        peak.append(_cp)

    return peak, _peak

# detect peak position
@time_deco(TIME_PRINT, __name__)
def detect_falling_point(x,t):
    a = moving_average(201, x)
    v, x_sd, tstat, dif_std = get_stats(t=t,x=x,_n=201)

    # 15 is magic number. threshold of velocity is -3*sigma and of tstat is 4*sigma
    pv, pv2 = detect_peak_i( _a = moving_average( n=15, x=v*(v<-3*np.std(v)) ), _t=t, _x=a )
    pv_d = {i:0 for i in pv}
    pt, pt2 = detect_peak_i(_a = moving_average( n=15, x=tstat*(tstat>3*np.std(tstat)) ), _t=t, _x=x )
    peak = [i for i in pt if i in pv_d]

    return peak # list of i at peak

def get_std_fit(x):
    from scipy import optimize
    _guess = [len(x*(x<6))/len(x), len(x*(x>=6))/len(x),  4, 8, 1, 1]
    _x=x.copy()
    _x.sort()
    _cur_f = lambda  x, a1, a2, m1, m2, std1, std2: a1*NORM_DIST_CDF(x, m1, std1) + a2*NORM_DIST_CDF(x, m2, std2)
    _params, _params_covariance = optimize.curve_fit( _cur_f , _x, np.linspace(0,1, _x.shape[0]), _guess)
    return _params

def dwt_haar(x=None):
    if len(x)<=1:return np.array([x])
    d, c, res = x[0::2]-x[1::2], x[1::2]/2+x[0::2]/2, []
    res.extend(dwt_haar(c))
    res.append(d)
    return res

def inv_dwt_haar(cds):
    if len(cds)==2: return np.array([cds[0][0]+cds[1][0]/2, cds[0][0]-cds[1][0]/2])
    c, d, x = inv_dwt_haar(cds[:-1]), cds[-1], np.zeros(len(cds[-1])*2)
    x[1::2], x[0::2] = c-d/2, c+d/2
    return x

def dwt_haar_filter(x=None,up_thr=None, print_thr=False, 
                    interpolate=False, interp_kwargs=None, padding=None, debug=False): # up_thr msec以下の最大の所まで
    from math import log2
    from scipy import interpolate

#---------padding--------------
# TODO: 両側にpaddingする
    _l, _n_sd = len(x), 0
    _x_l = int(2**(log2(len(x))//1+1)-len(x))
    if not log2(len(x))%1==0:
        if padding=="average": pad = np.ones(_x_l)*np.average(x[-len(x)//100:])
        if padding=="symmetry":pad = (x[len(x)-_x_l:])[::-1]
        if padding=="zero": pad = np.zeros(_x_l)
        x=np.concatenate( [x, pad])

#---------wavelet filtering--------------
    hdt=dwt_haar(x)
    for i in range(len(hdt)):
        if (len(x)/len(hdt[i]))/20 < up_thr: # 150msecの下のがほどよい
            _n_sd=i
            break
    y=inv_dwt_haar(hdt[:_n_sd])
    if print_thr: print(_n_sd, "{} msec".format(len(x)/len(y)/20))
    msec = len(x)/len(y)/20
#---------interpolation--------------
    if interpolate:
        t=np.linspace(0,len(x)/20/1000,len(x))
        # print(len(y), len(t[::len(x)//len(y)]), len(y)-len(x[::len(x)//len(y)]))
        f = interpolate.interp1d(t[::len(x)//len(y)],y, kind=interp_kwargs["kind"])
        _t = t[:-len(x)//len(y)]
        y = f(_t)
    y = y[:-_x_l//len(y)] if not interpolate else y[:_l]

    # TODO: ずらして平均取る
    return y, msec

def get_kde_xy(hist=None, xmin=None, xmax=None, bandwidth=None):
    hist = np.array(hist)
    from sklearn.neighbors.kde import KernelDensity
    _kde_y=KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.sort(hist).reshape(hist.shape[0],1))
    _kde_x=np.linspace(xmin,xmax,1501)
    _kde_y=np.exp(_kde_y.score_samples(_kde_x[:,np.newaxis]))
    _kde_y/=max(_kde_y)
    return _kde_x, _kde_y

def get_sd_thr(data, band_d, sd_window_size, plot_fig=False):
    """
    kernel density estimationで標準偏差のピークを二つ推定する。
    適したband_widthはデータによる。大きすぎず小さすぎず。
    sdをもとめる際のwindow_sizeは10msec(201)が程よいが、データによる。
    plot_fig=Trueで結果を保存する。
    """
    import numpy as np
    x = np.load(data.raw_x_path)["arr_0"]
    t = np.linspace(0,(len(x)-1)/20/1000, len(x))
    v, sd, tstat, dif_sd = get_stats(x=x,t=t,_n=sd_window_size*2+1) # 10mseccccccccccccccccccccccccccccccccccccccccccccccccccc
    sd, msec = dwt_haar_filter(x=sd, up_thr=sd_window_size*2, print_thr=False, interpolate=True, interp_kwargs={"kind":"linear"}, padding="symmetry")

    sd.sort()
    _kde_x, _kde_y = get_kde_xy(hist=sd[::len(sd)//(3*10**3)], xmin=0, xmax=15, bandwidth=band_d)
    pjs, pks = [], []
    for j in range(len(_kde_y)-2):
        if (_kde_y[j]-_kde_y[j+1])*(_kde_y[j+1]-_kde_y[j+2])<0 and _kde_y[j]-_kde_y[j+1]<0: pjs.append(j)
    for k in range(2):
        _j = np.argmax([_kde_y[j] for j in pjs])
        pks.append(pjs[_j])
        del pjs[_j]
    _kde_d={_kde_x[k]:k for k in pks}
    t1, t2 = min(_kde_d), max(_kde_d)
    thr_a, thr_d = t1+(t2-t1)/4, t2#-(t2-t1)/5

    if plot_fig:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=[18,5])
        _color = plt.cm.Set1(np.linspace(0,1,1))[0]
        _hist, _edge = np.histogram(sd, bins=np.linspace(0,15,151))
        ax.bar(_edge[:-1]+0.05, _hist/max(_hist), width=0.1, alpha=0.3)
        ax.plot(_kde_x, _kde_y, alpha=1, label="sd:{}msec".format(msec), linewidth=1, color=_color)
        _x1, _x2, _y1, _y2 = _kde_x[_kde_d[t1]], _kde_x[_kde_d[t2]], _kde_y[_kde_d[t1]], _kde_y[_kde_d[t2]]
        ax.scatter([_kde_x[k] for k in pks], [_kde_y[k] for k in pks], label="wavelet_filter:{}msec".format(msec), color=_color, s=150, alpha=0.7)
        ax.vlines(x=thr_a, ymin=0, ymax=1, label="thr_a", color="red")
        ax.vlines(x=thr_d, ymin=0, ymax=1, label="thr_d", color="blue")
        ax.set_ylim(0,1)
        ax.set_xlim(0,15)
        ax.set_title("id:{id}, band_d:{band_d}\nthr_a={a:0.4}, thr_d={d:0.4}".format(a=thr_a, d=thr_d, id=data.id, band_d=band_d), fontsize=20)
        ax.legend(fontsize=20)
        import os
        if not os.path.exists("threshold_10msec/"): os.mkdir("threshold_10msec/")
        plt.savefig("threshold_10msec/id={id}_sd-{sd_window_size}msec_wav-{wav}_{original_path}.png".format(
            id=data.id, wav=msec, sd_window_size=sd_window_size, original_path=data.original_path.split(os.sep)[-1][:30]))
        plt.close()

    return thr_a, thr_d

def run_print_pdf(anal=None, sd_window_size=None, v_window_size=None ,band_d=None):
    data=anal.lasertrap_rawdata
    x = np.load(data.raw_x_path)["arr_0"]
    t = np.linspace(0,(len(x)-1)/20/1000, len(x))
    v,sd,tstat,dif_sd = get_stats(x=x,t=t,_n=sd_window_size*2+1) # 10msec!!!!!!!!!!
    sd, msec = dwt_haar_filter(x=sd, up_thr=sd_window_size*2, print_thr=False, interpolate=True, interp_kwargs={"kind":"linear"}, padding="symmetry")
    thr_a, thr_d = get_sd_thr(data, band_d, sd_window_size, plot_fig=False)
    x_at, x_dt, x_oth  = x[:len(sd)]*(sd<thr_a), x[:len(sd)]*(sd>=thr_d), x[:len(sd)]*(sd<=thr_d)*(sd>thr_a)

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import os
    file_name = "id={id}_sd_{msec:0.3}msec_v-{v_window_size}msec_{note1}_{note2}".format(
        id=data.id, msec=msec, note1=data.original_path.split(os.sep)[-1][:25], 
        note2=data.original_path.split(os.sep)[-1][-18:-4], v_window_size=v_window_size)
    dir_name = "trace_sd_{msec:0.3}msec-thr/".format(msec=msec)
    if not os.path.exists(dir_name): os.mkdir(dir_name)
    # pp = PdfPages(dir_name+file_name+".pdf")

    pdf_separate_num = int(data.length/20/1000/160)+1

    for psn in range(pdf_separate_num):
        with PdfPages(dir_name+file_name+"-"+str(psn)+".pdf") as pdf:
            ### 1ファイル目のみ入れる________
            if psn==0:
                hist, _ = np.histogram(sd, bins=np.linspace(0,15,151))
                plt.hist(sd, bins=np.linspace(0,15,151), alpha=0.5, label="sd")
                plt.vlines(x=thr_a, ymin=0, ymax=10**8, label="thr_a", color="red")
                plt.vlines(x=thr_d, ymin=0, ymax=10**8, label="thr_d", color="blue")
                plt.xlim(0,15)
                plt.ylim(0,max(hist))
                plt.title(anal.name)
                plt.legend()
                # plt.savefig(pp, format="pdf")
                pdf.savefig()
                plt.close()

                ccc=plt.cm.Set1(np.linspace(0,1,4))
                _vel, _for, _dur = [r.velocity_max for r in anal.runs], [r.force_max for r in anal.runs], [r.duration_time for r in anal.runs]
                fig, axs = plt.subplots(2,2, figsize=[18,18])
                axs[0][0].scatter(_for, _vel, alpha=0.5, color=ccc[0], marker=".", s=1000, label="dil:{dil}".format(dil=data.dilution_rate) )
                axs[0][0].legend()

                ax, ax2, ax3 = axs[0][1], axs[1][0], axs[1][1]

                w=0.5
                bins = np.linspace(0,10,10//w+1)
                h, e = np.histogram(_for, bins=bins)
                h=h/max(h)
                ax.bar(e[1:]-w/2, h, width=w, color=ccc[1], label="dil:{}".format(data.dilution_rate), alpha=0.5)
                ax.plot(e[1:]-w/2, h, color=ccc[1])
                ax.set_title("force histogram")
                ax.legend()

                w=50
                bins = np.linspace(0,1100,1100//w+1)
                h, e = np.histogram(_vel, bins=bins)
                h=h/max(h)
                ax2.bar(e[1:]-w/2, h, width=w, color=ccc[2], label="dil:{}".format(data.dilution_rate), alpha=0.5)
                ax2.plot(e[1:]-w/2, h, color=ccc[2])
                ax2.set_title("velocity histogram")
                ax2.legend()

                w=0.5
                bins = np.linspace(0,30,30//w+1)
                h, e = np.histogram(_dur, bins=bins)
                h=h/max(h)
                ax3.bar(e[1:]-w/2, h, width=w, color=ccc[3], label="dil:{}".format(data.dilution_rate), alpha=0.5)
                ax3.plot(e[1:]-w/2, h, color=ccc[3])
                ax3.set_title("duration histogram")
                ax3.legend()

                # plt.savefig(pp, format="pdf")
                pdf.savefig()
                plt.close()
            ### 1ファイル目のみ入れる---------


            a = moving_average(21, x)
            v,_,tstat,__ = get_stats(x=a,t=t,_n=v_window_size*2+1) # 100msec!!!!!!!!!!!!
            v, a = moving_average(1001, v), moving_average(1001, a)

            t_start, t_end, a_start, a_end = [t[r.start_i] for r in anal.runs], [t[r.end_i] for r in anal.runs], [a[r.start_i] for r in anal.runs], [a[r.end_i] for r in anal.runs]
            BREAKNOW=False
            page_size=10
            for i in range(psn, min(psn+4, len(t)//20//1000//page_size//4 + 1)):
                fig, axs = plt.subplots(4,1, figsize=[12,12])
                for k in range(len(axs)):
                    xlim_min, sps, al, s, fsz = (4*i+k)*page_size, 100,0.3,10, 10
                    _min = xlim_min*20*1000
                    _max = (xlim_min+page_size)*20*1000
                    if _max>=len(t):
                        _max = len(t)-1
                        BREAKNOW=True
                    ax = axs[k]
                    ax.plot(t[_min:_max:sps], a[_min:_max:sps], alpha=0.8, color="black", linewidth=0.1)
                    ax.scatter(t[_min:_max:sps],x_at[_min:_max:sps], alpha=al, s=s, c='red')
                    ax.scatter(t[_min:_max:sps],x_dt[_min:_max:sps], alpha=al, s=s, c='blue')
                    ax.scatter(t[_min:_max:sps],x_oth[_min:_max:sps], alpha=al, s=s, c='black')
                    # print("pdf_sep_num:{psn}, {min}->{max}, xlim_min:{a}, size:{b}".format(psn=psn, min=_min, max=_max, a=xlim_min, b=xlim_min*20*1000))
                    c_ts, c_as = t_start*(t_start>t[_min])*(t_start<t[_max]), a_start*(t_start>t[_min])*(t_start<t[_max])
                    c_te, c_ae = t_end*(t_end>t[_min])*(t_end<t[_max]), a_end*(t_end>t[_min])*(t_end<t[_max])

                    ax2 = ax.twinx()
                    ax2.plot(t[_min:_max:sps], v[_min:_max:sps], alpha=0.4, color="magenta", label="velocity")
                    ax2.set_ylim(-max(v)/5, max(v))
                    ax2.grid()
                    ax2.legend(loc="upper center", fontsize=fsz)

                    ax3 = ax.twinx()
                    ax3.plot(t[_min:_max],sd[_min:_max], alpha=0.5, label="sd")
                    ax3.hlines(y=thr_a, xmin=t[xlim_min*20*1000], xmax=t[xlim_min*20*1000]+page_size, label="thr_a", color="red", alpha=0.2, linewidth=1)
                    ax3.hlines(y=thr_d, xmin=t[xlim_min*20*1000], xmax=t[xlim_min*20*1000]+page_size, label="thr_d", color="blue", alpha=0.2, linewidth=1)
                    ax3.set_ylim(-3, 15)#0,15)
                    ax3.grid()

                    ax.scatter(c_ts, c_as, s=200, c='green', label='starting point', marker="^", alpha=0.8) # starting point
                    ax.scatter(c_te, c_ae, s=200, c='orange', label='end point', marker="v", alpha=0.8) # end point
                    ax.set_xlabel("time [sec]", fontsize=fsz)
                    ax.set_ylabel("displacement [nm]", fontsize=fsz)
                    ax.legend(loc="upper left", fontsize=fsz)

                    for ax in [ax2, ax3, ax]:
                        plt.setp(ax.get_xticklabels(), fontsize=fsz)
                        plt.setp(ax.get_yticklabels(), fontsize=fsz)
                    from matplotlib import ticker
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))    #x軸主目盛り
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  #x軸補助目盛り

                    ax3.legend(loc="upper right", fontsize=fsz)
                    ax.set_xlim(t[xlim_min*20*1000], t[xlim_min*20*1000]+page_size)
                    ax.set_ylim(-50, 250)
                    if BREAKNOW:
                        BREAKNOW=False
                        break
                plt.tight_layout()
                # plt.savefig(pp, format="pdf") # pdffffffffffff
                pdf.savefig()
                plt.close()
            # pp.close() # pdffffffffffff


if __name__=='__main__':
    import sys, os, re
    # DB_RESET, RUN_RESET, isImported = False, False, True
    DB_RESET, RUN_RESET, isImported = True, True, False
    db_path = "../db/data2.db"
    if DB_RESET or RUN_RESET:
        if os.path.exists(db_path):os.remove(db_path)
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn
