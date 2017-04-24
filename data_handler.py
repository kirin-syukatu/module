import os, os.path, re, math, sys, traceback
from datetime import datetime
print(sys.version)
import importlib, database2, time_deco, data_analyzer
importlib.reload(database2)
importlib.reload(time_deco)
importlib.reload(data_analyzer)
import data_analyzer as da
from time_deco import time_deco
from data_analyzer import TIME_PRINT

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import pandas as pd
import pickle, seaborn, pywt

from moving_average_cy import moving_average_cy as moving_average
from thresholding3_cy import thresholding3_cy as thresholding3

TIME_PRINT = 1

print('my_module'+os.sep+'data_handler.py is loaded')

COM1=re.compile('\
(?P<file_number>(\d*))_\
(?P<dilution_rate>\d+)x\
(?P<construct_name>.*\d-EH-((short)|(long))|.*\d-EH)_\
(\d*)x\
(?P<beads_name>.*-(b|B)eads)_\
(?P<ATP_conc>\d+)mM(-|\B)ATP_\
(?P<assay_buffer>BRB\d+)\+\
(?P<glycerol_conc>\d+)%glycerol_\
(?P<casein_conc>\d)mg-mlCas_\
mixIn(?P<mixIn_buffer>BRB\d+)\
.*_sd-(?P<sd>\d+\.\d+)nm[_-](?P<beads_number>\d)-(?P<beads_separate_number>\d)\.txt')

COM2=re.compile('\
(?P<file_number>(\d*))_\
(?P<dilution_rate>\d+)x\
(?P<construct_name>.*\d-EH-((short)|(long))|.*\d-EH)_\
(\d*)x\
(?P<beads_name>.*(b|B)eads)_\
(?P<assay_buffer>BRB\d+)_\
(?P<ATP_conc>\d+)mM(-|\B)ATP_\
(((?P<glycerol_conc>\d+)%glycerol_)|(\B))\
(?P<casein_conc>\d+)mg-mlCas_\
mixIn(?P<mixIn_buffer>BRB\d+)_\
.*_sd-(?P<sd>\d+\.\d+)nm[_-](?P<beads_number>\d)-(?P<beads_separate_number>\d)\.txt')

COM3=re.compile('\
(?P<file_number>(\d*))_\
(?P<dilution_rate>\d+)x\
(?P<construct_name>.*\d-EH-((short)|(long))|.*\d-EH)_\
(\d*)x\
(?P<beads_name>.*(b|B)eads)_\
(?P<assay_buffer>BRB\d+)(_|\+)\
((((?P<glycerol_conc>\d+)%glycerol_)|(\B))|(\B))\
(((?P<ATP_conc>\d+)mM(-|\B)ATP_)|(\B))\
(?P<casein_conc>\d+)mg-mlCas_\
mixIn(?P<mixIn_buffer>BRB\d+)((\+1mg-mlCas)|(\B))\
.*_sd-(?P<sd>\d+\.\d+)nm[_-](?P<beads_number>\d)-(?P<beads_separate_number>\d)\.txt')

COM4=re.compile('\
(?P<file_number>(\d*))_\
(?P<dilution_rate>\d+)x\
(?P<construct_name>.*uG\dpBtHF-((short)|(long)))\
((\+(?P<dilution_rate2>\d+)x(?P<construct_name2>.*\d-EH-((short)|(long))|.*\d-EH))_|_)\
(\d*)x\
(?P<beads_name>.*(b|B)eads-200nm)_\
(((?P<ATP_conc>\d+)mM(-|\B)ATP_)|(\B))\
(?P<assay_buffer>BRB\d+)(_|\+)\
((((?P<glycerol_conc>\d+)%glycerol_)|(\B))|(\B))\
(?P<casein_conc>\d+)mg-mlCas_\
mixIn(?P<mixIn_buffer>BRB\d+)((\+1mg-mlCas)|(\B))\
.*_sd-(?P<sd>\d+\.\d+)nm[_-](?P<beads_number>\d)-(?P<beads_separate_number>\d)\.txt')

COMS = [COM1, COM2, COM3, COM4]

cnams = []#本来は名前のリストがある


def print_error(flag=None):
    exctype, value, tb = sys.exc_info()
    er = traceback.format_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
    for i,val in enumerate(er):
        if flag=='ASSERT':
            if i<=1:
                continue
        print(i,str(val).strip())

def print_now(_i, _beads_num, _file_num, cur_length, line, label):
    print(label, '_i:{}, beads_num:{}, file_num:{}'.format(_i, _beads_num, _file_num))
    print(label, 'len(x):{}, line:{}'.format(cur_length, line) )

def insert_Construct(db_d):
    session, Construct, Construct2, engine = db_d["session"], db_d["Construct"], db_d["Construct2"], db_d["engine"]
    import pandas as pd
    cur_df = pd.read_sql_table("construct", engine)
######### Construct 登録 ####################################
    for cnam in cnams:
        if not cnam in cur_df["name"]:
            session.add(Construct(name=cnam))
            session.add(Construct2(name=cnam))
    session.commit()
#########^^^^^^^^^^^^^^^^####################################

@time_deco(TIME_PRINT, __name__)
def insert_Lasertrap_rawdata_from_txt(file_path, db_d):
    """
    insert Lasertrap_rawdata into database from txt file
    """
    print("this function requires path as \"~date/file_name\"")
    session, Construct, Lasertrap_rawdata, engine = db_d["session"], db_d["Construct"], db_d["Lasertrap_rawdata"], db_d["engine"]

    data_attr = parse_filepath(file_path, db_d)
    import pandas as pd
    cur_df = pd.read_sql_table("lasertrap_rawdata", engine)
    if data_attr["original_path"] in cur_df["original_path"]:
        raise Exception("this file is exist in {db_name} already".format(db_name=engine.url))

    ### initialize current variants
    n, _beads_num, _file_num, MERGE, _line_length,     _di,                      com = \
    0,          0,         0, False,         None, {'t':0}, re.compile('#.{1} (.*)')

    ### set append function
    dd={'t':[], 'x':[], 'base_i':[], 'y':[], 'I':[]} # If want to change import data, change this line
    sa={ _k:dd[_k].append for _k in dd.keys() }

    ### start of conversion
    for i, line in enumerate(open(file_path, 'r', encoding='shift_jis')):
        line = line.strip().split('\t')

        ### pass header
        if n < 9 :
            ### データラベルと行内位置の対応着け ###
            if line[0] == 'ChannelTitle=':
                for k in range(len(line)):
                    for _k, _label in zip(['t', 'x', 'y', 'I'], ['time flies like an arrow', 'x raw', 'y raw', 'total intensity']):#zip(['t', 'x', 'y', 'x_30lp', 'x_30hp'], ['time is arrow', 'x raw', 'y raw', 'x 200Hz lp', 'チャンネル 3']):
                        # If want to change import data, change upper line
                        if _label in line[k]:
                            _di[_k] = k
                _line_length=len(line)
                if _line_length<7:
                    raise(Exception("line length is less than 7. It might be LabChart export missing"))
            n+=1
            continue
            ###############################

        ### main
        try:
            ### get comment
            if len(line)>_line_length:
                    _cur_res = re.search(com, line[-1])
                    if _cur_res:
                        if _cur_res.group(1)=='merge':
                            MERGE=True
                        elif _cur_res.group(1)=='base':
                            sa['base_i']( len(dd['t']) )
            elif line[0]=="Interval=" and MERGE:
                MERGE, n = False, 1
                print_now(i, _beads_num, _file_num, len(dd['x']), line, 'MERGE')
                continue

            ### append data to data_dictionary
            # line[_di['x']] = (lambda x: 2*10**(-6)*x**3 + 4*10**(-5)*x**2 + 1.0154*x + 0.6723) ( float(line[_di['x']]) )
            for _k in dd.keys():
                if _k=='base_i':continue
                sa[_k](float( line[_di[_k]] ))


        ### catch unintentional exceptioin
        except Exception:
            print_error()
            print_now(i, _beads_num, _file_num, len(dd['x']), line, 'unintentioanl error')

    dd["t"], dd["x"], dd["y"], dd["I"], dd["base_i"] = np.array(dd["t"]), np.array(dd["x"]), np.array(dd["y"]), np.array(dd["I"]), np.array(dd["base_i"])
    ### subtract background
    t,x = da.subtract_base(t=dd["t"], x=dd["x"], base_i=dd["base_i"])
    t,y = da.subtract_base(t=dd["t"], x=dd["y"], base_i=dd["base_i"])

    ### insert Lasertrap_rawdata
    Lrd = Lasertrap_rawdata(**data_attr)
    Lrd.raw_t_path, Lrd.raw_x_path, Lrd.raw_y_path, Lrd.length = dd["t"], dd["x"], dd["y"], len(dd["x"]) # "raw_I_path":dd["I"], 
    # Lrd.construct = _Cons
    print(Lrd)
    print("--------------------conversion has finished--------------------")
    session.add(Lrd)
    session.commit()
    print(Lrd.construct.name)

def parse_filepath(file_path, db_d):
    session, Construct, Construct2, Lasertrap_rawdata = db_d["session"], db_d["Construct"], db_d["Construct2"], db_d["Lasertrap_rawdata"]
    file_name = file_path.split(os.sep)[-1]
    import numpy as np
    COM_a = np.array([1 if re.search(COM, file_name) else 0 for COM in COMS])
    if not sum(COM_a):
        raise Exception("file name is invalid in PARSER")
    COM = COMS[np.argmax(COM_a)]
    dad = re.search(COM,file_name).groupdict()

    con_snams = {5:wild_sname, 6:S271N_sname, 7:A351V_sname, 8:V6I_sname}
    con_lnams = {5:wild_lname, 6:S271N_lname, 7:A351V_lname, 8:V6I_lname}
    uGnams = {5:uG5pBtHF_sname, 6:uG6pBtHF_sname, 7:uG7pBtHF_sname, 8:uG8pBtHF_sname}

    q_c = session.query(Construct).all()
    q_c2 = session.query(Construct2).all()


    if "pBtHF" in dad["construct_name"]:
        for j in [5,6,7,8]:
            if str(j) in dad["construct_name"]:
                _Cons = [c for c in q_c if uGnams[j] == c.name][0]
    elif "short" in dad["construct_name"]:
        for j in [5,6,7,8]:
            if str(j) in dad["construct_name"]:
                _Cons = [c for c in q_c if con_snams[j] == c.name][0]
    elif "long" in dad["construct_name"]:
        for j in [5,6,7,8]:
            if str(j) in dad["construct_name"]:
                _Cons = [c for c in q_c if con_lnams[j] == c.name][0]

    if not dad.get("construct_name2") is None:
        if "pBtHF" in dad["construct_name2"]:
            for j in [5,6,7,8]:
                if str(j) in dad["construct_name2"]:
                    _Cons2 = [c for c in q_c2 if uGnams[j] == c.name][0]
        elif "short" in dad["construct_name2"]:
            for j in [5,6,7,8]:
                if str(j) in dad["construct_name2"]:
                    _Cons2 = [c for c in q_c2 if con_snams[j] == c.name][0]
        elif "long" in dad["construct_name2"]:
            for j in [5,6,7,8]:
                if str(j) in dad["construct_name2"]:
                    _Cons2 = [c for c in q_c2 if con_lnams[j] == c.name][0]
    else:
        _Cons2=None

    data_attr={
        "date"          : file_path.split(os.sep)[-2],
        "dilution_rate" : int(dad["dilution_rate"]),
        "dilution_rate2": int(dad["dilution_rate2"]) if not dad["dilution_rate2"] is None else None,
        "beads_diameter": 200., 
        "beads_name"    : dad["beads_name"], 
        "motile_fraction":0.1, 
        "assay_buffer"  : dad["assay_buffer"],
        "ATP_conc"      : float(dad["ATP_conc"]) if not dad["ATP_conc"] is None else 2.,
        "casein_conc"   : float(dad["casein_conc"])  if not dad["casein_conc"] is None else 4.,
        "glycerol_conc" : float(dad["glycerol_conc"]) if not dad["glycerol_conc"] is None else 0., 
        "original_path" : file_path,
        "stiffness"     : 4.02/float(dad["sd"])**2,
        "beads_number"  : int(dad["beads_number"]),
        "beads_separate_number": int(dad["beads_separate_number"]),
        "file_number"   : int(dad["file_number"]),
        "construct"     : _Cons,
        "construct2"    : _Cons2
    }

    return data_attr

def insert_Run_Lasertrap_analysis(data=None, band_d=None, sd_window_size=10, v_window_size=100, plot_fig=False, db_d=None):
    """
    Lasertrap_analysisとRunを登録。
    """
    session, Lasertrap_analysis, Lasertrap_rawdata, Run = \
    db_d["session"], db_d["Lasertrap_analysis"], db_d["Lasertrap_rawdata"], db_d["Run"]

    x, t = np.load(data.raw_x_path)["arr_0"], np.linspace(0,data.length/20/1000, data.length)
    a = moving_average(x=x, n=201)
    v,sd,tstat,dif_sd = da.get_stats(x=x,t=t,_n=sd_window_size*2+1) # 10msec!!!!!!!!!!
    sd, msec = da.dwt_haar_filter(x=sd, up_thr=sd_window_size*2, print_thr=False, interpolate=True, interp_kwargs={"kind":"linear"}, padding="symmetry")
    thr_a, thr_d = da.get_sd_thr(data, band_d, sd_window_size, plot_fig=plot_fig)
    if data.id==35:
        thr_a, thr_d = 5.5, 7.5
    elif data.id==55:
        thr_a, thr_d = 5.5, 8.7
    name="separate runs with {wav}msec wavelet filtered {sd_name}, attach_threshold={a:0.4}, detach_threshold={d:0.4}".format(
        wav=msec, sd_name="sd", a=thr_a, d=thr_d)
    session.add(Lasertrap_analysis(name=name, lasertrap_rawdata=data))
    session.commit()
    from sqlalchemy import and_
    las_anal = session.query(Lasertrap_analysis)\
            .join(Lasertrap_analysis.lasertrap_rawdata)\
            .filter(and_(Lasertrap_analysis.name==name, Lasertrap_rawdata.id==data.id)).first()
################結合位置と解離位置の検出################################################################
    i_start, i_end, t_start, t_end, a_start, a_end = thresholding3(x=sd, t=t, a=a, thr=thr_a, thr_d=thr_d, dl=1)
###########################################################################################################
    a = moving_average(21, x)
    v,_,tstat,__ = da.get_stats(x=a,t=t,_n=v_window_size*2+1) # 100msec!!!!!!!!!!!!
    v, a = moving_average(1001, v), moving_average(1001, a)
    run_da=[]
    for i,(_s,_e) in enumerate(zip(i_start, i_end)):
        _s = _s-20*1000//10 + np.argmin(a[_s-20*1000//10:_s]) # 100msec遡った中の最小値を採用
        _e = _e-20*1000//20 + np.argmax(a[_e-20*1000//20:_e+20*1000//20]) # 50msec前後の中の最大値を採用
        if len(a[_s:_e])!=0:
            if a[_s]*data.stiffness>1.:continue
            _s, _e = int(_s), int(_e)
            run_da.append(
                {"start_i":_s, "end_i":_e, "velocity_max":max(v[_s:_e]), "force_max":max(a[_s:_e]*data.stiffness), 
                "duration_time":(_e-_s)/20/1000, "lasertrap_analysis_id":las_anal.id} )
    session.bulk_insert_mappings(Run, run_da)
    session.commit()
    return las_anal


class DataHandler():
    import pandas as pd
    def __init__(self, db_name, echo=False, DB_RESET=False):
        self.db_d = database2.start_db(db_name, _echo=echo, _DB_RESET=DB_RESET)

    def get_df_from_db(self, table_name):
        return pd.read_sql_table(table_name, self.db_d["engine"])

    def get_df_from_pickle(self, table_name, import_dir_path=None):
        engine = self.db_d["engine"]
        cur_time = max( [datetime.strptime("20"+dt, '%Y%m%d_%H%M%S')
                         for dt in os.listdir("E://gnuplot/notebook/db/archive/")] )\
                    .strftime('%Y%m%d_%H%M%S')[2:]
        if import_dir_path is None:
            dir_path = "E:/gnuplot/notebook/db/archive/" + cur_time + "/"
        return pd.read_pickle(dir_path + table_name)

    def get_session(self):
        return self.db_d["session"]

    def export_db_to_pickle(self, table_name):
        engine = self.db_d["engine"]
        n = datetime.now()
        dir_path = "E:/gnuplot/notebook/db/archive/" + n.strftime("%Y")[-2:] + n.strftime("%m%d_%H%M%S") + "/"
        if not os.path.exists(dir_path) : os.mkdir(dir_path)
        df = pd.read_sql_table(table_name, engine)
        df.to_pickle(dir_path+table_name)

    def export_df_to_pickle(self, df, table_name):
        n = datetime.now()
        cur_time = n.strftime("%Y")[-2:] + n.strftime("%m%d_%H%M%S")
        dir_path = "E:/gnuplot/notebook/db/archive/" + cur_time + "/"
        if not os.path.exists(dir_path) : os.mkdir(dir_path)
        df.to_pickle(dir_path+table_name)

    def import_db_from_pickle(self, table_name, import_dir_path=None):
        engine = self.db_d["engine"]
        cur_time = max( [datetime.strptime("20"+dt, '%Y%m%d_%H%M%S')
                         for dt in os.listdir("E://gnuplot/notebook/db/archive/")] )\
                    .strftime('%Y%m%d_%H%M%S')[2:]
        dir_path = "E:/gnuplot/notebook/db/archive/" + cur_time + "/" if import_dir_path is None else import_dir_path+"/"


        df = pd.read_pickle(dir_path + table_name)
        df.to_sql(table_name, engine, if_exists="append", index=False)

