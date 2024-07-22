import numpy as np
from multiprocessing.pool import Pool
import h5py
import tools
import time
import os 

eta_std = [-1.5, 1.5]
eta_sub1 = [-2,-0.3]
eta_sub2 = [0.3, 2]
eta_sub3 = [-0.5,0.5]
pt_cut = [0.2, 2]
corre_phi_bins = [-np.pi/3,np.pi/3]
info_type = np.dtype([('dN_deta',np.float64),
                    #q-vector in std eta range
                    ('q1_std',np.complex64),('q2_std',np.complex64),
                    ('q3_std',np.complex64),('q4_std',np.complex64),
                    ('q5_std',np.complex64),('q6_std',np.complex64),
                    #q-vector in sub eta range
                    ('q2_sub1',np.complex64),('q2_sub2',np.complex64),
                    ('q3_sub1',np.complex64),('q3_sub2',np.complex64),
                    ('q4_sub1',np.complex64),('q4_sub2',np.complex64),
                    #pT sum in std eta range
                    ('pT_1',np.float64),('pT_2',np.float64),('pT_3',np.float64),
                    #pT sum in sub eta range
                    ('pT_1_sub1',np.float64),('pT_1_sub2',np.float64),
                    ('pT_2_sub1',np.float64),('pT_2_sub2',np.float64),
                    #qnpTm in std eta range
                    ('q2pT1',np.complex64),('q3pT1',np.complex64),
                    #multiplicity
                    ('M_std',np.float64),('M_sub1',np.float64),
                    ('M_sub2',np.float64)
                    ])


def calculate_info(nsample,sample,phi,charge,eta,pt):
    Info_array = np.array([],dtype = info_type)
    this_hydro_mult_array = np.array([])
    for i in range(1, nsample+1):
                #read
                phi_std = phi[(sample == i) & (charge != 0) & (eta > eta_std[0]) & 
                            (eta < eta_std[1]) & (pt > pt_cut[0]) & (pt < pt_cut[1])]  #You can change the condition.
                phi_sub1 = phi[(sample == i) & (charge != 0) & (eta > eta_sub1[0]) & 
                            (eta < eta_sub1[1]) & (pt > pt_cut[0]) & (pt < pt_cut[1])]  #You can change the condition.
                phi_sub2 = phi[(sample == i) & (charge != 0) & (eta > eta_sub2[0]) & 
                            (eta < eta_sub2[1]) & (pt > pt_cut[0]) & (pt < pt_cut[1])]  #You can change the condition.
                if ((len(phi_std) <6) or (len(phi_sub1) <6) or (len(phi_sub2) <6)):
                    continue
                pT_std = pt[(sample == i) & (charge != 0)  & (eta > eta_std[0]) & 
                            (eta < eta_std[1]) & (pt > pt_cut[0]) & (pt < pt_cut[1])]
                pT_sub1 = pt[(sample == i) & (charge != 0)  & (eta > eta_sub1[0]) & 
                            (eta < eta_sub1[1]) & (pt > pt_cut[0]) & (pt < pt_cut[1])]
                pT_sub2 = pt[(sample == i) & (charge != 0)  & (eta > eta_sub2[0]) & 
                            (eta < eta_sub2[1]) & (pt > pt_cut[0]) & (pt < pt_cut[1])]
                #calculate
                dN_deta = (phi[(sample == i) & (charge != 0) & (eta > -0.5) & 
                            (eta < 0.5)]).size
                this_hydro_mult_array = np.append(this_hydro_mult_array,dN_deta)
                q1_std = tools.Qn(phi_std, 1)
                q2_std = tools.Qn(phi_std, 2)
                q3_std = tools.Qn(phi_std, 3)
                q4_std = tools.Qn(phi_std, 4)
                q5_std = tools.Qn(phi_std, 5)
                q6_std = tools.Qn(phi_std, 6)
                q2_sub1 = tools.Qn(phi_sub1, 2)
                q2_sub2 = tools.Qn(phi_sub2, 2)
                q3_sub1 = tools.Qn(phi_sub1, 3)
                q3_sub2 = tools.Qn(phi_sub2, 3)
                q4_sub1 = tools.Qn(phi_sub1, 4)
                q4_sub2 = tools.Qn(phi_sub2, 4)
                pT1 = tools.pTm(pT_std,1)
                pT2 = tools.pTm(pT_std,2)
                pT3 = tools.pTm(pT_std,3)
                pT1_sub1 = tools.pTm(pT_sub1,1)
                pT2_sub1 = tools.pTm(pT_sub1,2)
                pT1_sub2 = tools.pTm(pT_sub2,1)
                pT2_sub2 = tools.pTm(pT_sub2,2)
                q2pT1 = tools.QnpTm(phi_std,pT_std,2,1)
                q3pT1 = tools.QnpTm(phi_std,pT_std,3,1)
                M_std = phi_std.size
                M_sub1 = phi_sub1.size
                M_sub2 = phi_sub2.size
                #number_in_bin = angle_corre.get_parts_in_bin(phi_std,corre_phi_bins)
                Info_array = np.append(Info_array,np.array((dN_deta,
                                                    q1_std,q2_std,q3_std,q4_std,q5_std,q6_std,
                                                    q2_sub1,q2_sub2,q3_sub1,q3_sub2,q4_sub1,q4_sub2,
                                                    pT1,pT2,pT3,pT1_sub1,pT1_sub2,pT2_sub1,pT2_sub2,
                                                    q2pT1,q3pT1,
                                                    M_std,M_sub1,M_sub2),dtype=info_type))                
    return Info_array, this_hydro_mult_array


def pTspectra(nsample,sample,ID,y,pt,eta,charge):
    y_cut = [-0.1,0.1]
    tem_list = []
    cut_pt_bins = [0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    mid_point = np.array([0.225,0.275,0.325,0.375,0.45,0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85,1.95])
    dpt = np.array([0.05,0.05,0.05,0.05,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    for i in range(1, nsample+1):
        dN_deta = (pt[(sample == i) & (charge != 0) & (eta > -0.5) & 
                    (eta < 0.5)]).size
        this_hist, pt_bins = np.histogram(pt[(sample == i) & (ID == 211) & (y > y_cut[0]) 
                        & (y < y_cut[1])], bins = cut_pt_bins)
        #mid_pt = np.linspace(0.25,1.95,18)
        tem_list.append((dN_deta,this_hist/(2*np.pi*mid_point*dpt*0.2)))
    return tem_list
     

def read_h5(h5):
    '''
    Read all over sample event info of h5 file list 
    return a info_type array of it. 
    '''
    #print(os.getpid(),os.getppid())
    h5_info = np.array([],dtype = info_type)
    hydro_mult = np.array([])
    all_tem_hist = []
    with h5py.File(h5, 'r') as f:
        #print("pid = {}, ppid = {}, where = {}".format(os.getpid(),os.getppid(),read_h5))
        for event in f.keys():
            if (f[event]['sample'].size) == 0:
                continue
            nsample = f[event]['sample'][-1]
            sample = f[event]['sample']
            phi = f[event]['phi']
            charge = f[event]['charge']
            eta = f[event]['eta']
            pt = f[event]['pT']
            pid = f[event]['ID']
            y = f[event]['y']
            #hydro_mult = np.append(hydro_mult,(phi[(charge != 0) & (eta > -0.5) & 
                            #(eta < 0.5)]).size/nsample)
            tem_array, this_hydro_mult_array = calculate_info(nsample,sample,phi,charge,eta,pt)
            hydro_mult = np.append(hydro_mult,this_hydro_mult_array)
            all_tem_hist += pTspectra(nsample,sample,pid,y,pt,eta,charge)
            h5_info = np.append(h5_info,tem_array)
    return h5_info, hydro_mult, all_tem_hist


def read_h5_list(h5_list):
    with Pool() as pool:
        h5_list_info = pool.map(read_h5,h5_list)
    h5_info = [res[0] for res in h5_list_info]
    hydro_mult = [res[1] for res in h5_list_info]
    his_list_list = [res[2] for res in h5_list_info]
    his_data = []
    for his_list in his_list_list:
         his_data += his_list
    h5_info = np.concatenate(h5_info)
    hydro_mult = np.concatenate(hydro_mult)
    return h5_info, hydro_mult, his_data

def test_read_h5_list(h5_list):
    all_h5_info = []
    all_h5_mult = []
    for h5_file in h5_list:
        this_h5_info,this_hydro_mult = read_h5(h5_file)
        all_h5_info.append(this_h5_info)
        all_h5_mult.append(this_hydro_mult)
    all_h5_info = np.concatenate(all_h5_info)
    all_h5_mult = np.concatenate(all_h5_mult)
    return all_h5_info,all_h5_mult    
