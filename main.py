import sys
import os
import time
#from pympler import asizeof
import numpy as np
import datainfo
import datainfo_hydro
import tools
import observables
from multiprocessing.pool import Pool


#Variables definition

def calculation(this_all_info):
        tem_results = []
        this_info = np.random.choice(this_all_info,np.shape(this_all_info)[0],replace=True)
        #dNdeta = np.mean(hydro_mult[((hydro_mult>=low_cut) & (hydro_mult<high_cut))])
        v22 = observables.vn2_sub(this_info['q2_sub1'], this_info['q2_sub2'],
                        this_info['M_sub1'], this_info['M_sub2'])
        v2_sq = observables.vn_sq(this_info['q2_sub1'], this_info['q2_sub2'],
                        this_info['M_sub1'], this_info['M_sub2'])
        v32 = observables.vn2_sub(this_info['q3_sub1'], this_info['q3_sub2'],
                        this_info['M_sub1'], this_info['M_sub2'])
        #print("v32 done")
        v24 = observables.vn4(this_info['q2_std'],this_info['q4_std'],this_info['M_std'])
        #v24_2sub = observables.vn4_2sub(this_info['q2_sub1'], this_info['q2_sub2'],
                                       #this_info['q4_sub1'], this_info['q4_sub2'],
                                       # this_info['M_sub1'], this_info['M_sub2'])
        #print("v24 done")
        c2_6 = observables.cn6(this_info['q2_std'],this_info['q4_std'],this_info['q6_std'],this_info['M_std'])
        #print("c26 done")
        varv2_sq = observables.varv2_sq(this_info['q2_std'],this_info['q4_std'],
                                        this_info['q2_sub1'], this_info['q2_sub2'], 
                                        this_info['M_sub1'], this_info['M_sub2'],this_info['M_std'])
        nsc24 = observables.NSC(this_info['q4_std'],this_info['q2_std'],
                                this_info['q6_std'],this_info['q2_std'],this_info['M_std'])
        nsc32 = observables.NSC(this_info['q3_std'],this_info['q2_std'],
                                this_info['q5_std'],this_info['q1_std'],this_info['M_std'])
        mean_pT = observables.meanpT(this_info['pT_1'],this_info['M_std'])
        delta_pT_sq = observables.deltapTsq(this_info['pT_1'],this_info['pT_2'],this_info['M_std'])
        delta_pT_th = observables.deltapTthird(this_info['pT_1'],this_info['pT_2'],this_info['pT_3'],this_info['M_std'])
        delta_pT_over_pT = observables.deltapToverpTthird(this_info['pT_1'],this_info['pT_2'],this_info['pT_3'],this_info['M_std'])
        rho2_std = observables.pearson2(this_info['q2_std'], this_info['q4_std'], 
                                this_info['q2_sub1'], this_info['q2_sub2'], this_info['pT_1'], 
                                this_info['pT_2'], this_info['q2pT1'],this_info['M_std'],
                                this_info['M_sub1'],this_info['M_sub2'],mean_pT)
        covv2_pT_std = observables.cov_vn2_pT(this_info['q2_std'], this_info['q4_std'],
                                this_info['q2_sub1'], this_info['q2_sub2'], this_info['pT_1'],
                                this_info['pT_2'], this_info['q2pT1'],this_info['M_std'],
                                this_info['M_sub1'],this_info['M_sub2'],mean_pT)
        rho3_std = observables.pearson2(this_info['q3_std'], this_info['q6_std'], 
                                this_info['q3_sub1'], this_info['q3_sub2'], this_info['pT_1'], 
                                this_info['pT_2'], this_info['q3pT1'],this_info['M_std'],
                                this_info['M_sub1'],this_info['M_sub2'],mean_pT)
        cov_v2sq_ptsq = observables.cov_vn2_deltapt2(this_info['q2_sub1'],this_info['M_sub1'],
                                                        this_info['pT_1_sub2'],this_info['pT_2_sub2'],this_info['M_sub2'])
        cov_v3sq_ptsq = observables.cov_vn2_deltapt2(this_info['q3_sub1'],this_info['M_sub1'],
                                                        this_info['pT_1_sub2'],this_info['pT_2_sub2'],this_info['M_sub2'])
        tem_results.append([v22,v32,v24,np.nan,c2_6,varv2_sq,nsc24,nsc32,
                            mean_pT,delta_pT_sq,delta_pT_th,
                            rho2_std,rho3_std,cov_v2sq_ptsq,cov_v3sq_ptsq,v2_sq,covv2_pT_std,delta_pT_over_pT])
        return tem_results



def mainprocess(files_list,cal_type='event_by_event'):
    results = []
    pt_sp_res = []
    #parts_number = 0
    print('This parts files are:')
    print(files_list)
    print('***Reading information from hdf5 files...***')
    if cal_type == 'event_by_event':
        h5_list_info, hydro_mult, pt_hist = datainfo.read_h5_list(files_list)
    elif cal_type == 'hydro_event':
        h5_list_info, hydro_mult = datainfo_hydro.read_h5_list(files_list)
    print('***The shape of Information array = {}***'.format(h5_list_info.shape))
    #np.save('raw_hydro_data',h5_list_info)
    #array_memory = asizeof.asizeof(h5_list_info)/(1024*1024)
    #print('***Information array takes memory of {:.2f} mb***'.format(array_memory))
    hydro_mult = np.sort(hydro_mult)[::-1]
    #central_bin = np.array([0,99.99])
    #central_bin = np.array([0,5,10,20,30,40,50,60,70,80])
    #central_bin = np.array([0,5])
    #central_bin = np.array([0,1.5,5,7.5,10,15,20,25,30,35,40,45,50])
    central_bin = np.array([0,2,4,6,8,10,15,20,30])
    corrcentral_multcut = hydro_mult[(central_bin*0.01*len(hydro_mult)).astype(int)]
    print('dNdeta cut in centrality(%) {} is {}'.format(central_bin,corrcentral_multcut))
    #if cal_type == 'event_by_event':
        #fin_pt_spect = observables.pt_process(corrcentral_multcut[0],corrcentral_multcut[1],pt_hist)
        #print(fin_pt_spect)
        #pt_sp_res.append(fin_pt_spect)
    #print('The max and min mult is {}, {}'.format(np.max(hydro_mult),np.min(hydro_mult)))
    #corrcentral_multcut = np.linspace(900,100,81,dtype=int)
    #bootstrap process:
    print('Doing bootstrap process...')
    for i in range(0,len(corrcentral_multcut)-1):
        high_cut = corrcentral_multcut[i]
        low_cut = corrcentral_multcut[i+1]
        this_all_info = h5_list_info[((h5_list_info['dN_deta']>=low_cut) & (h5_list_info['dN_deta']<high_cut))]
        print(np.shape(this_all_info))
        resample_time = 5000
        if resample_time < 2500:
            with Pool() as pool:
                tem_res = pool.map(calculation,[this_all_info]*resample_time)
            results.append(np.array(tem_res))
        else:
            resize_resample_time = 2500
            loop_time = int(resample_time/resize_resample_time)
            print("loop_time = {}".format(loop_time))
            tem_res_list = []
            for j in range(0,loop_time):
                print(j)
                with Pool() as pool:
                    tem_res = []
                    tem_res = pool.map(calculation,[this_all_info]*resize_resample_time)
                    tem_res_list += tem_res
            results.append(np.array(tem_res_list))   
        #print('this part done')
    return np.array(results)


#go!
if __name__ == "__main__":
    begin_time = time.time()
    print("Let's go!")
    files_dir = sys.argv[1:]
    files_dir=[os.path.abspath(dir_name) for dir_name in files_dir]
    print('The directory we are going to read is {} '.format(files_dir))
    final_lists = []
    for dir_path in files_dir:
        name = os.listdir(dir_path)
        files_list = [os.path.join(dir_path,i) for i in name]
        final_lists += files_list

    event_type = 'event_by_event'
    #event_type = 'hydro_event'
    if event_type == 'event_by_event':
        #results, pt_res = mainprocess(final_lists,event_type)
        results = mainprocess(final_lists,event_type)
    elif event_type == 'hydro_event':
        results = mainprocess(final_lists,event_type)
    else:
        print('Wrong calculation type!')
        exit()
    print('result shape = {}'.format(np.shape(results)))
    #if event_type == 'event_by_event':
        #print('pt_sp_res = {}'.format(pt_res))
        #pt_sp_mean=np.mean(pt_res,axis=0)
        #pt_sp_std=np.std(pt_res,axis=0)
    results_mean = np.mean(results,axis=1)
    results_std = np.std(results,axis=1)
    #print(results)
    print(results_mean)
    print(results_std)
    #print(results_mean[:,1])
    end_time = time.time()
    print('It takes {:.2f} minutes to do all calculation'.format((end_time-begin_time)/60))
    if event_type == 'event_by_event':
        #np.savez('{}_calculation'.format(files_dir[0].split('/')[-1]), raw=results ,mean = results_mean,
            #var = results_std,pt_spe_mean=pt_sp_mean,pt_spe_std=pt_sp_std)
        np.savez('{}_calculation'.format(files_dir[0].split('/')[-1]) ,mean = results_mean,
            var = results_std)
    elif event_type == 'hydro_event':
        np.savez('{}_calculation'.format(files_dir[0].split('/')[-1]) ,mean = results_mean,
            var = results_std)
