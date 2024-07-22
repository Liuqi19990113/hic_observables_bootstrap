import numpy as np
import tools


def vn2_sub(Qn_a, Qn_b, M_a, M_b):
    '''
    This function is used to calculate v_n{2} with 2 sub-events
    return v_n2
    '''
    vn2_numerator = (tools.TwoPartCorr_N_MinN_Sub(Qn_a, Qn_b, M_a, M_b)*(M_a*M_b)).sum()
    vn2_denominator = (M_a*M_b).sum()
    return np.sqrt(vn2_numerator/vn2_denominator)

def vn_sq(Qn_a, Qn_b, M_a, M_b):
    '''
    This function is used to calculate v_n{2} with 2 sub-events
    return v_n2
    '''
    vn2_numerator = (tools.TwoPartCorr_N_MinN_Sub(Qn_a, Qn_b, M_a, M_b)*(M_a*M_b)).sum()
    vn2_denominator = (M_a*M_b).sum()
    return vn2_numerator/vn2_denominator


def vn4(Qn, Q2n, M_std):
    '''
    This function is used to calculate v_n{4}
    return v_n4
    '''
    FourPartave = (((tools.FourPartCorr_N_N_MinN_MinN(Qn,Q2n,M_std)*
                    (M_std)*(M_std-1)*(M_std-2)*(M_std-3)).sum())/
                    ((M_std)*(M_std-1)*(M_std-2)*(M_std-3)).sum())
    TwoPartave = ((tools.TwoPartCorr_N_MinN(Qn, M_std)*(M_std)*(M_std-1)).sum()/
             ((M_std)*(M_std-1)).sum())
    cn4 =  FourPartave - 2*np.power(TwoPartave,2)
    if (cn4 > 0):
        return np.nan
    return np.power(-cn4,1/4)


def varv2_sq(Qn_std,Q2n_std,Qn_a, Qn_b, M_a, M_b,M_std):
    FourPartave = ((tools.FourPartCorr_N_N_MinN_MinN(Qn_std,Q2n_std,M_std)*
                    (M_std)*(M_std-1)*(M_std-2)*(M_std-3)).sum()/
                    ((M_std)*(M_std-1)*(M_std-2)*(M_std-3)).sum())
    TwoPartave = ((tools.TwoPartCorr_N_MinN(Qn_std, M_std)*(M_std)*(M_std-1)).sum()/
             ((M_std)*(M_std-1)).sum())
    cn2_sub = (tools.TwoPartCorr_N_MinN_Sub(Qn_a, Qn_b, M_a, M_b)*(M_a*M_b)).sum()/(M_a*M_b).sum()
    cn4 =  FourPartave - 2*np.power(TwoPartave,2)
    return np.power(cn2_sub,2) + cn4


def pearson(Qn_std, Q2n_std, Qn_a, Qn_b, pT1, pT2, QpT_nm, M_std, M_a, M_b, mean_pT):
    '''
    return rho(vn^2, [pT])
    '''
    pearson_num = (tools.cov_vn2_pt(Qn_std, pT1, QpT_nm, M_std, mean_pT)*
                    M_std*(M_std-1)*(M_std-2)).sum()/(M_std*(M_std-1)*(M_std-2)).sum()
    var_1 = varv2_sq(Qn_std,Q2n_std,Qn_a, Qn_b, M_a, M_b,M_std)
    var_2 = (tools.deltapT_sq(pT1, pT2, mean_pT, M_std)*(M_std)*(M_std-1)).sum()/((M_std)*(M_std-1)).sum()
    return pearson_num/np.sqrt(var_1*var_2)


def pearson2(Qn_std, Q2n_std, Qn_a, Qn_b, pT1, pT2, QpT_nm, M_std, M_a, M_b, mean_pT):
    '''
    return rho(vn^2, [pT])
    '''
    pearson_num = np.mean(tools.cov_vn2_pt(Qn_std, pT1, QpT_nm, M_std, mean_pT))
    var_1 = varv2_sq(Qn_std,Q2n_std,Qn_a, Qn_b, M_a, M_b,M_std)
    var_2 = np.mean(tools.deltapT_sq(pT1, pT2, mean_pT, M_std))
    return pearson_num/np.sqrt(var_1*var_2)

def cov_vn2_pT(Qn_std, Q2n_std, Qn_a, Qn_b, pT1, pT2, QpT_nm, M_std, M_a, M_b, mean_pT):
    '''
    return rho(vn^2, [pT])
    '''
    pearson_num = np.mean(tools.cov_vn2_pt(Qn_std, pT1, QpT_nm, M_std, mean_pT))
    #var_1 = varv2_sq(Qn_std,Q2n_std,Qn_a, Qn_b, M_a, M_b,M_std)
    #var_2 = np.mean(tools.deltapT_sq(pT1, pT2, mean_pT, M_std))
    return pearson_num

def NSC(Qm, Qn, Q_mplusn, Q_mminn, M):
    '''
    return NSC(m,n), m>n
    '''
    avevm2vn2 = ((tools.FourPartCorr_M_N_MinM_MinN(Qm, Qn, Q_mplusn, Q_mminn, M)
                *M*(M-1)*(M-2)*(M-3)).sum()/(M*(M-1)*(M-2)*(M-3)).sum())
    avevm2_avevn2 = (((tools.TwoPartCorr_N_MinN(Qm, M)*(M*(M-1))).sum()/((M*(M-1)).sum()))*
                    ((tools.TwoPartCorr_N_MinN(Qn, M)*(M*(M-1))).sum()/((M*(M-1)).sum())))
    return (avevm2vn2 - avevm2_avevn2)/avevm2_avevn2


def meanpT(pT1,M):
    return np.mean(pT1/M)


def deltapTsq(pT1, pT2, M):
    return np.mean((pT1*pT1-pT2)/(M*(M-1))) - np.power(np.mean((pT1/M)),2)


def deltapTthird(pT1, pT2, pT3, M):
    return (np.mean((pT1*pT1*pT1 - 3*pT1*pT2 + 2*pT3)/(M*(M-1)*(M-2)))
     - 3*np.mean((pT1*pT1-pT2)/(M*(M-1)))*np.mean((pT1/M))
     + 2*np.power(np.mean((pT1/M)),3))

def deltapToverpTthird(pT1, pT2, pT3, M):
    num = (np.power(pT1,3) - 3*pT1*pT2 + 2*pT3 -3*(M-2)*np.mean(pT1/M)*(np.power(pT1,2)-pT2)
    + 3*(M-1)*(M-2)*pT1*np.mean(pT1/M) - M*(M-1)*(M-2)*np.power(np.mean(pT1/M),3))
    dom = np.power(pT1,3) - 3*pT1*pT2 + 2*pT3
    return np.mean(num/dom)

def pt_process(low_cut,high_cut,info_list):
    mult_array = np.array([info[0] for info in info_list])
    ptspectra_array = np.array([info[1] for info in info_list])
    print(mult_array)
    print(ptspectra_array)
    print(np.shape(ptspectra_array))
    needed_ptspectra_array = ptspectra_array[np.where((mult_array>low_cut)&(mult_array<high_cut))]
    print(needed_ptspectra_array)
    print(np.shape(needed_ptspectra_array))
    return np.mean(needed_ptspectra_array,axis=0)

def vn4_2sub(Qn_a, Qn_b, Q2n_a, Q2n_b, M_a, M_b):
    '''
    This function is used to calculate v_n{4} with 2 sub-events
    return v_n4
    '''
    four_parts_correlator = ((((Qn_a*Qn_a - Q2n_a)*(np.conjugate(Qn_b*Qn_b - Q2n_b))).real).sum()
                             /(M_a*M_b).sum())
    two_parts_correlator = (((Qn_a*np.conjugate(Qn_b)).real).sum()/(M_a*M_b).sum())
    cn4_2sub = four_parts_correlator - 2*np.power(two_parts_correlator,2)
    return np.power(-cn4_2sub,1/4)


def cov_vn2_deltapt2(qn_sub1,M_sub1,pT1_sub2,pT2_sub2,M_sub2):
    '''
    4-particle correlation cov(v_{n}^2,delta([pT])^2)
    '''
    vn2_sub1 = ((qn_sub1*np.conjugate(qn_sub1)).real-M_sub1)/(M_sub1*(M_sub1-1))
    deltapTsq_sub2 = ((pT1_sub2*pT1_sub2 - pT2_sub2
                    - 2*(M_sub2-1)*pT1_sub2*np.mean(pT1_sub2/M_sub2)
                    + M_sub2*(M_sub2-1)*np.mean(pT1_sub2/M_sub2)*np.mean(pT1_sub2/M_sub2))
                    /(M_sub2*(M_sub2-1)))
    return np.mean(vn2_sub1*deltapTsq_sub2)-np.mean(vn2_sub1)*np.mean(deltapTsq_sub2)


def cn6(qn_std,q2n_std,q3n_std,M_std):
    '''
    return cn6
    '''
    six_parts_correlation = ((tools.SixPartcorre(qn_std, q2n_std, q3n_std, M_std)
    *M_std*(M_std-1)*(M_std-2)*(M_std-3)*(M_std-4)*(M_std-5)).sum()
    /(M_std*(M_std-1)*(M_std-2)*(M_std-3)*(M_std-4)*(M_std-5)).sum())

    four_parts_correlation = ((tools.FourPartCorr_N_N_MinN_MinN(qn_std,q2n_std,M_std)*
                    (M_std)*(M_std-1)*(M_std-2)*(M_std-3)).sum()/
                    (((M_std)*(M_std-1)*(M_std-2)*(M_std-3)).sum()))
    
    two_parts_correlation = ((tools.TwoPartCorr_N_MinN(qn_std, M_std)*(M_std)*(M_std-1)).sum()/
                ((M_std)*(M_std-1)).sum())
    
    return (six_parts_correlation - 9*two_parts_correlation*four_parts_correlation
            + 12*np.power(two_parts_correlation,3))
