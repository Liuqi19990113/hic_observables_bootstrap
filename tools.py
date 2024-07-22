import numpy as np


def Qn(phi, n):
    '''
    return Qn = sum(exp(in*theta_i))
    '''
    phi = (np.array(phi)).astype(complex)
    Qn_res = (np.exp(1j*n*phi)).sum()
    return Qn_res


def pTm(pT, m):
    '''
    return pTm = sum((pT_i)^m)
    '''
    pT = np.array(pT)
    pTm_res = (np.power(pT,m)).sum()
    return pTm_res


def QnpTm(phi, pT, n, m):
    '''
    return QnpTm = sum(exp(in*theta_i)*(pT_i)^m)
    '''
    phi = (np.array(phi)).astype(complex)
    pT = np.array(pT)
    QnpTm_res = (np.exp(1j*n*phi)*np.power(pT,m)).sum()
    return QnpTm_res


def TwoPartCorr_N_MinN_Sub(Qn_a, Qn_b, M_a, M_b):
    '''
    Qa, Qb is sub event Q-vector
    return <2>_{a|b}
    '''
    return (Qn_a*np.conjugate(Qn_b)).real/(M_a*M_b)


def TwoPartCorr_N_MinN(Qn, M_std):
    '''
    Q_n is Q-vector
    return <2>_{n,-n}
    '''
    return ((Qn*np.conjugate(Qn)).real - M_std)/(M_std*(M_std-1))


def ThreePartCorr(Qn,Qm,Q_nminm, M):
    '''
    return <3>_{n,-m,m-n}
    '''
    weight = M*(M-1)*(M-2)
    Qn_sq = (Qn*np.conjugate(Qn)).real
    Qm_sq = (Qm*np.conjugate(Qm)).real
    Q_nminm_sq = (Q_nminm*np.conjugate(Q_nminm)).real
    res = ((Qn*np.conjugate(Qm)*np.conjugate(Q_nminm)).real 
            -Qn_sq-Qm_sq-Q_nminm_sq+2*M)
    return res/weight


def FourPartCorr_M_N_MinM_MinN(Qm, Qn, Q_mplusn, Q_mminn, M):
    '''
    return <4>_{n,-n,m,-m}
    '''
    weight = M*(M-1)*(M-2)*(M-3)
    Qm_sq = (Qm*np.conjugate(Qm)).real
    Qn_sq = (Qn*np.conjugate(Qn)).real
    res = (Qm_sq*Qn_sq + (4-M)*Qm_sq + (4-M)*Qn_sq
            + (Q_mminn*np.conjugate(Q_mminn)).real
            + (Q_mplusn*np.conjugate(Q_mplusn)).real
            - 2*(np.conjugate(Qn)*np.conjugate(Qm)*Q_mplusn).real
            - 2*(Qn*np.conjugate(Qm)*Q_mminn).real
            + M*M - 6*M)
    return res/weight


def FourPartCorr_N_N_MinN_MinN(Qn, Q2n, M_std):
    '''
    return <4>_{n,-n,n,-n}
    '''
    part_1 = (np.power((Qn*np.conjugate(Qn)).real,2) + (Q2n*np.conjugate(Q2n)).real
    - 2*(Q2n*np.conjugate(Qn)*np.conjugate(Qn)).real)
    part_2 = 4*(M_std-2)*(Qn*np.conjugate(Qn)).real - 2*M_std*(M_std-3)
    weight = M_std*(M_std-1)*(M_std-2)*(M_std-3)
    return (part_1-part_2)/weight


def cov_vn2_pt(Q_std, pT1, QpT_nm, M_std, mean_pT):
    '''
    3-particle correlation cov(v_{n}^2,[pT])
    '''
    weight = M_std*(M_std-1)*(M_std-2)
    corre = ((Q_std*np.conjugate(Q_std)).real*pT1 + (2-M_std)*pT1 -
            2*(QpT_nm*np.conjugate(Q_std)).real +
            (2-M_std)*mean_pT*((Q_std*np.conjugate(Q_std)).real-M_std))
    return corre/weight


def deltapT_sq(pT1, pT2, mean_pT, M):
    weight = M*(M-1)
    res = pT1*pT1-pT2-2*(M-1)*pT1*mean_pT+M*(M-1)*mean_pT*mean_pT
    return res/weight


def SixPartcorre(qn_std, q2n_std, q3n_std, M_std):
    res = (np.power((qn_std*np.conjugate(qn_std)).real,3)
    +9*((q2n_std*np.conjugate(q2n_std)).real)*((qn_std*np.conjugate(qn_std)).real)
    -6*(q2n_std*qn_std*np.conjugate(qn_std)*np.conjugate(qn_std)*np.conjugate(qn_std)).real
    +4*(q3n_std*np.conjugate(qn_std)*np.conjugate(qn_std)*np.conjugate(qn_std)).real
    -12*(q3n_std*np.conjugate(q2n_std)*np.conjugate(qn_std)).real
    +18*(M_std-4)*(q2n_std*np.conjugate(qn_std)*np.conjugate(qn_std)).real
    +4*(q3n_std*np.conjugate(q3n_std)).real
    -9*(M_std-4)*((qn_std*np.conjugate(qn_std)).real)*((qn_std*np.conjugate(qn_std)).real)
    -9*(M_std-4)*(q2n_std*np.conjugate(q2n_std)).real
    +18*(M_std-2)*((qn_std*np.conjugate(qn_std)).real)
    -6*M_std*(M_std-4))
    weight = M_std*(M_std-1)*(M_std-2)*(M_std-3)*(M_std-4)*(M_std-5)
    return res/weight
