import numpy as np
from scipy.fftpack import fft,ifft

def fftrl(wave,t): 
    nt=len(t)
    fftfull = fft(wave, nt) 
    
    fnyq=1. / (2*(t[1]-t[0]))
    nf=int(np.floor(nt/2+1))
    df=2*fnyq/nt;
    f = df*(np.linspace(0, nf-1, nf))
    
    spec=fftfull[0:nf]
    
    return spec, f, nt


def fftrl2(wave,t): 

    trace_num=len(wave)
    sample_num = len(wave[0])

    nf=int(np.floor(sample_num/2+1))
    temp=np.zeros(sample_num)
    spec2=np.zeros((trace_num,nf),dtype = complex)

    for n in range(trace_num):
        temp=wave[n]
        spec, f ,nt= fftrl(temp,t)
        spec2[n]=spec

    return spec2, f, trace_num, sample_num


def ifftrl(spec, f, nt_o):

    n=len(spec)
    
    if(nt_o%2==0):
        spec_conj = np.flip(np.conj((np.delete(spec, n-1))))
    else:                                                                                                                                   
        spec_conj = np.flip(np.conj(spec))
    
    spec_full=np.append(spec,spec_conj)
    
    if(nt_o%2==0):
        temp=np.delete(spec_full, (n-1)*2)
    else:
        temp=np.delete(spec_full, (n-1)*2+1)
        
    new_wave=np.real(ifft(temp, len(temp)) )
    
    if(nt_o%2==0):
        nt=(n-1)*2
    else:
        nt=(n-1)*2+1
    
    df=f[1]-f[0]
    dt=1/(nt*df)
    t = dt*(np.linspace(0, nt-1, nt))
    
    return new_wave, t

def ifftrl2(spec, f, trace_num, sample_num):

    temp=np.zeros(sample_num)
    new_wave2=np.zeros((trace_num,nt))

    for n in range(trace_num):
        temp=spec2[n]
        new_wave, t=ifftrl(temp, f, nt)
        new_wave2[n]=new_wave

    return new_wave2, t

def tptran(wave,t, xoff, pmin, pmax, dp, p_num):
    p = np.linspace(pmin, pmax, p_num)
    
    trace_num = len(wave)
    sample_num = len(wave[0])

    #padding FFT
    nt2=int(2**(np.ceil(np.log2(sample_num))))
    if sample_num<nt2:
        padding=np.zeros((nt2-sample_num,trace_num),dtype = complex)
        wave_padding=np.c_[wave,padding.T]
        tau = (t[1]-t[0])*(np.linspace(0, nt2-1, nt2))
    
    spec2, f, trace_num, sample_num = fftrl2(wave_padding,tau)
    
    stp=np.zeros((p_num,nt2),dtype = complex)
    
    for loop_p in range(p_num):
        dtx=p[loop_p]*xoff

        shiftr=(np.zeros((len(f),nx),dtype = complex)+(1j))*2.0*np.pi

        for l in range(len(f)):
            for m in range(len(dtx)):
                shiftr[l][m]=np.exp(shiftr[l][m]*f[l]*dtx[m])

        trcf=np.sum(np.transpose(spec2)*shiftr,axis=1)

        trcf[len(f)-1]=np.real(trcf[len(f)-1])
        new_wave, t = ifftrl(trcf,f,nt2)
        stp[loop_p]=new_wave

    stp=stp/p_num

    
    return stp, tau
