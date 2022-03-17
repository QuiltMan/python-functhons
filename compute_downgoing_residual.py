def compute_downgoing_residual(residual, dobs, dsyn):

    dsyn_np=np.zeros((len(dsyn.data),len(dsyn.data[0])))
    dobs_np=np.zeros((len(dobs.data),len(dobs.data[0])))
    
    dsyn_np=np.array(dsyn.data[:])
    dobs_np=np.array(dobs.data[:])
    
    ######PARAMETERS#########
    nx=101;
    nt=573;
    dx=10;
    dt=1.75e-3;
    v0=2000
    
    x = dx*(np.linspace(0, nx-1, nx))
    xs=(nx/2)*dx;
    xoff=x-xs;

    pmin=-2/v0
    pmax=-pmin
    dp=.25*(pmax-pmin)/nx

    p_num=405
    
    tm = np.linspace(0, 1.00100000, 573)
    ######PARAMETERS#########

    dsyn_stp, tau, p=tptran(np.transpose(dsyn_np),tm, xoff, pmin, pmax, dp, p_num)
    
    dobs_stp, tau, p=tptran(np.transpose(dobs_np),tm, xoff, pmin, pmax, dp, p_num)
    ######PARAMETERS#########
    xmin=xoff[0]
    xmax=xoff[len(xoff)-1]
    #dx=2*(xmax-xmin)/(len(p)-1);
    ######PARAMETERS#########

    for ppp in range(int(p_num/2)+1):
        dsyn_stp[ppp]=0
        dobs_stp[ppp]=0
    
    dsyn_transmitted, it, ix=itptran(dsyn_stp,tau, p, xmin, xmax, dx)   
    dobs_transmitted, it, ix=itptran(dobs_stp,tau, p, xmin, xmax, dx)
    
    residual_np=np.zeros((len(dobs_transmitted),len(dobs_transmitted[0])))
    
    residual_np=dsyn_transmitted-dobs_transmitted        

    residual_temp=np.transpose(residual_np)
    
    for xx in range(len(residual.data)):
        for yy in range(len(residual.data[0])):
            residual.data[xx][yy]=residual_temp[xx][yy]
    
    return dsyn_transmitted,dobs_transmitted
