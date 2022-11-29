import numpy as np
import matplotlib.pyplot as plt


MAX_T1_IN_SECONDS = 10.
IGNORE_NUMPY_DIVISION_WARNING = True
A = 0.95089378
B = 1.48087682

def lookup(T1):

    n  = 116
    m  = 232 
    Tb = 5.199

    # OPTION 1
    #Ta = 853

    # OPTION 2
    Ta = 255. #- Tb*(n-1) => 853-Tb(n-1) = 255
    TR = 3000.
    Tw = TR -Ta - (m-1)*Tb

    #MO= 1 
    def Mn(T1):
        alpha = np.cos( np.deg2rad(8) ) 
        delta = np.exp( -Tw/T1)
        beta  = np.exp( -Tb/T1)
        gamma = np.exp( -Ta/T1)
        rho   = np.exp( -TR/T1)

        Meq =  -(1.-delta + delta*alpha*(1.-beta)*(1.-(alpha*beta)**(m-1) )/(1.-alpha*beta ) + delta*alpha*(alpha*beta)**(m-1) - rho*alpha**m )/ (1.+rho*alpha**m)

        return (1.-beta)*(1. - (alpha*beta)**(n-1) ) /(1.-alpha*beta)  + (alpha*beta)**(n-1)*(1.-gamma) + gamma*(alpha*beta)**(n-1)*Meq 


    SIs = Mn(T1)
    # plt.clf()
    return T1, SIs


def threshold(arr, minval=None, maxval=None, print_message=False):
    """Replace values higher/lower than maxval/minval with maxval/minval, respectively

    Args:
        arr (np.array): array to apply threshold to
        minval ([type], optional): [description]. Defaults to None.
        maxval ([type], optional): [description]. Defaults to None.
        print (bool, optional): Print info on how many numbers are below/above threshold. Defaults to False.

    Returns:
        np.array: thresholded array
    """

    if minval is None and maxval is None:
        return arr

    if type(arr) == np.ndarray:
        mask = np.where(np.isnan(arr), True, False)
        
        arr = np.nan_to_num(arr, nan=0)

    if minval is not None:
        if print_message:
            n = np.sum(np.where(arr < minval, 1, 0))
            if n > 0:
                print("Throwing out ", n, "/", arr.size, "points below lower threshold =", minval)
        
        arr = np.where(arr < minval, minval, arr)

    if maxval is not None:
        if print_message:
            n = np.sum(np.where(arr > maxval, 1, 0))
            if n > 0:
                print("Throwing out ", n, "/", arr.size, "points above upper threshold =", maxval)
        
        arr = np.where(arr > maxval, maxval, arr)
    
    if type(arr) == np.ndarray:
        arr[mask] = np.nan


    return arr


def f(T1, A=A, b=B, Tmin=None, Tmax=None):
    """Evaluate fit of f,
    the function f in the Supplementary Material to
    Valnes et al. "Apparent diffusion coefficient [...]" Sci. Rep. 2020
    Args:
        T1 (np.array): relaxation time T1 in seconds
        A (float, optional): Fit parameter A. Defaults to 0.95089378.
        b (float, optional): Fit parameter b. Defaults to 1.48087682.
    Returns:
        np.array: f
    """

    if T1 is None:
        return None

    T1 = threshold(T1, minval=Tmin, maxval=Tmax)


    retval = A * np.exp(- b * T1)
    
    return retval


def f_inverse(f_val, A=A, b=B, Tmin=None, Tmax=None):
    """evaluate fit of inverse of f
    Args:
        f (np.array): value of the function f
        A (float, optional): Fit parameter A. Defaults to 0.95089378.
        b (float, optional): Fit parameter b. Defaults to 1.48087682.
    Returns:
        np.array: T = f^(-1)(f)
    """

    # f is monotonously decreasing
    fmin = f(T1=Tmax)
    fmax = f(T1=Tmin)

    f_val = threshold(f_val, minval=fmin, maxval=fmax)

    t1 = np.log(f_val / A) / (- b)

    return t1

def concentration_T1_relation(T_1_0, T_1_t, r_1=3.2):
    """Compute c in equation (S3) in the Supplementary Material to
    Lars Magnus Valnes et al. "Apparent diffusion coefficient [...]" Sci. Rep. 2020
    
    Args:
        T_1_0 ([np.array]): T1 without tracer, can be given as a t1map of the brain, in seconds
        T_1_t ([np.array]): estimate or measured T1 with tracer, in seconds
        r_1 (float, optional): relaxivity of tracer in L mmol^{-1} s^{-1}. Defaults to 3.2, 
                                the value Gadobutrol ("Gadovist") at 3 T
                                see Rohrer et al. "Comparison of Magnetic Properties of MRI Contrast
                                Media Solutions at Different Magnetic Field Strengths"
                                Investigative Radiology 20
    Returns:
        np.array: concentration in mmol/L
    """

    return 1 / r_1 * (1 / T_1_t - 1 / T_1_0)


def estimate_T1(S_t, S_0, T_1_0, T_min=None, T_max=None):
    """Compute solution T_1^c (=T_1_t here) to equation (S13) in the Supplementary Material to
    Lars Magnus Valnes et al. "Apparent diffusion coefficient [...]" Sci. Rep. 2020

    Args:
        S_t (np.array): signal (normalized MRI) at time t
        S_0 (np.array): baseline (normalized MRI) at time t == 0
        T_1_0 ([np.array]): T1 without tracer, can be given as a t1map of the brain
        T_max ([None or float], optional): Lower threshold for "allowed" T1 values.. Defaults to None.
        T_min ([None or float], optional): Upper threshold for "allowed" T1 values.. Defaults to None.

    Returns:
        np.array: estimate for T1 given the signal with tracer and baseline
    """
    if IGNORE_NUMPY_DIVISION_WARNING:
        with np.errstate(divide='ignore'):
            f_t = (S_t / S_0) * f(T_1_0, Tmin=T_min, Tmax=T_max)
    else:
        f_t = (S_t / S_0) * f(T_1_0, Tmin=T_min, Tmax=T_max)


    assert np.min(f_t[~np.isnan(f_t)]) >= 0

    if T_max is not None:
        # Apply reverse treshold to ensure T_min <= T_1_t <= T_max
        f_t = threshold(f_t, minval=f(T_max), maxval=f(T_min))
        
    T_1_t = f_inverse(f_t, Tmin=T_min, Tmax=T_max)

    return T_1_t


def concentration_signal_relation(S_t, S_0, t1_map, mask, T_max=None, threshold_negatives=True,
                                    T_min=None, r_1=3.2):
    """ Combine equations (S3) and (S13) in the Supplementary Material to
    Lars Magnus Valnes et al. "Apparent diffusion coefficient [...]" Sci. Rep. 2020
    to compute c(S).

    Args:
        S_t (np.array): signal (normalized MRI) at time t
        S_0 (np.array): baseline (normalized MRI) at time t == 0
        t1_map ([type]): a t1map of the brain
        threshold_negatives  (bool): whether or not to replace negative values by 0
        T_max ([type], optional): Lower threshold for "allowed" T1 values. Defaults to None.
        T_min ([type], optional): Upper threshold for "allowed" T1 values. Defaults to None.
        r_1 (float, optional): relaxivity of tracer in L mmol^{-1} s^{-1}. Defaults to 3.2, 
                                the value Gadobutrol ("Gadovist") at 3 T
                                see Rohrer et al. "Comparison of Magnetic Properties of MRI Contrast
                                Media Solutions at Different Magnetic Field Strengths"
                                Investigative Radiology 20

    Returns:
        [np.array]: c in units mmol/L

    """

    if T_max is not None:
        assert T_min is not None
        # Apply thresholds before feeding T_1 values to f
        T_1_0 = threshold(t1_map, minval=T_min, maxval=T_max)
    else:
        T_1_0 = t1_map
        print("Applying no threshold to T1")

    assert np.min(T_1_0[~np.isnan(T_1_0)]) >= 0

    SIR = S_t / S_0
    assert np.min(SIR[~np.isnan(SIR)]) >= 0    

    T_1_t = estimate_T1(S_t=S_t, S_0=S_0, T_1_0=T_1_0, T_min=T_min, T_max=T_max)

    c = concentration_T1_relation(T_1_0=T_1_0, T_1_t=T_1_t, r_1=r_1)

    if threshold_negatives:

        c_ = c
        num_violations = np.where(c_ < 0, 1, 0).sum()
        
        if num_violations > 0:
            # Throw out negative concentrations
            c = np.where(c > 0, c, 0)
            
            if mask is not None:
                num_violations = np.where(c_[mask] < 0, 1, 0).sum()
                print("Thresholding negative c to 0 in", num_violations, "/", np.sum(mask), 
                        "voxels in mask.")
            else:
                print("Thresholding negative c to 0 in", num_violations, "voxels.")
        # The unit of c is mmol/L

    return c




if __name__ == "__main__":

    import matplotlib.pyplot as plt


    for Npoints in [1024, 4048, 4048 * 2, 4048 * 4]:

        t = np.linspace(0.2, 4, Npoints) * 1e3

        t, f_val = lookup(t)

        f_val_fit=f(t / 1e3)

        print("rel. error between fit and f", np.mean((f_val - f_val_fit) ** 2) / np.mean((f_val) ** 2))

    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 24)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 24)
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=6)

    
    plt.plot(t, f_val, label="MPRAGE")
    plt.plot(t, f_val_fit, color="red", label="fit")


    plt.xlabel(" T1 [ms]",fontsize=20)
    plt.ylabel(" f(T1) [1] ", rotation=0,x=-0.0, y=1.1,fontsize=28) 
    plt.tight_layout()
    plt.fill_betweenx(f_val, 800,1200 ,facecolor='white',alpha=0.2)
    plt.fill_betweenx(f_val, 1600,2000 ,facecolor='grey',alpha=0.2)
    plt.fill_betweenx(f_val, 2900,4000 ,facecolor='b',alpha=0.2)

    # plt.savefig("T1function.png")

    plt.legend()
    plt.show()