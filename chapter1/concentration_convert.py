import numpy as np
import pickle

MAX_T1_IN_SECONDS = 10.
IGNORE_NUMPY_DIVISION_WARNING = True
USE_SPLINES = False


def check_T1(T1, mask):

    if mask is not None:
        assert np.max(T1 * mask) < MAX_T1_IN_SECONDS


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
                print("Throwing out ", n, "/", arr.size, "points below threshold")
        
        arr = np.where(arr < minval, minval, arr)

    if maxval is not None:
        if print_message:
            n = np.sum(np.where(arr > maxval, 1, 0))
            if n > 0:
                print("Throwing out ", n, "/", arr.size, "points above threshold")
        
        arr = np.where(arr > maxval, maxval, arr)
    
    if type(arr) == np.ndarray:
        arr[mask] = np.nan


    return arr


def plot_f():
    import matplotlib.pyplot as plt

    t = np.linspace(0.1, 6, 200)

    USE_SPLINES = False

    plt.plot(t, f(t))
    plt.show()


def f(T1, A=0.95089378, b=1.48087682, Tmin=None, Tmax=None):
    """Load and evaluate fit or spline interpolation of f,
    the function f in the Supplementary Material to
    Lars Magnus Valnes et al. "Apparent diffusion coefficient [...]" Sci. Rep. 2020
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

    check_T1(T1, mask=None)

    if USE_SPLINES:
        global f_splines
        try:
            retval = f_splines(T1)
        except NameError:
            print("Loaded f_splines from pickle file")
            # dirty workaround: 
            # Use __file__ to to get install dir of cats  
            with open(__file__[:-33] + 'constants/f_splines.pkl', 'rb') as input:
                f_splines = pickle.load(input)
                # assert T1.max() < 4.8 and T1.min() > 0.2, "spline interpolation invalid outside"
            retval = f_splines(T1)
        return retval

    else:
        retval = A * np.exp(- b * T1)
        return retval


def f_inverse(f_val, A=0.95089378, b=1.48087682, Tmin=None, Tmax=None):
    """Load and evaluate fit or spline interpolation of inverse of f
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

    if USE_SPLINES:
        global f_inverse_splines
        try:
            retval = f_inverse_splines(f_val)
        except NameError:
            # dirty workaround: 
            # Use __file__ to to get install dir of cats 
            print("Loaded f_splines from pickle file")
            with open(__file__[:-33] + 'constants/f_inverse_splines.pkl', 'rb') as input:
                f_inverse_splines = pickle.load(input)
            retval = f_inverse_splines(f_val)
        return retval
    else:

        t1 = np.log(f_val / A) / (- b)

        # breakpoint()

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

    # assert np.min(T_1_t[~np.isnan(T_1_t)]) >= 0

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

    check_T1(t1_map, mask)

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
    # try:

    #     assert np.min(T_1_t[~np.isnan(T_1_t)]) >= 0
    # except AssertionError as e:
    #     print(np.min(T_1_t[~np.isnan(T_1_t)]))
    #     raise e

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


def mris_to_concentrations(mri_list, t1_map, T_min=0.2, T_max=4.5, mask=None,
                            use_splines=False, ignore_numpy_warning=True):
    """Process a list of raw MRI.
    Assumes that the first item of <mri_list> contains a baseline MRI (without tracer):
    mri_list[0]["rel_time"] == 0.

    Args:
        mri_list (dict): list of dicts containing items
                        "rel_time" (time rel. to first MRI),
                        "raw_mri",
                        "ref_value" (signal in ref region to normalize MRI with,
                        "filename" (to keep track of filenames for export of concentration later)
        t1_map (np.array): t1 map (256x256x256 float array)
        T_min (float, optional): Lower threshold for "allowed" T1 values.
                        Defaults to T_1_min = 0.2 seconds as in Lars' work
        T_max (float, optional): Upper threshold for "allowed" T1 values.
                        Defaults to 4.5 seconds like in Lars' work
        use_splines (bool, optional): Whether to use spline interpolatation of f and f_inverse instead of fit
        ignore_numpy_warning (bool, optional): print numpy division by 0 warning
        mask (np.array): A mask in which the values are checked to be < MAX_T1_IN_SECONDS

    Returns:
        list: list of dicts containing items
        "rel_time"
        "concentration" (256x256x256 float array)
        "filename"
    """
    global USE_SPLINES
    USE_SPLINES = use_splines
    global IGNORE_NUMPY_DIVISION_WARNING
    IGNORE_NUMPY_DIVISION_WARNING = ignore_numpy_warning
    if IGNORE_NUMPY_DIVISION_WARNING:
        print("Ignoring numpy division by 0 warning")
        np.seterr(divide='ignore', invalid='ignore')
    
    if t1_map.max() > MAX_T1_IN_SECONDS:
        print("---------------------------------------------------------")
        print("Assuming T1 map is in milliseconds, converting to seconds")
        print("---------------------------------------------------------")
        t1_map *= 1e-3    

    if np.min(t1_map) < 0.:
        print("---------------------------------------------------------")
        print("WARNING: T1 map contains negative values, thresholding")
        print("---------------------------------------------------------")
        t1_map = np.where(t1_map < 0, 0, t1_map)

    check_T1(t1_map, mask=mask)    

    assert mri_list[0]["rel_time"] == 0., "baseline mri (at t=0) missing"
    
    mri_0 = mri_list[0]["raw_mri"]
    ref_val_0 = mri_list[0]["ref_value"]
    
    S_0 = mri_0 / ref_val_0
    
    concentration_images = []

    for mri in mri_list:

        if mask is not None:
            maskmri = mri["raw_mri"][mask]
            if maskmri.min() == 0:
                print(np.where(maskmri == 0, 1, 0).sum(), "/",
                    maskmri.size, "voxels of T1-weighted MRI in masked region are 0")

        # normalize mri by value in reference region
        S = mri["raw_mri"] / mri["ref_value"]

        # Check for "unphysical" signal ratios: The signal should not decrease
        if mask is not None:
            num_violations = np.sum(np.where((S / S_0)[mask] < 0, 1, 0))
            if num_violations > 0:
                print("Thresholding", num_violations, "voxels in mask with SIR < 0 in", mri["filename"])
        
        # throw out "unphysical" relative signal increases (S/S_0 < 1)
        S = np.where(S / S_0 - 1 <= 0, 0, S)

        c = concentration_signal_relation(S_t=S, S_0=S_0, mask=mask,
                            t1_map=t1_map, T_max=T_max, T_min=T_min)

        #if float(mri[0]) > 0:
        #    breakpoint()
        
        # store as list
        concentration_images.append({
            "rel_time": mri["rel_time"],
            "concentration": c,
            "filename": mri["filename"]}
        )

    return concentration_images


def mris_to_sir(mri_list, mask=None, ignore_numpy_warning=True):
    """Process a list of raw MRI.
    Assumes that the first item of <mri_list> contains a baseline MRI (without tracer):
    mri_list[0]["rel_time"] == 0.

    Args:
        mri_list (dict): list of dicts containing items
                        "rel_time" (time rel. to first MRI),
                        "raw_mri",
                        "ref_value" (signal in ref region to normalize MRI with,
                        "filename" (to keep track of filenames for export of concentration later)
        ignore_numpy_warning (bool, optional): print numpy division by 0 warning
        mask (np.array): A mask in which the values are checked to be < MAX_T1_IN_SECONDS

    Returns:
        list: list of dicts containing items
        "rel_time"
        "relative signal increase ratio" (256x256x256 float array)
        "filename"
    """

    global IGNORE_NUMPY_DIVISION_WARNING
    IGNORE_NUMPY_DIVISION_WARNING = ignore_numpy_warning
    if IGNORE_NUMPY_DIVISION_WARNING:
        print("Ignoring numpy division by 0 warning")
        np.seterr(divide='ignore', invalid='ignore')
    

    assert mri_list[0]["rel_time"] == 0., "baseline mri (at t=0) missing"

    mean_ref = np.mean([x["ref_value"] for x in mri_list])
        
    for idx, mri in enumerate(mri_list):
        mri_list[idx]["ref_value"] = mri["ref_value"] / mean_ref

    mri_0 = mri_list[0]["raw_mri"]
    ref_val_0 = mri_list[0]["ref_value"]
    
    S_0 = mri_0 / ref_val_0
    sir_images = []
    for mri in mri_list:

        if mask is not None:
            maskmri = mri["raw_mri"][mask]
            if maskmri.min() == 0:
                print(np.where(maskmri == 0, 1, 0).sum(), "/",
                    maskmri.size, "voxels of T1-weighted MRI in masked region are 0")

        # normalize mri by value in reference region
        S = mri["raw_mri"] / mri["ref_value"]

        # Check for "unphysical" signal ratios: The signal should not decrease
        if mask is not None:
            num_violations = np.sum(np.where((S / S_0)[mask] < 0, 1, 0))
            if num_violations > 0:
                print("Thresholding", num_violations, "voxels in mask with SIR < 0 in", mri["filename"])
        
        # throw out "unphysical" relative signal increases (S/S_0 < 1)
        S = np.where(S / S_0 - 1 <= 0, 0, S)

        sir = 100 * (S - S_0) / S_0
        
        
        # store as list
        sir_images.append({
            "rel_time": mri["rel_time"],
            "sir": sir,
            "filename": mri["filename"]}
        )

    return sir_images




if __name__ == "__main__":

    plot_f()