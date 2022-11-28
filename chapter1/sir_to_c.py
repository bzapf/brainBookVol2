""" 
Compute concentrations from <percentage-baseline-increase> 
as stored in /FIGURES/ by Vegards' script
"""

import os
import pathlib
import argparse
import numpy as np
import nibabel

# from pathlib import Path
from concentration_convert import concentration_signal_relation, MAX_T1_IN_SECONDS
from helpers import subcortex_gm



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputfolder", required=True, type=str)
    parser.add_argument("--exportfolder", required=True, type=str)
    parser.add_argument('--t1map', required=True, type=str)
    parser.add_argument('--refroi', required=True, type=str)
    parser.add_argument('--mask', type=str)

    parserargs = parser.parse_args()
    parserargs = vars(parserargs)
    
    exportfolder = pathlib.Path(parserargs["exportfolder"])
    
    # assert not os.path.isdir(exportfolder)
    os.makedirs(exportfolder, exist_ok=True)

    t1_map = nibabel.load(parserargs["t1map"]).get_fdata()

    refroi = nibabel.load(parserargs["refroi"]).get_fdata()
    
    # t1_map = np.where(~np.isnan(t1_map), t1_map, 0)
    # t1_voxels = (~np.isnan(t1_map)).sum()

    # r1path = patfolder + "mri/r1mapping.mgz"
    # r1image = nibabel.load(r1path)
    # r1map = r1image.get_fdata()
    # parc_data = nibabel.load(patfolder + "mri/wmparc.mgz").get_fdata()
    # subcortex = subcortex_gm(parc_data).astype('int')
    # mask_csf = (r1map == 1)
    # mask_white = (r1map == 2) - subcortex
    # mask_white = mask_white.astype(bool)
    # mask_grey = (r1map == 3)
    # mask = mask_white + mask_grey

    # # breakpoint()
    # figfolder = patfolder + "FIGURES/"
    # # throw out temp or error files
    # figures = sorted([f for f in os.listdir(figfolder) if "201" in f and "baseline" not in f])
    # baselinepath = figfolder + figures[0] # [f for f in  if "baseline" in f][0]
    # baseline_img = nibabel.load(baselinepath)
    # nibabel.save(nibabel.Nifti1Image(mask.astype(int), baseline_img.affine), "/home/basti/Desktop/wg.mgz")
    # breakpoint()

    if parserargs["mask"] is not None:
        mask = nibabel.load(parserargs["mask"]).get_fdata()
        mask = np.where(mask > 0, True, False)



    if 1e2 < t1_map.max():
        print("Assuming t1map is given in milliseconds, converting to seconds")
        t1_map *= 1e-3


    if (t1_map * mask).max() > MAX_T1_IN_SECONDS:

        raise NotImplementedError()
        print("--------------------------------------------------------")
        print("Warning: T1 Map contains", np.sum(np.where(t1_map * mask > MAX_T1_IN_SECONDS, 1, 0)),
                "voxels > ", MAX_T1_IN_SECONDS, ", will replace by median around")
        print("--------------------------------------------------------")

        t1 = t1_map * mask

        # nibabel.save(nibabel.Nifti1Image((t1).astype(int), baseline_img.affine), patfolder + "t1test.mgz")
        exit()


        idx = np.transpose((t1 > MAX_T1_IN_SECONDS).nonzero())
        # breakpoint()

        
        for i in range(idx.shape[0]):
            
            window = t1[(idx[i, 0] - 1):(idx[i, 0] + 2), 
                    (idx[i, 1] - 1):(idx[i, 1] + 2),
                    (idx[i, 2] - 1):(idx[i, 2] + 2)]
            
            med = np.median(window)
            
            # print("Problematic value=", t1_map[idx[i, 0], idx[i, 1], idx[i, 2]], 
            #     "Median =", med, "(standard dev =", np.std(window), ")")

            if not med < MAX_T1_IN_SECONDS:
                med = 1.
                print("---------------------------------------------------------------------")
                print("WARNING: Median around problematic voxel > 10 s, setting value to 1s")
                print("---------------------------------------------------------------------")

            t1_map[idx[i, 0], idx[i, 1], idx[i, 2]] = med
            # print(t1[idx[i, 0], idx[i, 1], idx[i, 2]])


    assert np.max(t1_map * mask) <= MAX_T1_IN_SECONDS

    T_min = 0.2
    T_max = 4.5


    # throw out temp or error files
    images = sorted([f for f in os.listdir(parserargs["inputfolder"]) if f.endswith(".mgz")])


    baseline = exportfolder / images[0]

    print("Loading baseline image", baseline)
    baseline_img = nibabel.load(baseline)
    baseline = baseline_img.get_fdata()

    exit()

    for f in figures:
        figure = figfolder + f
        print("Converting", figure)
        
        new = nibabel.load(figure).get_fdata()
        # "new" is computed by Vegard like
        # new = 100 * (S(t)-S(0))/S(0) for t>0
        # where S(t)  = MRI(t) / ref_val(t) (for MRI in /CONFORM/) is a T1-weighted image
        # --> S(t) = S(0)(1 + new/100)

        S_t = baseline * (1 + new / 100)

        mask = ~np.isnan(new)

        SIR = S_t / baseline

        assert np.min(SIR[mask]) >= 0

        print("max SIR", np.max(SIR[mask]), np.argmax(SIR[mask]))
        print("min SIR", np.min(SIR[mask]), np.argmin(SIR[mask]))


        c = concentration_signal_relation(S_t, S_0=baseline, t1_map=t1_map, mask=mask, threshold_negatives=False,
                            T_max=T_max, T_min=T_min, r_1=3.2)
        

        print(np.nanmean(c[c > - 1e12]))
        # c = np.nan_to_num(c)

        # print("min c", np.min(c[mask]), np.argmin(c[mask]))
        # print("max c", np.max(c[mask]), np.argmax(c[mask]))

        # assert -10 < np.min(np.nan_to_num(c))

        # mask = ~np.isnan(new)

        # mask = mask.astype(int)

        # c *= mask


        # conc = c
        # breakpoint()

        if not (parserargs["t1map"] == "wg_cohort_nothreshold"):
            # Throw out negative concentrations
            print("Thresholding negative concentration values to 0 in", 
                    format(100 * np.where(c < 0, 1, 0).sum() / t1_voxels, ".2f"), 
                    " % the voxels in t1map (white and gray matter).")
            c = np.where(c > 0, c, 0)
        else:
            print("negative concentration values in", 
                format(100 * np.where(c < 0, 1, 0).sum() / t1_voxels, ".2f"), 
                    " % the voxels in t1map (white and gray matter).")

            # store to exportfolder
        nibabel.save(nibabel.Nifti1Image(c, baseline_img.affine), 
                    exportfolder + f[:16] + "concentration.mgz")

    # for file in os.listdir(patfolder + "FIGURES_CONC/"):
    #     c1 = nibabel.load(patfolder + "FIGURES_CONC/" + file).get_fdata()
    #     c2 = nibabel.load(patfolder + "TESTFIGS/" + file).get_fdata()
    #     print(np.abs(c1 - c2).sum())
        