""" 

"""

import os
import pathlib
import argparse
import numpy
import nibabel
from relations import concentration_signal_relation

T_MIN = 0.2 # units of s
T_MAX = 4.5 # units of s

RELAXIVITY_CONSTANT = 3.2 # units of L / (mmol * s) from 
# Rohrer et al. "Comparison of Magnetic Properties of MRI Contrast Media Solutions at Different Magnetic Field Strengths" Investigative Radiology 20

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputfolder", required=True, type=str, help="path to folder in which the normalized images are stored.")
    parser.add_argument("--exportfolder", required=True, type=str, help="path to folder in which the estimated concentraion. will be stored. Will be created if it does not exist.")
    parser.add_argument('--t1map', required=True, type=str)

    parser.add_argument('--mask', default=None, type=str, help="binary mask given as .mgz file, every voxel outside mask will be set to 0")

    parserargs = parser.parse_args()
    parserargs = vars(parserargs)
    
    inputfolder = pathlib.Path(parserargs["inputfolder"])
    exportfolder = pathlib.Path(parserargs["exportfolder"])
    
    os.makedirs(exportfolder, exist_ok=True)

    t1_map = nibabel.load(parserargs["t1map"])
    t1_map_affine = t1_map.affine
    t1_map = t1_map.get_fdata()

    if 1e2 < t1_map.max():
        print("Assuming T1 map values are in milliseconds, converting to seconds")
        t1_map *= 1e-3
    else:
        print("Assuming T1 Map values are in seconds")

    # We can only compute concentration in the voxels where we have T1 values. Hence everything else will be set to nan
    t1_mask = numpy.where(t1_map > 0, True, False)
    
    if parserargs["mask"] is not None:
        mask = nibabel.load(parserargs["mask"])
        mask_affine = mask.affine
    
        assert numpy.sum(t1_map.affine - mask_affine) < 1e-14, "Affine transformations differ, are you sure the images are registered properly?"

        mask = mask.get_fdata().astype(bool)
        
        t1_mask = mask * t1_mask
    
    images = sorted([f for f in os.listdir(parserargs["inputfolder"]) if f.endswith(".mgz")])

    baseline = inputfolder / images[0]

    print("Loading baseline image", baseline)

    baseline_img = nibabel.load(baseline)

    affine = baseline_img.affine

    assert numpy.sum(affine - t1_map_affine) < 1e-14, "Affine transformations differ, are you sure the images are registered properly?"

    baseline = baseline_img.get_fdata()

    for imagename in images:

        imagepath = inputfolder / imagename
        
        assert imagepath.is_file()

        print("Converting", imagepath)
        
        image = nibabel.load(imagepath)
        
        assert numpy.sum(affine - image.affine) < 1e-14, "Affine transformations differ, are you sure the images are registered properly?"
        
        image = image.get_fdata()

        # TODO FIXME
        # Is it necessary / good to pass mask here for check ? 

        concentration = concentration_signal_relation(S_t=image, S_0=baseline, t1_map=t1_map, mask=t1_mask, threshold_negatives=False,
                                                        T_max=T_MAX, T_min=T_MIN, r_1=RELAXIVITY_CONSTANT)

        if parserargs["mask"] is not None:
            concentration = concentration * mask

        assert exportfolder.is_dir()

        nibabel.save(nibabel.Nifti1Image(concentration, affine), exportfolder / imagename)