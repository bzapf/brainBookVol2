import re
import nibabel as nb
import os
import numpy as np
from cats.helpers import subcortex_gm, get_time_in_hours, date_in_filename


def read_masks(folder, include_subcortex_gray=True):
    """ Read a patfolder an extract raw data, ref values and masks

    Args:
        folder (string): Path to patient folder
        include_subcortex_gray (bool, optional): Include subcortical gray matter to the "gray" mask.
                    Defaults to True.


    Returns:
        masks: dict with masks "gray", "white" "csf"
    """

    assert(os.path.exists("{}/mri/wmparc.mgz".format(folder)))
    assert(os.path.exists("{}/mri/r1mapping.mgz".format(folder)))
    # assert(os.path.exists("{}/mri/ribbon.mgz".format(folder)))
    # Not needed in the current script, ask Lars Magnus Valnes if curious
    # vox2ras = nb.load("{}/mri/r1mapping.mgz".format(folder)).header.get_vox2ras()

    # If you want a ROI focus on parenchyma, then this can be usefulle
    # ribbon = nb.load("{}/mri/ribbon.mgz".format(folder)).get_fdata()
    # ROI = ndimage.binary_dilation(ribbon,iterations=2)

    parc_data = nb.load("{}/mri/wmparc.mgz".format(folder)).get_fdata()

    r1map = nb.load("{}/mri/r1mapping.mgz".format(folder)).get_fdata()
    # The ref_roi is a region of interest outside the masks.
    # It should be used to compute
    # normalized T1 signal  = voxel_data/ref_value
    # where ref_value = median(voxeldata[ref_roi])
    
    # Fixing segmentation of subcortex.
    subcortex = subcortex_gm(parc_data).astype('int')

    assert np.max(subcortex) == 1., "subcortex is not binary?"

    mask_csf = (r1map == 1)
    mask_white = (r1map == 2) - subcortex
    mask_white = mask_white.astype(np.bool)

    mask_grey = (r1map == 3)

    if include_subcortex_gray:
        
        mask_grey += subcortex.astype(np.bool)
        print("Included subcortical gray matter")

    masks = {"csf": mask_csf,
             "gray": mask_grey,
             "white": mask_white
             }

    for key, item in masks.items():
        #masks[key] = item.astype(np.bool)
        assert item.dtype == np.bool, "mask should be bool"
        
    return masks


def read_gMRI_data(folder, t1orientation, ref_roi=None, refvals=None, conformpath="/CONFORM"):
    """ Read a patfolder an extract raw data, ref values and masks
    Function was provided by Lars M. Valnes

    Args:
        folder (string): Path to patient folder
        ref_roi: np.array: Bool reference ROI to estimate reference signal (for normalization purpose).
                    Defaults to None.

    Returns:
        Data: dict
        ref_roi: np.bool array for ref ROI mask (used to obtain normalization constant)
    """

    # Assertions Not all IDs can be used.
    if conformpath == "/CONFORM":
        assert "REGISTERED" not in os.listdir(folder)
    # assert("PatID" in folder)
   
    if ref_roi is None and refvals is None:
        assert(os.path.exists("{}/mri/ref_rois.mgz".format(folder)))
        refroifilepath = "{}/mri/ref_rois.mgz".format(folder)
        ref_roi = nb.load(refroifilepath).get_fdata()
        ref_roi = (ref_roi > 0)
        print("Loaded ref ROI from ", refroifilepath)

    if refvals is not None:
        ref_roi = None

    # Sort all files based on date
    gMRI_files = []
    files_ = os.listdir(("{}" + conformpath).format(folder))

    def criterion(x):
        retval = x.endswith(".mgz") and "template" not in x
        retval = retval and "20" in x
        return retval

    print("-----------------------------")
    print("Omitting the following files:")
    for x in files_:
        if not criterion(x):
            print(x)

    print("-----------------------------")

    files = [x for x in files_ if criterion(x)]

    for i in sorted(files, key=date_in_filename):
        if i.endswith(".mgz"):
        #if i.startswith("pat") and i.endswith(".mgz"):
            gMRI_files.append(("{}" + conformpath + "/{}").format(folder, i))

    assert len(gMRI_files) > 0
    if refvals is not None:
        assert len(gMRI_files) == len(refvals)
 
    # print(gMRI_files)
    hours = get_time_in_hours(gMRI_files)

    Data = []
    for idx, file in enumerate(gMRI_files):
        print("Loaded ", file + ", rel. t = ", format(hours[idx], ".2f"))

        voxeldata = nb.load(file)
        if t1orientation is not None:
            assert t1orientation == nb.aff2axcodes(voxeldata.affine)
        voxeldata = voxeldata.get_fdata().astype(np.float32)
        if refvals is None:
            ref_value = np.median(voxeldata[ref_roi])
        else:
            ref_value = refvals[idx]
        # print(ref_value)
        
        # There should be MRI signal in the reference ROI:
        assert np.abs(ref_value) > 1e-6

        Data.append({
            "rel_time": hours[idx],
            "raw_mri": voxeldata,
            "ref_value": ref_value,
            "filename": file})

    return Data, ref_roi
