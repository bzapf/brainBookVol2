import datetime
import numpy as np
from pathlib import Path
import nibabel



def export(concentrations, exportfolder, 
        npy=False, mgz=False, vox2ras=None, vti=False, filename="original"):
    """Export concentrations to exportfolder, in different formats (npy, vti and mgz)

    Args:
        concentrations list: list of dicts containing items
                "rel_time"
                "concentration" (256x256x256 float array)
                "filename"
        exportfolder (string): path to a folder to save the results in.
        npy (bool): if concentration should be saved as npy array. Defaults to False.
        mgz (bool): if concentration should be saved as mgz. Needs vox2ras. Defaults to False.
        vox2ras (np.array): 4x4 np.array, probably some sort of rotation/translation ? 
                Needed to make a .mgz file from .npy data.
                Ask Lars Magnus if curious.
        vti (bool): if concentration should be saved as vti. Defaults to False.
        filename (str, optional):  Specify the style of the filename.
                "float" or "original".
                "float" will save files as 00.00.npy, 03.14.npy etc, while
                "original" will use the same file name as the raw MRI had
                Defaults to "original".

    Raises:
        ValueError: if filename is not "float" or "original"
    """
    
    for c_dict in concentrations:
        
        if filename == "float":
            t = c_dict["rel_time"]
            c_file_name = float_to_filestring(t)
        elif filename == "original":
            c_file_name = str(Path(c_dict["filename"]).stem) 
        elif filename == "_concentration":
            c_file_name = date_in_filename(str(Path(c_dict["filename"]).stem))[0] + filename

            # date_in_filename(c_dict["filename"])
        else:
            raise ValueError("unknown filename specified")
        
        c = c_dict["concentration"]

        assert c.dtype == np.float32

        if npy:
            np.save(exportfolder + c_file_name + ".npy", c)
        if mgz:
            assert vox2ras is not None
            c_img = nibabel.MGHImage(c, vox2ras)
            nibabel.save(c_img, exportfolder + c_file_name + ".mgz")
        if vti:
            raise NotImplementedError()
            pyevtk.hl.imageToVTK(exportfolder + c_file_name, cellData={'array': np.asarray(c, dtype=float)})

    print("Exported ", len(concentrations), "concentrations to ", exportfolder)




def float_to_filestring(t):
    """format t to have two digits before and two digits after comma

    Args:
        t ([type]): [description]

    Returns:
        string: formated time
    """
    if type(t) == str:
        t = float(t)
    t = format(t, ".2f")
    if len(t) < 5:
        t = "0" + t
    return t


def subcortex_gm(voxeldata):
    # functionality of mri_binarize --subcort-gm
    # provided by Lars Magnus Valnes
    mask = np.zeros(voxeldata.shape)
    for i in [10, 11, 12, 13, 17, 18, 26, 28, 27, 8, 49, 50, 52, 53, 54, 58, 60, 59, 47, 16]:
        mask = mask + (voxeldata == i)
    return mask


def date_in_filename(string):

    # found = False
    if "." in string:
        assert "mgz" in string or ".nii" in string or ".npy" in string

    
    # Automatically find the time in the filename:
    for time_format in ["%Y%m%d_%H%M%S", "%Y%m%d-%H%M%S"]:
        for idx in range(30):
            def filter_fun(x, idx):
                return x[-15-idx:-idx]

            try:
                # print(filter_fun(files[0]))
                datetime_ = datetime.datetime.strptime(filter_fun(string, idx), time_format)
                return filter_fun(string, idx), datetime_
                #found = True
            except ValueError:
                pass
            # if found:
            #     break
        # if found:
        #     break
    raise ValueError("No date found in filename")
    
    


def get_time_in_hours(files):
    """
    Given a list of filenames, get a list of times relative to first timepoint in the filenames
    provided by Lars Magnus Valnes

    Args:
        files ([type]): [description]

    Returns:
        [type]: [description]
    """



    files = sorted(files)
    
    datetimes = [date_in_filename(j)[1] for j in files]
    temp = [j - datetimes[0] for j in datetimes]
    # print(datetimes)
    hours = [j.days * 24 + j.seconds / 3600 for j in temp]
    return hours



