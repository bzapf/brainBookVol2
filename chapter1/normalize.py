import os
import pathlib
import argparse
import numpy
import nibabel

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputfolder", required=True, type=str, 
        help="path to folder in which the registered images are stored.")
    parser.add_argument("--exportfolder", required=True, type=str, 
        help="path to folder in which the normalized images will be stored. Will be created if it does not exist.")
    parser.add_argument('--refroi', required=True, type=str, 
        help="binary mask given as .mgz file that defines the ROI for normalization.")
    parserargs = parser.parse_args()
    parserargs = vars(parserargs)
    
    inputfolder = pathlib.Path(parserargs["inputfolder"])
    assert inputfolder.is_dir()
    exportfolder = pathlib.Path(parserargs["exportfolder"])
    assert inputfolder != exportfolder

    os.makedirs(exportfolder, exist_ok=True)

    refroi = nibabel.load(parserargs["refroi"])
    refroi_affine = refroi.affine
    refroi = refroi.get_fdata().astype(bool)
    
    images = sorted([f for f in os.listdir(parserargs["inputfolder"]) if f.endswith(".mgz")])

    for imagename in images:

        imagepath = inputfolder / imagename
        
        image = nibabel.load(imagepath)
        
        normalized_image = image.get_fdata() / numpy.median(image[refroi])

        print(refroi_affine)
        if numpy.sum(refroi_affine - image.affine) > 1e-16:
            raise ValueError("Affine transformations differ, are you sure the images are registered properly?")

        nibabel.save(nibabel.Nifti1Image(normalized_image, refroi_affine), exportfolder / imagename)

        print("Normalized", imagepath, "stored to", exportfolder / imagename)