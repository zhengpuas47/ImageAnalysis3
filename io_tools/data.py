import os, glob, sys, time
import numpy as np 
import re
import tifffile
# Import here to avoid making astropy mandatory for everybody.
try:
    from astropy.io import fits
except ImportError:
    pass



### This sub package is aiming to manage data folders ###

def get_hybe(folder):
    #this sorts by the region number Rx.
    try: return int(os.path.basename(folder).split('H')[-1].split('R')[0])
    except: return np.inf

def get_folders(data_folder, feature='H', verbose=True):
    '''Function to get all subfolders
    given data_folder directory and certain features
        feature: 'H' for hybes, 'B' for beads
    Inputs:
        data_folder: full directory for data_folder, str of path
        feature: 1 letter feature that is used to find folders, str (default: 'H')
        verbose: say something!, bool
    Outputs:
        folders: list of full directory of folders, list of strings
        fovs: list of field-of-view file basenames, list of strings
    '''
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"data_folder:{data_folder} not exist.")

    folders = [folder for folder in glob.glob(data_folder+os.sep+'*') if os.path.basename(folder)[:len(feature)]==feature] # get folders start with 'H'
    #for __name in sorted(__color_dic.keys(), key=lambda _v:int( re.split(r'^H([0-9]+)[RQBUGCMP](.*)', _v)[1] ) ):
    # try sort folder by hyb
    try:
        folders = list(sorted(folders, key=lambda _path:int(re.split(r'^H([0-9]+)[RQBUGCMP](.*)', os.path.basename(_path) )[1])  ) ) 
    except:
        pass
    
    if len(folders) > 0:
        fovs = sorted(list(map(os.path.basename,glob.glob(folders[0]+os.sep+'*.dax'))),key=lambda l:int(l.split('.dax')[0].split('_')[-1]))
    else:
        raise FileNotFoundError(f"No sub-folders detected in {data_folder}!")
    
    if verbose:
        print("Get Folder Names: (ia.get_img_info.get_folders)")
        print("- Number of folders:", len(folders))
        print("- Number of field of views:", len(fovs))
    return folders, fovs



def copy_fov_for_data(data_folder, target_parent_directory, 
                      fov_id, feature='H', include_analysis=True,
                      include_nondata_folders=False, 
                      overwrite=False, verbose=True):
    """Function to copy the entire field of fiew dataset to a new location"""
    ## check inputs
    from shutil import copyfile
    if not os.path.isdir(target_parent_directory):
        raise FileNotFoundError(f"target directory: {target_parent_directory} doesn't exist, exit.")

    ## get folders and field of views
    _folders, _fovs = get_folders(data_folder, feature=feature, verbose=verbose)
    _fov_name = _fovs[int(fov_id)]
    if verbose:
        print(f"- Copy file of view: {_fov_name} to folder: {target_parent_directory}:", end=' ')

    # create target folder if necessary
    target_folder = os.path.join(target_parent_directory, 
                                 os.path.basename(data_folder))
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # copy files
    for _fd in _folders:
        if verbose:
            print(f"{os.path.basename(_fd)}", end='\t')
        # find target sub folder and create if necessary
        _target_subfd = os.path.join(target_folder, os.path.basename(_fd))
        if not os.path.exists(_target_subfd):
            os.makedirs(_target_subfd)
        for _basename in os.listdir(_fd):
            if _fov_name.split('.dax')[0] in _basename:
                if not os.path.exists(os.path.join(_target_subfd, _basename)) or overwrite:
                    copyfile(os.path.join(_fd, _basename), 
                             os.path.join(_target_subfd, _basename), )   

    if verbose:
        print(f"done.")
    if include_analysis:
        if verbose:
            print(f"-- copy analysis information")
        old_analysis_folder = os.path.join(data_folder, 'Aanlysis')
        new_analysis_folder = os.path.join(target_folder, 'Analysis')
        # get all objects in analysis folder
        for _obj in os.listdir(old_analysis_folder):
            _filename = os.path.join(old_analysis_folder, _obj)
            # copy if it's a csv file (which is usually reference files)
            if os.path.isfile(_filename) and '.' in _filename:
                if _filename.split('.')[-1] == 'csv':
                    if not os.path.exists(os.path.join(new_analysis_folder, _obj)) or overwrite:
                        copyfile(_filename, os.path.join(new_analysis_folder, _obj))
            # copy if it is a directory and contain corresponding fov information
            elif os.path.isdir(_filename):
                for _basename in os.listdir(_filename):
                    if _fov_name.split('.dax')[0] in _basename:
                        if not os.path.exists(os.path.join(new_analysis_folder, _obj, _basename)) or overwrite:
                            copyfile(os.path.join(_filename, _basename),
                                    os.path.join(new_analysis_folder, _obj, _basename))
        
    return 

class Writer(object):
    
    def __init__(self, width = None, height = None, **kwds):
        super(Writer, self).__init__(**kwds)
        self.w = width
        self.h = height

    def frameToU16(self, frame):
        frame = frame.copy()
        frame[(frame < 0)] = 0
        frame[(frame > 65535)] = 65535

        return np.round(frame).astype(np.uint16)

        
class DaxWriter(Writer):

    def __init__(self, name, **kwds):
        super(DaxWriter, self).__init__(**kwds)
        
        self.name = name
        if len(os.path.dirname(name)) > 0:
            self.root_name = os.path.dirname(name) + "/" + os.path.splitext(os.path.basename(name))[0]
        else:
            self.root_name = os.path.splitext(os.path.basename(name))[0]
        self.fp = open(self.name, "wb")
        self.l = 0

    def addFrame(self, frame):
        frame = self.frameToU16(frame)

        if (self.w is None) or (self.h is None):
            [self.h, self.w] = frame.shape
        else:
            assert(self.h == frame.shape[0])
            assert(self.w == frame.shape[1])

        frame.tofile(self.fp)
        self.l += 1
        
    def close(self):
        self.fp.close()

        self.w = int(self.w)
        self.h = int(self.h)
        
        inf_fp = open(self.root_name + ".inf", "w")
        inf_fp.write("binning = 1 x 1\n")
        inf_fp.write("data type = 16 bit integers (binary, little endian)\n")
        inf_fp.write("frame dimensions = " + str(self.w) + " x " + str(self.h) + "\n")
        inf_fp.write("number of frames = " + str(self.l) + "\n")
        inf_fp.write("Lock Target = 0.0\n")
        if True:
            inf_fp.write("x_start = 1\n")
            inf_fp.write("x_end = " + str(self.w) + "\n")
            inf_fp.write("y_start = 1\n")
            inf_fp.write("y_end = " + str(self.h) + "\n")
        inf_fp.close()
