import os, glob, sys, time
import numpy as np 

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
    folders = [folder for folder in glob.glob(data_folder+os.sep+'*') if os.path.basename(folder)[0]==feature] # get folders start with 'H'
    folders = list(np.array(folders)[np.argsort(list(map(get_hybe,folders)))])
    if len(folders) > 0:
        fovs = sorted(list(map(os.path.basename,glob.glob(folders[0]+os.sep+'*.dax'))),key=lambda l:int(l.split('.dax')[0].split('_')[-1]))
    else:
        raise IOError("No folders detected!")
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

