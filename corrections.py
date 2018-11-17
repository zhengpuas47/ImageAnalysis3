from . import get_img_info, visual_tools, analysis, classes
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy,_image_size
import numpy as np
import pickle
import matplotlib.pylab as plt
import os, glob, sys, time
from scipy.stats import scoreatpercentile
import multiprocessing as mp
import ctypes

def __init__():
    pass

def get_STD_beaddrift(bead_ims, bead_names, analysis_folder, fovs, fov_id,
                      illumination_correction=False, ic_channel=488,
                      correction_folder=_correction_folder,
                      repeat=True, plt_val=False, cutoff_=3, xyz_res_=1,
                      coord_sel=None, sz_ex=100, ref=0,
                      force=False, save=True, th_seed=150, dynamic=False, verbose=True):
    """Given a list of bead images This handles the fine bead drift correction.
    If save is true this requires global paramaters analysis_folder,fovs,fov_id
    Inputs:
        bead_ims: list of images, list of ndarray
        analysis_folder: full directory to store analysis files, string
        fovs: names for all field of views, list of strings
        fov_id: the id for field of view to be analysed, int
        """
    # define a sub function to do fitting
    from scipy.stats import scoreatpercentile

    # Initialize failed counts
    fail_count = 0

    # if save, check existing pickle file, if exist, don't repeat
    if save:
        save_cor = analysis_folder+os.sep+fovs[fov_id].replace('.dax','__current_cor.pkl')
        if os.path.exists(save_cor):
            total_drift = pickle.load(open(save_cor,'rb'))
            if len(list(total_drift.keys()))==len(bead_ims):
                repeat=False
    repeat = repeat or force

    # if do illumination_correction:
    if illumination_correction:
        _bead_ims = [Illumination_correction(_im,ic_channel, correction_folder=correction_folder, verbose=False) for _im in bead_ims]
    else:
        _bead_ims = bead_ims;


    # repeat if no fitted data exist
    if repeat:
        # choose reference image
        if ref is None: ref = 0
        im_ref = _bead_ims[ref]
        if verbose:
            print("Fitting reference:", ref)
        if coord_sel is None:
            coord_sel = np.array(im_ref.shape)/2
        coord_sel1 = np.array([0,-sz_ex,-sz_ex]) + coord_sel
        if dynamic:
            th_seed = scoreatpercentile(im_ref, 99) * 0.5
        im_ref_sm = visual_tools.grab_block(im_ref,coord_sel1,[sz_ex]*3)
        cents_ref1 = get_STD_centers(im_ref_sm, th_seed=th_seed)#list of fits of beads in the ref cube 1
        coord_sel2 = np.array([0,sz_ex,sz_ex]) + coord_sel
        im_ref_sm = visual_tools.grab_block(im_ref,coord_sel2,[sz_ex]*3)
        cents_ref2 = get_STD_centers(im_ref_sm, th_seed=th_seed)#list of fits of beads in the ref cube 2

        txyzs = []
        for iim,im in enumerate(_bead_ims):
            # if this frame is reference, continue
            if iim == ref:
                txyzs.append(np.array([0.,0.,0.]));
                continue;
            if dynamic:
                th_seed = scoreatpercentile(im,99)*0.5
            im_sm = visual_tools.grab_block(im,coord_sel1,[sz_ex]*3)
            cents1 = get_STD_centers(im_sm, th_seed=th_seed)#list of fits of beads in the cube 1
            im_sm = visual_tools.grab_block(im,coord_sel2,[sz_ex]*3)
            cents2 = get_STD_centers(im_sm, th_seed=th_seed)#list of fits of beads in the cube 2
            if verbose:
                print("Aligning "+str(iim))
            txyz1 = visual_tools.translation_aling_pts(cents_ref1,cents1,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)
            txyz2 = visual_tools.translation_aling_pts(cents_ref2,cents2,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)

            txyz = (txyz1+txyz2)/2.
            if np.sum(np.abs(txyz1-txyz2))>3:
                fail_count += 1; # count times of suspected failure
                print("Suspecting failure.")
                #sz_ex+=10
                coord_sel3 = np.array([0,sz_ex,-sz_ex])+coord_sel
                im_ref_sm = visual_tools.grab_block(im_ref,coord_sel3,[sz_ex]*3)
                cents_ref3 = get_STD_centers(im_ref_sm, th_seed=th_seed)#list of fits of beads in the ref cube 3
                im_sm = visual_tools.grab_block(im,coord_sel3,[sz_ex]*3)
                cents3 = get_STD_centers(im_sm, th_seed=th_seed)#list of fits of beads in the cube 3
                txyz3 = visual_tools.translation_aling_pts(cents_ref3,cents3,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)
                #print txyz1,txyz2,txyz3
                if np.sum(np.abs(txyz3-txyz1))<np.sum(np.abs(txyz3-txyz2)):
                    txyz = (txyz1+txyz3)/2.
                    print(txyz1,txyz3)
                else:
                    txyz = (txyz2+txyz3)/2.
                    print(txyz2,txyz3)

            txyzs.append(txyz)
        # convert to dic
        total_drift = {_name:_dft for _name, _dft in zip( bead_names, txyzs) }
        if save:
            save_cor = analysis_folder+os.sep+fovs[fov_id].replace('.dax','__current_cor.pkl')
            pickle.dump(total_drift,open(save_cor,'wb'))

    return total_drift, repeat, fail_count

## Fit bead centers

def get_STD_centers(im, th_seed=150, close_threshold=0.01, plt_val=False,
                    save=False, save_folder='', save_name='', force=False, verbose=False):
    '''Fit beads for one image:
    Inputs:
        im: image, ndarray
        th_seeds: threshold for seeding, float (default: 150)
        close_threshold: threshold for removing duplicates within a distance, float (default: 0.01)
        plt_val: whether making plot, bool (default: False)
        save: whether save fitting result, bool (default: False)
        save_folder: full path of save folder, str (default: None)
        save_name: full name of save file, str (default: None)
        force: whether force fitting despite of saved file, bool (default: False)
        verbose: say something!, bool (default: False)
    Outputs:
        beads: fitted spots with information, n by 4 array'''
    import os
    import pickle as pickle
    if not force and os.path.exists(save_folder+os.sep+save_name) and save_name != '':
        if verbose:
            print("- loading file:,", save_folder+os.sep+save_name)
        beads = pickle.load(open(save_folder+os.sep+save_name, 'rb'), encoding='latin1');
        if verbose:
            print("--", len(beads), " of beads loaded.")
        return beads
    else:
        # seeding
        seeds = visual_tools.get_seed_points_base(im, gfilt_size=0.75,filt_size=3,th_seed=th_seed,hot_pix_th=4)
        # fitting
        pfits = visual_tools.fit_seed_points_base_fast(im,seeds,width_z=1.8*1.5/2,width_xy=1.,radius_fit=5,n_max_iter=10,max_dist_th=0.25,quiet=not verbose)
        # get coordinates for fitted beads
        remove = 0
        if len(pfits) > 0:
            beads = pfits[:,1:4]
            # remove very close spots
            for i,bead in enumerate(beads):
                if np.sum(np.sum((beads-bead)**2, axis=1)<close_threshold) > 1:
                    beads = np.delete(beads, i-remove, 0)
                    remove += 1
        else:
            beads = None
        if verbose:
            print(remove, "points removed given smallest distance", close_threshold)
        # make plot if required
        if plt_val:
            plt.figure()
            plt.imshow(np.max(im,0),interpolation='nearest')
            plt.plot(beads[:,2],beads[:,1],'or')
            plt.show()
        # save to pickle if specified
        if save:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if verbose:
                print("-- saving beads to", save_folder+os.sep+save_name)
            pickle.dump(beads, open(save_folder+os.sep+save_name, 'wb'));

        return beads

def tempfile_beaddrift(filenames, bead_names=None, ref_id=0, drift_size=300, num_threads=12, 
                       im_size=_image_size, coord_sel=None, align_cutoff=3, align_res=1,
                       dynamic_seeding=True, dynamic_th_percent=80, seed_th=300, 
                       save=True, save_folder=None, save_postfix='_sequential_current_cor.pkl', overwrite=False, verbose=True):
    """Function to generate drift profile given a list of corrected bead files
    Inputs:
    filenames: filenames of temp-files or raw images, list of string(path)s
    bead_names: reference names(H1R1\Conv_Zscan_03.dax), used as reference key for drift_dic, list of strings(default:None, generated by filenames)
    ref_id: reference frame index, int (defualt: 0)
    drift_size: window size of drift image which is used for cropping, int (default: 300)
    num_threads: number of threads used in fitting beads, int (default: 12)
    coord_sel: coordinate to pick windows, array-like of 2 elements (default: None, which means center)
    align_cutoff: cutoff for trans-point alignment, int (default: 3)
    align_res: alignment resolution, int (default: 1)
    dynamic_seeding: whether use dynamic seeding threshold, bool (defualt: True)
    dynamic_th_percent: dynamic seeding threshold percentile, int (defualt: 80)
    seed_th: seeding threshold as default, which is used when dynamic=False or dynamic seeding failed, float (default: 300)
    save: whether save drift result dictionary to pickle file, bool (default: True)
    save_folder: folder to save drift result, required if save is True, string of path (default: None)
    save_postfix: drift file postfix, string (default: '_sequential_current_cor.pkl')
    overwrite: whether overwrite existing drift_dic, bool (default: False)
    verbose: say something during the process! bool (default:True)


    """
    ## check inputs
    # check filenames
    if not isinstance(filenames, list):
        raise ValueError("Wrong input type of filenames, should be a list")
        # check filename length
    if len(filenames) == 0:
        raise ValueError("Kwd filenames should have at least one element")
    for _fl in filenames:
        if not os.path.isfile(_fl):
            raise IOError(f"one of input file:{_fl} doesn't exist, exit!")
    # check bead_names
    if bead_names is not None:
        if not isinstance(bead_names, list) or len(filenames) != len(bead_names):
            raise ValueError("Wrong input type of bead_names, should be a list with matched length of filenames")
    else: #generate bead_names from filenames
        bead_channel = '488'
        temp_postfix = 'corrected'
        bead_names = []
        for _fl in filenames:
            _fn = os.path.basename(_fl)
            if bead_channel in _fn:
                _fn = _fn.split('_'+ bead_channel)[0]
            else:
                _fn = _fn.split('_'+ temp_postfix)[0]
            bead_names.append(_fn.replace('-',os.sep)+'.dax')
    # check ref-id
    if ref_id >= len(filenames) or ref_id <= -len(filenames):
        raise ValueError(f"Ref_id should be valid index of filenames, however {ref_id} is given")
    # check save-folder
    if save:
        if save_folder is None:
            raise ValueError("Kwd drift_folder is required when save is True")
        elif not os.path.exists(save_folder):
            if verbose:
                print(f"Create drift_folder:{save_folder}")
            os.makedirs(save_folder)
    # start loading existing profile
    _save_name = bead_names[ref_id].split(os.sep)[-1].replace('.dax', save_postfix)
    _save_filename = os.path.join(save_folder, _save_name)
    if not overwrite and os.path.isfile(_save_filename):
        if verbose:
            print(f"-- loading existing drift info from file:{_save_filename}")
        old_drift_dic = pickle.load(open(_save_filename, 'rb'))
        _key_exist = [_bead_name in old_drift_dic for _bead_name in bead_names]
        if sum(_key_exist) == len(bead_names):
            if verbose:
                print("-- All frames exists in original drift file, exit.")
            return old_drift_dic, 0
    else:
        if verbose:
            print(f"-- starting a new drift correction")
            print(bead_names[ref_id])
            print(f"{_save_name},{_save_filename}")
        old_drift_dic = {}
    ## initialize
    # record start time
    _start_time = time.time()
    # whether drift for this frame changes
    if len(old_drift_dic) > 0:
        old_ref_frames = [_hyb_name for _hyb_name,
                         _dft in old_drift_dic.items() if (np.array(_dft)==0).all()]
        if len(old_ref_frames) > 1 or len(old_ref_frames) == 0:
            print("Ref frame not unique, start over!")
            old_drift_dic = {}
        else:
            old_ref_frame = old_ref_frames[0]
            # if ref-frame not overlapping, remove the old one for now
            if old_ref_frame != bead_names[ref_id]:
                if verbose:
                    print(f"old-ref:{old_ref_frame}, delete old refs because ref doesn't match")
                del old_drift_dic[old_ref_frame]
    else:
        old_ref_frame = None
    ## collect reference image info
    if coord_sel is None:
        coord_sel = np.array([int(im_size[-2]/2), int(im_size[-1]/2)], dtype=np.int)
    # collect crop coordinates (slices)
    crop0 = np.array([[0,im_size[0]],
                      [max(coord_sel[-2]-drift_size,0), coord_sel[-2]],
                      [max(coord_sel[-1]-drift_size,0), coord_sel[-1]]],dtype=np.int)
    crop1 = np.array([[0,im_size[0]],
                      [coord_sel[-2], min(coord_sel[-2]+drift_size, im_size[-2])],
                      [coord_sel[-1], min(coord_sel[-1]+drift_size, im_size[-1])]], dtype=np.int)
    crop2 = np.array([[0, im_size[0]],
                      [coord_sel[-2], min(coord_sel[-2]+drift_size, im_size[-2])],
                      [max(coord_sel[-1]-drift_size, 0), coord_sel[-1]]],dtype=np.int)
    selected_crops = np.stack([crop0,crop1,crop2]) # merge into one array which is easier to feed into function
    # collect ref image
    ref_ims = [visual_tools.slice_image(filenames[ref_id],im_size,_crop[0],_crop[1],_crop[2],
                                        zstep=1,zstart=0) for _crop in selected_crops]
    if dynamic_seeding:
        ref_seed_ths = [scoreatpercentile(rim, dynamic_th_percent)*0.45 for rim in ref_ims]
    else:
        ref_seed_ths = [seed_th for rim in ref_ims]
    ref_centers = [get_STD_centers(rim, th_seed=rseed, verbose=verbose) for rim, rseed in zip(ref_ims, ref_seed_ths)]
    # retrieve files requiring drift correction
    args = []
    new_refnames = []
    for _fl, _ref_name in zip(filenames, bead_names):
        # append all new frames except ref-frame. ref-frame will be assigned to 0
        if _ref_name not in old_drift_dic and _ref_name != bead_names[ref_id]:
            new_refnames.append(_ref_name)
            args.append((_fl, selected_crops, ref_centers, im_size, align_cutoff, align_res,
                         dynamic_seeding, dynamic_th_percent, seed_th, verbose))
    ## multiprocessing 
    if verbose:
        print(f"- Start multi-processing drift correction with {num_threads} threads")
    with mp.Pool(num_threads) as drift_pool:
        align_results = drift_pool.starmap(_align_single_image, args)
        drift_pool.close()
        drift_pool.terminate()
        drift_pool.join()
    # clear 
    del(args)
    classes.killchild()
    new_drift_dic = {_ref_name:_ar[0] for _ref_name, _ar in zip(new_refnames, align_results)}
    fail_count = sum([_ar[1] for _ar in align_results])
    new_drift_dic[bead_names[ref_id]] = np.zeros(3) # for ref frame, assign to zero
    if old_ref_frame is not None and old_ref_frame in new_drift_dic:
        ref_drift = new_drift_dic[old_ref_frame]
        if verbose:
            print(f"-- drift of reference: {ref_drift}")
        # update drift in old ones
        for _old_name, _old_dft in old_drift_dic.items():
            if _old_name not in new_drift_dic:
                new_drift_dic[_old_name] = _old_dft + ref_drift
    ## save
    if save:
        if verbose:
            print(f"- Saving drift to file:{_save_filename}")
        pickle.dump(new_drift_dic, open(_save_filename, 'wb'))
    if verbose:
        print(f"-- Total time cost in drift correction: {time.time()-_start_time}")
        if fail_count > 0:
            print(f"-- number of failed drifts: {fail_count}")
    return new_drift_dic, fail_count


def _align_single_image(_filename, _selected_crops, _ref_centers, _im_size=_image_size,
                        _align_cutoff=3, _align_res=1, _dynamic_seeding=True,
                        _dynamic_th_percent=80, _seed_th=300, _verbose=True):
    """align single image used by multiprocessing-drift-correction
    Inputs:
        _filename: filename of a temp file"""
    _failed_count = 0
    _target_ims = [visual_tools.slice_image(_filename, _im_size,_crop[0],_crop[1],_crop[2],
                                        zstep=1,zstart=0) for _crop in _selected_crops[:2]]
    if _dynamic_seeding:
        _target_seed_ths = [scoreatpercentile(rim, _dynamic_th_percent)*0.45 for rim in _target_ims]
    else:
        _target_seed_ths = [_seed_th for rim in _target_ims]
    _target_centers = [get_STD_centers(rim, th_seed=rseed, verbose=_verbose) for rim, rseed in zip(_target_ims, _target_seed_ths)]
    if _verbose:
        print(f"- Aligning {os.path.basename(_filename)}")
    _aligns = [visual_tools.translation_aling_pts(_ref_cent, _tar_cent, cutoff=_align_cutoff,
        xyz_res=_align_res) for _ref_cent, _tar_cent in zip (_ref_centers[:2], _target_centers[:2])]
    _drift = np.mean(_aligns, 0)
    if np.linalg.norm(_aligns[0]-_aligns[1], 1) > 3:
        if _verbose:
            print(f"Suspecting failure for {os.path.basename(_filename)}")
        _failed_count += 1
        _target_ims.append(visual_tools.slice_image(_filename, _im_size, _selected_crops[2][0], 
                            _selected_crops[2][1],_selected_crops[2][2],zstep=1, zstart=0))
        if _dynamic_seeding:
            _target_seed_ths.append(scoreatpercentile(_target_ims[-1], _dynamic_th_percent)*0.45)
        else:
            _target_seed_ths.append(_seed_th)
        _target_centers.append(get_STD_centers(_target_ims[-1],th_seed=_target_seed_ths[-1],verbose=_verbose))
        _aligns.append(visual_tools.translation_aling_pts(_ref_centers[2], _target_centers[2], 
                                                           cutoff=_align_cutoff, xyz_res=_align_res))
        if np.linalg.norm(_aligns[0]-_aligns[2],1) <= np.linalg.norm(_aligns[1]-_aligns[2],1):
            _drift = (_aligns[0]+_aligns[2])/2
            if _verbose:
                print(f"-- selected drifts:{_aligns[0]}, {_aligns[2]}")
        else:
            _drift = (_aligns[1]+_aligns[2])/2
            if _verbose:
                print(f"-- selected drifts:{_aligns[1]}, {_aligns[2]}")

    return _drift, _failed_count


def STD_beaddrift_sequential(bead_ims, bead_names, drift_folder, fovs, fov_id,
                             drift_size=200, coord_sel=None, cutoff_=3, xyz_res_=1,
                             dynamic=False, dynamic_th_percent=80, th_seed=300,
                             illumination_correction=False, ic_channel='488', correction_folder=_correction_folder,
                             overwrite=False, plt_val=False, save=True, save_postfix='_sequential_current_cor.pkl',
                             verbose=True):
    """Given a list of bead images This handles the fine bead drift correction.
    If save is true this requires global paramaters drift_folder,fovs,fov_id
    Inputs:
        bead_ims: list of images, list of ndarray
        bead_names: names for bead files, list of strings
        drift_folder: full directory to store analysis files, string
        fovs: names for all field of views, list of strings
        fov_id: the id for field of view to be analysed, int
        drift_size: size of window used in drift correction, int (defualt: 200)
        coord_sel: whether select a coordinate as center, None or len-3 vector (default:None, means middle of image)
        cutoff_: point alignment cutoff, int (default: 3)
        xyz_res_: resolution in xyz during alignment, int (default: 1)
        dynamic: whether use dynamic seeding threashold, bool (default: False)
        dynamic_th_percent: if dynamic, the threshold is this percentile of total image intensity, float in [0,100] (default:80)
        th_seed: fixed seed threshold if not dynamic, float (default:150)
        Illumination_correction: whether do illumination correction, bool
        ic_channel: illumination_correction channel, str or int (default: 488)
        correction_folder: folder for correction files, string (path)
        overwrite: whether overwrite existing drifts, bool (default: False)
        plt_val: whether plot alignment result, bool (default: False)
        save: whether save drift to file, bool (default: True)
        save_postfix: postfix for save filename after field-of-view name, str(default:'_sequential_current_cor.pkl')
        verbose: Say something during the process! bool (default: True)
    Outputs:
        total_drift: dictionary from bead_names to len=3 vector representing drifts in z,x,y, dict
        _failed_count: number of suspicious failures during drifts, int
        """
    if verbose:
        print("- Drift correction starts!")
    # Initialize failed counts
    fail_count = 0
    # whether drift for this frame changes
    change_markers = {}
    # record start time
    _start_time = time.time()
    # check existing pickle file, if exist, don't repeat
    save_filename = os.path.join(drift_folder, fovs[fov_id].replace('.dax', save_postfix))
    # if savefile already exists, load
    if os.path.exists(save_filename):
        total_drift = pickle.load(open(save_filename,'rb'))
    else: # else create an empty dic
        total_drift = {}
    # if do illumination_correction:
    if illumination_correction:
        _bead_ims = [Illumination_correction(_im,ic_channel, correction_folder=correction_folder, verbose=False) for _im in bead_ims]
    else:
        _bead_ims = bead_ims

    # repeat if no fitted data exist
    # initialize drifts
    if len(_bead_ims) == 0:
        raise ValueError("Wrong dimension of _bead_ims, at least 1 image required")
    else:
        # initialize old_ref_frame
        old_ref_frame = ''
        txyzs = [np.array([0.,0.,0.])]; # initialize with zeros, representing image0 itself
        if len(total_drift) > 0 and bead_names[0] in total_drift:
            change_markers[bead_names[0]] = False # marker for whether made any changes
        else: # ref frame is not in total_drift dic: create a zero array
            if verbose:
                print("Ref frame changed")
            # old ref-frame
            for _hyb_name, _dft in list(total_drift.items()):
                if np.sum(_dft == 0 ) == len(_dft):
                    print('old-ref:',_hyb_name)
                    old_ref_frame = _hyb_name
                    del total_drift[_hyb_name]
            # new ref-frame
            change_markers[bead_names[0]] = True
            total_drift[bead_names[0]] = np.array([0,0,0])
            if verbose:
                print(f"-- Modifying existing refrence frame to {bead_names[0]}")

    # define selected coordinates in each field of view
    if coord_sel is None:
        coord_sel = np.array(_bead_ims[0].shape)/2
    coord_sel1 = np.array([0,-drift_size,-drift_size]) + coord_sel
    coord_sel2 = np.array([0,drift_size,drift_size]) + coord_sel

    # initialize ref image
    ref = 0; # initialize reference id
    im_ref = _bead_ims[ref]; # initialize reference image
    # dynamic seeding threhold
    if dynamic:
        if np.max(im_ref) < 30000:
            th_seed = scoreatpercentile(im_ref, dynamic_th_percent) * 0.5;
    # start fitting image 0
    im_ref_sm = visual_tools.grab_block(im_ref,coord_sel1,[drift_size]*3)
    cents_ref1 = get_STD_centers(im_ref_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the ref cube 1
    im_ref_sm = visual_tools.grab_block(im_ref,coord_sel2,[drift_size]*3)
    cents_ref2 = get_STD_centers(im_ref_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the ref cube 2

    for iim,(im, _name) in enumerate(zip(_bead_ims[1:], bead_names[1:])):
        # check if key exists
        if _name in total_drift and not overwrite:
            change_markers[_name] = False
            continue
        else:
            change_markers[_name] = True
            # dynamic seeding
            if dynamic:
                th_seed = scoreatpercentile(im, dynamic_th_percent) * 0.45;
            # fit target image
            im_sm = visual_tools.grab_block(im,coord_sel1,[drift_size]*3)
            cents1 = get_STD_centers(im_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the cube 1
            im_sm = visual_tools.grab_block(im,coord_sel2,[drift_size]*3)
            cents2 = get_STD_centers(im_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the cube 2
            if verbose:
                print("Aligning "+str(iim+1), _name)
            # calculate drift
            txyz1 = visual_tools.translation_aling_pts(cents_ref1,cents1,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=plt_val)
            txyz2 = visual_tools.translation_aling_pts(cents_ref2,cents2,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=plt_val)
            txyz = (txyz1+txyz2)/2.
            # if two drifts are really different:
            if np.sum(np.abs(txyz1-txyz2))>3:
                fail_count += 1; # count times of suspected failure
                print(f"Suspecting failure for {_name}")
                #drift_size+=10
                coord_sel3 = np.array([0,drift_size,-drift_size])+coord_sel
                im_ref_sm = visual_tools.grab_block(im_ref,coord_sel3,[drift_size]*3)
                cents_ref3 = get_STD_centers(im_ref_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the ref cube 3
                im_sm = visual_tools.grab_block(im,coord_sel3,[drift_size]*3)
                cents3 = get_STD_centers(im_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the cube 3
                txyz3 = visual_tools.translation_aling_pts(cents_ref3,cents3,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=plt_val)
                #print txyz1,txyz2,txyz3
                if np.sum(np.abs(txyz3-txyz1))<np.sum(np.abs(txyz3-txyz2)):
                    txyz = (txyz1+txyz3)/2.
                    print(txyz1,txyz3)
                else:
                    txyz = (txyz2+txyz3)/2.
                    print(txyz2,txyz3)

            txyzs.append(txyz)
            # store in total_drift if overwrite / not exist
            total_drift[_name] = sum(txyzs)
            # inherit centers and ref image
            cents_ref1 = cents1
            cents_ref2 = cents2
            ref = iim
            im_ref = im
    # if ref-frame changed, modify old drift files:
    if old_ref_frame != '':
        ref_drift = total_drift[old_ref_frame]
        for _hyb_name, _mk in change_markers.items():
            if not _mk: # if not changed
                total_drift[_hyb_name] += ref_drift
                if verbose:
                    print(f"Update drift for {_hyb_name}")

    # if any_changes exist and save, do save
    if True in list(change_markers.values()) and save:
        if not os.path.exists(drift_folder):
            os.makedirs(drift_folder)
        pickle.dump(total_drift,open(save_filename,'wb'))
    if verbose:
        print("-- total time spent in drift correction:", time.time()-_start_time)

    return total_drift, fail_count


# function to generate illumination profiles
def generate_illumination_correction(ims, threshold_percentile=98, gaussian_sigma=40,
                                     save=True, save_name='', save_dir=r'.', make_plot=False):
    '''Take into a list of beads images, report a mean illumination strength image'''
    from scipy import ndimage
    import os
    import pickle as pickle
    from scipy.stats import scoreatpercentile
    # initialize total image
    _ims = ims
    total_ims = []
    # calculate total(average) image
    for _i,_im in enumerate(_ims):
        im_stack = _im.mean(0)
        threshold = scoreatpercentile(_im, threshold_percentile);
        im_stack[im_stack > threshold] = np.nan;
        total_ims.append(im_stack);
    # gaussian fliter total image to denoise
    total_im = np.nanmean(np.array(total_ims),0)
    total_im[np.isnan(total_im)] = np.nanmean(total_im)
    if gaussian_sigma:
        fit_im = ndimage.gaussian_filter(total_im, gaussian_sigma);
    else:
        fit_im = total_im;
    fit_im = fit_im / np.max(fit_im);

    # make plot
    if make_plot:
        f = plt.figure()
        plt.imshow(fit_im)
        plt.title('Illumination profile for '+ save_name)
        plt.colorbar()
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_filename = save_dir + os.sep + 'Illumination_correction_'+save_name+'.pkl';
        pickle.dump(fit_im, open(save_filename, 'wb'));
        if make_plot:
            plt.savefig(save_filename.replace('.pkl','.png'));
    return fit_im

def Illumination_correction(ims, correction_channel, correction_power=1.75,
                            correction_folder=_correction_folder, verbose=True):
    '''illumination correction for one list of images
    Inputs:
        ims: list of images, list
        correction_channel: '750','647','561','488','405'
        correction_power: power correction
        correction_folder: path to correction pickle files, string
    Outputs:
        _ims: corrected images, list'''
    # check correction_channel input
    _channel_names = ['750','647','561','488','405'];
    if verbose:
        print("- Processing illumination correction for channel:",correction_channel);
    if str(correction_channel) not in _channel_names:
        raise ValueError('wrong channel name input! should be among '+str(_channel_names));
    # check if ims is a list or single image
    if not isinstance(ims, list):
        _ims = [ims]
    else:
        _ims = ims
    # load correcton profile
    _corr_filename = os.path.join(correction_folder, 'Illumination_correction_'+str(correction_channel)+'.pkl')
    if not os.path.exists(_corr_filename):
        raise IOError(f"Required illumiation correction file {_corr_filename} does not exist!")
    with open(_corr_filename, 'rb') as handle:
        _ic_profile = pickle.load(handle)
    # do correction
    if verbose:
        print("-- Number of images to be corrected:", len(_ims))
    if len(_ims[0].shape) == 2: # if 2D
        _ims = [(_im/_ic_profile**1.75).astype(np.unit16) for _im in _ims];
    else: # else, 3D
        _ims = [(_im/_ic_profile[np.newaxis,:,:]**correction_power).astype(np.uint16) for _im in _ims];
    del(_ic_profile) # clear 
    return _ims

# Function to generate chromatic abbrevation profile
def generate_chromatic_abbrevation_correction(ims, names, master_folder, channels, corr_channel, ref_channel,
                                              fitting_save_subdir=r'Analysis\Beads', seed_th_per=99.92,
                                              correction_folder=_correction_folder,
                                              make_plot=False,
                                              save=True, save_folder=_correction_folder,
                                              force=False, verbose=True):
    '''Generate chromatic abbrevation profile from list of images
    Inputs:
        ims: images, list of 3D image for beads, in multi color
        names: list of corresponding names
        master_folder: master directory of beads data, string
        channels: list of channels used in this data, list of string or ints
        corr_channel: channel to be corrected, str or int
        ref_channel: reference channel, str or int
        fitting_save_subdir: sub_directory under master_folder to save fitting results, str (default: None)
        seed_th_per: intensity percentile for seeds during beads-fitting, float (default: 99.92)
        correction_folder: full directory for illumination correction profiles, string (default: ...)
        save: whether directly save profile, bool (default: True)
        save_folder: full path to save correction file, str (default: correction_folder)
        force: do fitting and saving despite of existing files, bool (default: False)
        verbose: say something!, bool (default: True)
    Output:
        _cc_profiles: list of correction profiles (in order of z,x,y, based on x,y coordinates)
    '''
    # Check inputs
    if len(ims) != len(names): # check length
        raise ValueError('Input images and names doesnt match!');
    _default_channels = ['750','647','561','488','405'];
    _channels = [str(_ch) for _ch in channels];
    for _ch in _channels: # check channels
        if _ch not in _default_channels:
            raise ValueError('Channel '+_ch+" not exist in default channels");
    if str(corr_channel) not in _channels:
        raise ValueError("correction channel "+str(corr_channel)+" is not given in channels");
    if str(ref_channel) not in _channels:
        raise ValueError("reference channel "+str(ref_channel)+" is not given in channels");

    # imports
    from scipy.stats import scoreatpercentile
    import os
    import matplotlib.pyplot as plt
    import scipy.linalg

    # split images into channels
    _splitted_ims = get_img_info.split_channels(ims, names, num_channel=len(_channels), buffer_frames=10, DAPI=False)
    _cims = _splitted_ims[_channels.index(str(corr_channel))]
    _rims = _splitted_ims[_channels.index(str(ref_channel))]
    # initialize list of beads centers:
    _ccts, _rcts, _shifts = [], [], []

    # loop through all images and calculate profile
    for _cim,_rim,_name in zip(_cims,_rims,names):
        try:
            # fit correction channel
            _cim = Illumination_correction(_cim, corr_channel, correction_folder=correction_folder,
                                                        verbose=verbose)[0]
            _cct = get_STD_centers(_cim, scoreatpercentile(_cim, seed_th_per), verbose=verbose,
                                        save=save, force=force, save_folder=master_folder+os.sep+fitting_save_subdir,
                                        save_name=_name.split(os.sep)[-1].replace('.dax', '_'+str(corr_channel)+'_fitting.pkl'))
            # fit reference channel
            _rim = Illumination_correction(_rim, ref_channel, correction_folder=correction_folder,
                                                        verbose=verbose)[0]
            _rct = get_STD_centers(_rim, scoreatpercentile(_rim, seed_th_per), verbose=verbose,
                                        save=save, force=force, save_folder=master_folder+os.sep+fitting_save_subdir,
                                        save_name=_name.split(os.sep)[-1].replace('.dax', '_'+str(ref_channel)+'_fitting.pkl'))
            # Align points
            _aligned_cct, _aligned_rct, _shift = visual_tools.beads_alignment_fast(_cct ,_rct, outlier_sigma=1, unique_cutoff=2)
            # append
            _ccts.append(_aligned_cct);
            _rcts.append(_aligned_rct);
            _shifts.append(_shift);
            # make plot
            if make_plot:
                fig = plt.figure()
                plt.plot(_aligned_rct[:,2], _aligned_rct[:,1],'r.', alpha=0.3)
                plt.quiver(_aligned_rct[:,2], _aligned_rct[:,1], _shift[:,2], _shift[:,1])
                plt.imshow(_rim.sum(0))
                plt.show()
        except:
            pass
    ## do fitting to explain chromatic abbrevation
    # merge
    _corr_beads = np.concatenate(_ccts);
    _ref_beads = np.concatenate(_rcts);
    _shift_beads = np.concatenate(_shifts);
    # initialize
    dz, dx, dy = _cims[0].shape
    _cc_profiles = []
    # loop through
    for _j in range(3): # 3d correction
        if _j == 0: # for z, do order-2 polynomial fitting
            _data = np.concatenate((_ref_beads[:,1:]**2, (_ref_beads[:,1] * _ref_beads[:,2])[:,np.newaxis], _ref_beads[:,1:], np.ones([_ref_beads.shape[0],1])),1);
            _C,_r,_,_ = scipy.linalg.lstsq(_data, _shift_beads[:,_j])    # coefficients
            if verbose:
                print('Z axis fitting R^2=', 1 - _r / sum((_shift_beads[:,_j] - _shift_beads[:,_j].mean())**2))
            _cc_profile = np.zeros([dx,dy]);
            for _m in range(dx):
                for _n in range(dy):
                    _cc_profile[_m, _n] = np.dot(_C, [_m**2, _n**2, _m*_n, _m, _n, 1]);
        else: # for x and y, do linear decomposition
            _data = np.concatenate((_ref_beads[:,1:], np.ones([_ref_beads.shape[0],1])),1);
            _C,_r,_,_ = scipy.linalg.lstsq(_data, _shift_beads[:,_j])    # coefficients
            if verbose:
                print('axis'+str(_j)+' fitting R^2=', 1 - _r / sum((_shift_beads[:,_j] - _shift_beads[:,_j].mean())**2))
            _cc_profile = np.zeros([dx,dy]);
            for _m in range(dx):
                for _n in range(dy):
                    _cc_profile[_m, _n] = np.dot(_C, [_m,_n,1]);
        _cc_profiles.append(_cc_profile);
    if save:
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        _save_file = save_folder + os.sep + "Chromatic_correction_"+str(corr_channel)+'_'+str(ref_channel)+'.pkl';
        if verbose:
            print("-- save chromatic abbrevation correction to file", _save_file)
        pickle.dump(_cc_profiles, open(_save_file, 'wb'))
        if make_plot:
            for _d, _cc in enumerate(_cc_profiles):
                plt.figure()
                plt.imshow(_cc)
                plt.colorbar()
                plt.savefig(_save_file.replace('.pkl', '_'+str(_d)+'.png'))

    return _cc_profiles

# Function to do chromatic abbrevation
def Chromatic_abbrevation_correction(ims, correction_channel,
                                     correction_folder=_correction_folder,
                                     target_channel='647', verbose=True):
    '''Chromatic abbrevation correction
        correct everything into 647 channel
    Inputs:
        ims: list of images, list
        correction_channel: '750','647','561','488','405' ('488' and '405' not supported yet)
        correction_folder: path to correction pickle files, string
    Outputs:
        _ims: corrected images, list'''
    if verbose:
        print("- Processing chromatic correction for channel:",correction_channel);
    # check if ims is a list or single image
    if not isinstance(ims, list):
        _ims = [ims]
    else:
        _ims = ims
    # check correction_channel input
    _channel_names = ['750','647','561','488','405']
    if str(correction_channel) not in _channel_names:
        raise ValueError('wrong channel name input! should be among '+str(_channel_names));
    elif str(correction_channel) == str(target_channel):
        return _ims

    from scipy.ndimage.interpolation import map_coordinates

    # load correcton profile
    _correction_file = correction_folder+os.sep+'Chromatic_correction_'+str(correction_channel)+'_'+str(target_channel)+'.pkl';
    if os.path.isfile(_correction_file):
        _cc_profile = pickle.load(open(_correction_file,'rb'))
    else:
        raise IOError("- No chromatic correction profile file founded!");
        return None
    # check correction profile dimensions
    _shape_im = np.shape(_ims[0])[-2:]
    _shape_cc = np.shape(_cc_profile[0])[-2:]
    for _si, _sc in zip(_shape_im, _shape_cc):
        if _si != _sc:
            raise IndexError('Dimension of chromatic abbrevation profile doesnot match image: '+str(_si)+", "+str(_sc));
    ## do correction
    if verbose:
        print("-- Number of images to be corrected:", len(_ims), ", dimension:", len(_ims[0].shape))
    # loop through all dimensions to generate coordinate
    _im = _ims[0]
    _coord = np.indices(np.shape(_im))
    if len(_im.shape) == 2: # if 2D
        _coord = _coord + np.array(_cc_profile)[-2:,:,:]
    else:
        _coord = _coord + np.array(_cc_profile)[:,np.newaxis,:,:]
    del(_cc_profile) # clear
    # loop through images
    _corr_ims = []; # initialize corrected images
    for _im in _ims:

        if len(_im.shape) == 2: # if 2D
            _cim = map_coordinates(_im, _coord.reshape(2,-1), mode='nearest')
            _cim = _cim.reshape(_im.shape)
            _corr_ims.append(_cim.astype(np.uint16))
        else: # else, 3D
            _cim = map_coordinates(_im, _coord.reshape(3,-1), mode='nearest')
            _cim = _cim.reshape(_im.shape)
            _corr_ims.append(_cim.astype(np.uint16))
    del(_coord) # clear
    return _corr_ims



## FFT alignment, used for fast alignment
def fftalign(im1,im2,dm=100,plt_val=False):
	"""
	Inputs: 2 images im1, im2 and a maximum displacement max_disp.
	This computes the cross-cor between im1 and im2 using fftconvolve (fast) and determines the maximum
	"""
	from scipy.signal import fftconvolve
	sh = np.array(im2.shape)
	dim1,dim2 = np.max([sh-dm,sh*0],0),sh+dm
	im2_=np.array(im2[dim1[0]:dim2[0],dim1[1]:dim2[1],dim1[2]:dim2[2]][::-1,::-1,::-1],dtype=float)
	im2_-=np.mean(im2_)
	im2_/=np.std(im2_)
	sh = np.array(im1.shape)
	dim1,dim2 = np.max([sh-dm,sh*0],0),sh+dm
	im1_=np.array(im1[dim1[0]:dim2[0],dim1[1]:dim2[1],dim1[2]:dim2[2]],dtype=float)
	im1_-=np.mean(im1_)
	im1_/=np.std(im1_)
	im_cor = fftconvolve(im1_,im2_, mode='full')

	xyz = np.unravel_index(np.argmax(im_cor), im_cor.shape)
	if np.sum(im_cor>0)>0:
		im_cor[im_cor==0]=np.min(im_cor[im_cor>0])
	else:
		im_cor[im_cor==0]=0
	if plt_val:
		plt.figure()
		x,y=xyz[-2:]
		im_cor_2d = np.array(im_cor)
		while len(im_cor_2d.shape)>2:
			im_cor_2d = np.max(im_cor_2d,0)
		plt.plot([y],[x],'ko')
		plt.imshow(im_cor_2d,interpolation='nearest')
		plt.show()
	xyz=np.round(-1 * np.array(im_cor.shape)/2.+xyz).astype(int)
	return xyz

## Translate images given drift
def fast_translate(im,trans):
	shape_ = im.shape
	zmax=shape_[0]
	xmax=shape_[1]
	ymax=shape_[2]
	zmin,xmin,ymin=0,0,0
	trans_=np.array(np.round(trans),dtype=int)
	zmin-=trans_[0]
	zmax-=trans_[0]
	xmin-=trans_[1]
	xmax-=trans_[1]
	ymin-=trans_[2]
	ymax-=trans_[2]
	im_base_0 = np.zeros([zmax-zmin,xmax-xmin,ymax-ymin])
	im_zmin = min(max(zmin,0),shape_[0])
	im_zmax = min(max(zmax,0),shape_[0])
	im_xmin = min(max(xmin,0),shape_[1])
	im_xmax = min(max(xmax,0),shape_[1])
	im_ymin = min(max(ymin,0),shape_[2])
	im_ymax = min(max(ymax,0),shape_[2])
	im_base_0[(im_zmin-zmin):(im_zmax-zmin),(im_xmin-xmin):(im_xmax-xmin),(im_ymin-ymin):(im_ymax-ymin)]=im[im_zmin:im_zmax,im_xmin:im_xmax,im_ymin:im_ymax]
	return im_base_0

def Remove_Constant_Junk(ims, std_th_ratio=6, median_th_ratio=3):
    pass

# correct for illumination _shifts across z layers
def Z_Shift_Correction(im, style='mean', normalization=False, verbose=False):
    '''Function to correct for each layer in z, to make sure they match in term of intensity'''
    if verbose:
        print("- Correct Z axis illumination shifts.")
    if style not in ['mean','median','interpolation']:
        raise ValueError('wrong style input for Z shift correction!')
    _nim = np.zeros(np.shape(im))
    if style == 'mean':
        _norm_factors = [np.mean(_lyr) for _lyr in im]
    elif style == 'median':
        _norm_factors = [np.median(_lyr) for _lyr in im]
    elif stype == 'interpolation':
        _means = np.array([np.mean(_lyr) for _lyr in im])
        _interpolation = _means
        _interpolation[1:-1] += _means[:-2]
        _interpolation[1:-1] += _means[2:]
        _interpolation[1:-1] = _interpolation[1:-1]/3
        _norm_factors = _interpolation
    # loop through layers
    for _i, _lyr in enumerate(im):
        if _norm_factors[_i] > 0:
            _nim[_i] = _lyr / _norm_factors[_i]
        else:
            _nim[_i] = _lyr / np.mean(im)
    # if not normalization
    if not normalization:
        _nim = _nim * np.mean(im)

    return _nim.astype(np.uint16)

def Remove_Hot_Pixels(im, hot_pix_th=0.50, interpolation_style='nearest', hot_th=5, verbose=False):
    '''Function to remove hot pixels by interpolation in each single layer'''
    if verbose:
        print("- Remove hot pixels")
    # create convolution matrix, ignore boundaries for now
    _conv = (np.roll(im,1,1)+np.roll(im,-1,1)+np.roll(im,1,2)+np.roll(im,1,2))/4
    # hot pixels must be have signals higher than average of neighboring pixels by hot_th in more than hot_pix_th*total z-stacks
    _hotmat = im > hot_th * _conv
    _hotmat2D = np.sum(_hotmat,0)
    _hotpix_cand = np.where(_hotmat2D > hot_pix_th*np.shape(im)[0])
    del(_conv, _hotmat, _hotmat2D)
    # if no hot pixel detected, directly exit
    if len(_hotpix_cand[0]) == 0:
        return im
    # create new image to interpolate the hot pixels with average of neighboring pixels
    _nim = np.zeros(np.shape(im))+im
    if interpolation_style == 'nearest':
        for _x, _y in zip(_hotpix_cand[0],_hotpix_cand[1]):
            if _x > 0 and  _y > 0 and _x < im.shape[1]-1 and  _y < im.shape[2]-1:
                _nim[:,_x,_y] = (_nim[:,_x+1,_y]+_nim[:,_x-1,_y]+_nim[:,_x,_y+1]+_nim[:,_x,_y-1])/4
    del(_hotpix_cand)
    return _nim.astype(np.uint16)

# wrapper
def correction_wrapper(im, channel, correction_folder=_correction_folder,
                       z_shift_corr=True, hot_pixel_remove=True,
                       illumination_corr=True, chromatic_corr=True,
                       temp_folder='I:\Pu_temp',temp_name='',
                       overwrite_temp=True, return_type='filename', verbose=False):
    """wrapper for all correction steps to one image, used for multi-processing
    Inputs:
        im: image
        channel: which channel is this image
        correction_folder: path to find correction files
        z_shift_corr: whether do z-shift correction, bool (default: True)
        hot_pixel_remove: whether remove hot-pixels, bool (default: True)
        illumination_corr: whether do illumination correction, bool (default: True)
        chromatic_corr: whether do chromatic abbrevation correction, bool (default: True)
        verbose: whether say something!, bool
        """

    # create temp folder if necessary
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder);
    if temp_name == '':
        temp_name = 'temp_'+str(channel)+'_corrected';
    # temp file
    _temp_fl = os.path.join(temp_folder,
                            temp_name)
    if overwrite_temp and os.path.isfile(_temp_fl):
        os.remove(_temp_fl);
    # check return type
    return_type = return_type.lower()
    if return_type not in ['filename','mmap','image']:
        raise ValueError('Wrong kwd return_type given!');
    # if not overwrite and file exist: directly loading
    if not overwrite_temp and os.path.isfile(_temp_fl+'.npy'):
        if verbose:
            print("-- reading from temp-file:", _temp_fl+'.npy')
        if return_type == 'filename':
            return _temp_fl+'.npy'
        elif return_type == 'mmap':
            _im_mmap = np.load(_temp_fl+'.npy', mmap_mode='r+')
            return _im_mmap
        elif return_type == 'image':
            _im = np.load(_temp_fl+'.npy', mmap_mode=None)
            return _im
    # localize
    _corr_im = im
    if z_shift_corr:
        # correct for z axis shift
        _corr_im = Z_Shift_Correction(_corr_im, verbose=verbose)
    #print("pass1")
    if hot_pixel_remove:
        # correct for hot pixels
        _corr_im = Remove_Hot_Pixels(_corr_im, verbose=verbose)
    #print("pass2")
    if illumination_corr:
        # illumination correction
        _corr_im = Illumination_correction(_corr_im, correction_channel=channel,
                    correction_folder=correction_folder, verbose=verbose)[0]
    #print("pass3")
    if chromatic_corr:
        # chromatic correction
        _corr_im = Chromatic_abbrevation_correction(_corr_im, correction_channel=channel,
                    correction_folder=correction_folder, verbose=verbose)[0]
    #print("pass4")
    # if save temp file, save to a file, release original one, return a memory-map
    if return_type == 'filename':
        if not os.path.exists(temp_folder):
            print(f"Create Temp folder:{temp_folder}")
            os.makedirs(_temp_folder)
        if verbose:
            print(f"--- saving temp to file:{_temp_fl}")
        np.save(_temp_fl, _corr_im)
        del(_corr_im, im)
        return _temp_fl+'.npy'
    elif return_type == 'mmap':
        if verbose:
            print(f"--- saving temp to file for mmap:{_temp_fl}")
        np.save(_temp_fl, _corr_im)
        del(_corr_im, im)
        _im_mmap = np.load(_temp_fl+'.npy', mmap_mode='r+');
        return _im_mmap
    # else: directly return the array
    else:
        del(im)
        return _corr_im


def correct_single_image(filename, im_size, channels, target_channel, raw_im=None,
                         num_buffer_frames=10, return_type='image', correction_folder=_correction_folder,
                         z_shift_corr=True, hot_pixel_remove=True, illumination_corr=True, chromatic_corr=True,
                         save=True, save_folder='I:\Pu_temp', save_name='', save_filetype='.npy', overwrite=False, verbose=False):
    """wrapper for all correction steps to one image, used for multi-processing
    Inputs:
        filename: full filename of a dax_file or npy_file for one image, string
        im_size: z-x-y size of the image, list of 3
        channels: channels used in this image, list of str(for example, ['750','647','561'])
        target_channel: target_channel to be extracted, str (for example. '647')
        raw_im: directly give image rather than loading, this will overwrite filename etc.
        num_buffer_frames: number of buffer frame in front and back of image, int (default:10)
        return_type: whether return filename or image directly, str('filename' or 'image')
        correction_folder: path to find correction files
        z_shift_corr: whether do z-shift correction, bool (default: True)
        hot_pixel_remove: whether remove hot-pixels, bool (default: True)
        illumination_corr: whether do illumination correction, bool (default: True)
        chromatic_corr: whether do chromatic abbrevation correction, bool (default: True)
        save_folder: folder to save the temp file (if generated), str (default: 'I:\Pu_temp')
        save_name: default savename of temp file, str (default: ''+'_corrected')
        save_filetype: file type to be saved, '.npy' or '.dax'
        overwrite: whether overwrite existing temp-file, bool (default: False)
        verbose: whether say something!, bool (default:True)
        """
    ## check inputs
    return_type = return_type.lower()
    if return_type not in ['filename', 'image']:
        raise ValueError(
            f"Wrong return_type given! should be 'filename' or 'image' but {return_type} is given")
    save_filetype = save_filetype.lower()
    if save_filetype not in ['.npy', '.dax']:
        raise ValueError(
            f"Wrong save_filetype given! should be '.npy' or '.dax' but {save_filetype} is given")
    target_channel = str(target_channel)
    channels = [str(ch) for ch in channels]
    if target_channel not in channels:
        raise ValueError(
            f"Target channel {target_channel} doesn't exist in channels:{channels}")
    # if return filename ,file must be saved
    if return_type == 'filename':
        save = True
    # generate full savename if not given
    if save_name == '':
        save_name = filename.split(
            '.'+filename.split('.')[-1])[0].split(os.sep)[-1]+'_corrected'
    # only 3 channels requires chromatic correction
    if target_channel not in ['750', '647', '561']:
        chromatic_corr = False
    # save file
    _save_fl = os.path.join(save_folder, save_name)

    # if not overwrite and file exist: directly loading
    if not overwrite and os.path.isfile(_save_fl+save_filetype):
        if verbose:
            print("-- reading from temp-file:", _save_fl+save_filetype)
        if return_type == 'filename':
            return _save_fl+save_filetype
        elif return_type == 'image':
            if save_filetype == '.npy':
                _corr_im = np.load(_save_fl+save_filetype)
            elif save_filetype == '.dax':
                _corr_im = visual_tools.DaxReader(
                    _save_fl+save_filetype).loadAll()
            return _corr_im
    # no file or overwrite:
    else:
        # check if image is given
        if raw_im is not None:
            if len(np.shape(raw_im)) == 3:
                if (np.array(raw_im.shape) == np.array(im_size[:3])).all():
                    _corr_im = raw_im
                else:
                    if verbose:
                        print("- shape of raw image doesn't match, proceed to load from file.")
            else:
                if verbose:
                    print("- dimension of raw image doesn't match, proceed to load from file.")   
            
        # load image
        if '_corr_im' not in locals():
            _channel_id = channels.index(target_channel)
            _corr_im = visual_tools.slice_image(filename, list(im_size),
                                                [num_buffer_frames, im_size[0] -
                                                    num_buffer_frames],
                                                [0, im_size[1]], [0, im_size[2]], len(channels), _channel_id)
        if z_shift_corr:
            # correct for z axis shift
            _corr_im = Z_Shift_Correction(_corr_im, verbose=verbose)
        #print("pass1")
        if hot_pixel_remove:
            # correct for hot pixels
            _corr_im = Remove_Hot_Pixels(_corr_im, verbose=verbose)
        #print("pass2")
        if illumination_corr:
            # illumination correction
            _corr_im = Illumination_correction(_corr_im, correction_channel=target_channel,
                                                           correction_folder=correction_folder, verbose=verbose)[0]
        #print("pass3")
        if chromatic_corr:
            # chromatic correction
            _corr_im = Chromatic_abbrevation_correction(_corr_im, correction_channel=target_channel,
                                                                    correction_folder=correction_folder, verbose=verbose)[0]
        #print("pass4")

        ## save
        if save:
            if not os.path.exists(save_folder):
                print(f"Create Temp folder:{save_folder}")
                os.makedirs(save_folder)
            if verbose:
                print(f"--- saving temp to file:{_save_fl+save_filetype}")
            if save_filetype == '.npy':
                np.save(_save_fl, _corr_im)
            elif save_filetype == '.dax':
                _corr_im.tofile(_save_fl+save_filetype)
        ## return
        if return_type == 'filename':
            del(_corr_im)
            return _save_fl+save_filetype
        elif return_type == 'image':
            return _corr_im
