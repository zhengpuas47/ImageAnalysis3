from ImageAnalysis3 import visual_tools as vis
import ImageAnalysis3 as ia
import numpy as np
import pickle
import matplotlib.pylab as plt
import os, glob, sys

sys.path.append(r'C:\Users\puzheng\Documents\python-functions\python-functions-library')

def get_STD_beaddrift(bead_ims, bead_names, analysis_folder, fovs, fov_id,
                      illumination_correction=False, ic_channel=488,
                      correction_folder=r'C:\Users\Pu Zheng\Documents\Corrections',
                      repeat=True, plt_val=False, cutoff_=3, xyz_res_=1,
                      coord_sel=None, sz_ex=100, ref=0, force=False, save=True, quiet=False, th_seed=150, dynamic=False):
    """Given a list of bead images This handles the fine bead drift correction.
    If save is true this requires global paramaters analysis_folder,fovs,fov_id
    Inputs:
        bead_ims: list of images, list of ndarray
        analysis_folder: full directory to store analysis files, string
        fovs: names for all field of views, list of strings
        fov_id: the id for field of view to be analysed, int
        """
    # define a sub function to do fitting
    import ia.visual_tools as vis
    from scipy.stats import scoreatpercentile

    # Initialize failed counts
    fail_count = 0;

    # if save, check existing pickle file, if exist, don't repeat
    if save:
        save_cor = analysis_folder+os.sep+fovs[fov_id].replace('.dax','__current_cor.pkl')
        if os.path.exists(save_cor):
            total_drift = pickle.load(open(save_cor,'rb'), encoding='latin1')
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
        if not quiet:
            print("Fitting reference:", ref)
        if coord_sel is None:
            coord_sel = np.array(im_ref.shape)/2
        coord_sel1 = np.array([0,-sz_ex,-sz_ex]) + coord_sel
        if dynamic:
            th_seed = scoreatpercentile(im_ref, 99) * 0.5
        im_ref_sm = vis.grab_block(im_ref,coord_sel1,[sz_ex]*3)
        cents_ref1 = get_STD_centers(im_ref_sm, th_seed=th_seed)#list of fits of beads in the ref cube 1
        coord_sel2 = np.array([0,sz_ex,sz_ex]) + coord_sel
        im_ref_sm = vis.grab_block(im_ref,coord_sel2,[sz_ex]*3)
        cents_ref2 = get_STD_centers(im_ref_sm, th_seed=th_seed)#list of fits of beads in the ref cube 2

        txyzs = []
        for iim,im in enumerate(_bead_ims):
            # if this frame is reference, continue
            if iim == ref:
                txyzs.append(np.array([0.,0.,0.]));
                continue;
            if dynamic:
                th_seed = scoreatpercentile(im,99)*0.5
            im_sm = vis.grab_block(im,coord_sel1,[sz_ex]*3)
            cents1 = get_STD_centers(im_sm, th_seed=th_seed)#list of fits of beads in the cube 1
            im_sm = vis.grab_block(im,coord_sel2,[sz_ex]*3)
            cents2 = get_STD_centers(im_sm, th_seed=th_seed)#list of fits of beads in the cube 2
            if not quiet:
                print("Aligning "+str(iim))
            txyz1 = vis.translation_aling_pts(cents_ref1,cents1,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)
            txyz2 = vis.translation_aling_pts(cents_ref2,cents2,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)

            txyz = (txyz1+txyz2)/2.
            if np.sum(np.abs(txyz1-txyz2))>3:
                fail_count += 1; # count times of suspected failure
                print("Suspecting failure.")
                #sz_ex+=10
                coord_sel3 = np.array([0,sz_ex,-sz_ex])+coord_sel
                im_ref_sm = vis.grab_block(im_ref,coord_sel3,[sz_ex]*3)
                cents_ref3 = get_STD_centers(im_ref_sm, th_seed=th_seed)#list of fits of beads in the ref cube 3
                im_sm = vis.grab_block(im,coord_sel3,[sz_ex]*3)
                cents3 = get_STD_centers(im_sm, th_seed=th_seed)#list of fits of beads in the cube 3
                txyz3 = vis.translation_aling_pts(cents_ref3,cents3,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)
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

def get_STD_centers(im, th_seed=150, close_threshold=0.01, quiet=False, plt_val=False,
                    save=False, save_folder='', save_name='', force=False):
    '''Fit beads for one image:
    Inputs:
        im: image, ndarray
        th_seeds: threshold for seeding, float (default: 150)
        close_threshold: threshold for removing duplicates within a distance, float (default: 0.01)
        quiet: say nothing!, bool (default: False)
        plt_val: whether making plot, bool (default: False)
        save: whether save fitting result, bool (default: False)
        save_folder: full path of save folder, str (default: None)
        save_name: full name of save file, str (default: None)
        force: whether force fitting despite of saved file, bool (default: False)
    Outputs:
        beads: fitted spots with information, n by 4 array'''
    import os
    import pickle as pickle
    if not force and os.path.exists(save_folder+os.sep+save_name) and save_name != '':
        if not quiet:
            print("- loading file:,", save_folder+os.sep+save_name)
        beads = pickle.load(open(save_folder+os.sep+save_name, 'rb'), encoding='latin1');
        if not quiet:
            print("--", len(beads), " of beads loaded.")
        return beads
    else:
        # seeding
        seeds = vis.get_seed_points_base(im,gfilt_size=0.75,filt_size=3,th_seed=th_seed,hot_pix_th=4)
        # fitting
        pfits = vis.fit_seed_points_base_fast(im,seeds,width_z=1.8*1.5/2,width_xy=1.,radius_fit=5,n_max_iter=10,max_dist_th=0.25,quiet=quiet)
        # get coordinates for fitted beads
        if len(pfits) > 0:
            beads = pfits[:,1:4];
            # remove very close spots
            remove = 0;
            for i,bead in enumerate(beads):
                if np.sum(np.sum((beads-bead)**2, axis=1)<close_threshold) > 1:
                    beads = np.delete(beads, i-remove, 0);
                    remove += 1;
        else:
            beads = None;
        if not quiet:
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
            if not quiet:
                print("-- saving beads to", save_folder+os.sep+save_name)
            pickle.dump(beads, open(save_folder+os.sep+save_name, 'wb'));

        return beads


def STD_beaddrift_sequential(bead_ims, bead_names, drift_folder, fovs, fov_id,
                      illumination_correction=False, ic_channel=488,
                      correction_folder=r'C:\Users\Pu Zheng\Documents\Corrections',
                      repeat=True, plt_val=False, cutoff_=3, xyz_res_=1,
                      coord_sel=None, sz_ex=200, force=False, save=True, quiet=False, th_seed=150,
                      dynamic=False, dynamic_th_percent=80):
    """Given a list of bead images This handles the fine bead drift correction.
    If save is true this requires global paramaters drift_folder,fovs,fov_id
    Inputs:
        bead_ims: list of images, list of ndarray
        bead_names: names for bead files, list of strings
        drift_folder: full directory to store analysis files, string
        fovs: names for all field of views, list of strings
        fov_id: the id for field of view to be analysed, int
        Illumination_correction: whether do illumination correction, bool

        """
    # define a sub function to do fitting
    from scipy.stats import scoreatpercentile
    # Initialize failed counts
    fail_count = 0;

    # if save, check existing pickle file, if exist, don't repeat
    if save:
        save_cor = drift_folder+os.sep+fovs[fov_id].replace('.dax','_sequential_current_cor.pkl')
        # if savefile already exists, check whether dimension was correct
        if os.path.exists(save_cor):
            total_drift = pickle.load(open(save_cor,'rb'), encoding='latin1')
            if len(list(total_drift.keys()))==len(bead_ims):
                if not quiet:
                    print("length matches:",len(bead_ims), "and forcing calculation="+str(force))
                repeat=False
            else:
                if not quiet:
                    print("length doesnot match, redo drift correction!");
    repeat = repeat or force

    # if do illumination_correction:
    if illumination_correction:
        _bead_ims = [Illumination_correction(_im,ic_channel, correction_folder=correction_folder, verbose=False) for _im in bead_ims]
    else:
        _bead_ims = bead_ims;

    # repeat if no fitted data exist
    if repeat:
        # initialize drifts
        txyzs = [];
        txyzs.append(np.array([0.,0.,0.]));
        # define selected coordinates in each field of view
        if coord_sel is None:
            coord_sel = np.array(_bead_ims[0].shape)/2
        coord_sel1 = np.array([0,-sz_ex,-sz_ex]) + coord_sel
        coord_sel2 = np.array([0,sz_ex,sz_ex]) + coord_sel

        # if more than one image provided:
        if len(_bead_ims) > 1:

            ref = 0; # initialize reference id
            im_ref = _bead_ims[ref]; # initialize reference image
            if dynamic:
                th_seed = scoreatpercentile(im_ref, dynamic_th_percent) * 0.5;
            # start fitting image 0
            im_ref_sm = vis.grab_block(im_ref,coord_sel1,[sz_ex]*3)
            cents_ref1 = get_STD_centers(im_ref_sm, th_seed=th_seed)#list of fits of beads in the ref cube 1
            im_ref_sm = vis.grab_block(im_ref,coord_sel2,[sz_ex]*3)
            cents_ref2 = get_STD_centers(im_ref_sm, th_seed=th_seed)#list of fits of beads in the ref cube 2


            for iim,im in enumerate(_bead_ims[1:]):
                if dynamic:
                    th_seed = scoreatpercentile(im, dynamic_th_percent) * 0.5;
                #print th_seed
                # fit target image
                im_sm = vis.grab_block(im,coord_sel1,[sz_ex]*3)
                cents1 = get_STD_centers(im_sm, th_seed=th_seed)#list of fits of beads in the cube 1
                im_sm = vis.grab_block(im,coord_sel2,[sz_ex]*3)
                cents2 = get_STD_centers(im_sm, th_seed=th_seed)#list of fits of beads in the cube 2
                if not quiet:
                    print("Aligning "+str(iim+1))
                # calculate drift
                txyz1 = vis.translation_aling_pts(cents_ref1,cents1,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)
                txyz2 = vis.translation_aling_pts(cents_ref2,cents2,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)
                txyz = (txyz1+txyz2)/2.
                # if two drifts are really different:
                if np.sum(np.abs(txyz1-txyz2))>3:
                    fail_count += 1; # count times of suspected failure
                    print("Suspecting failure.")
                    #sz_ex+=10
                    coord_sel3 = np.array([0,sz_ex,-sz_ex])+coord_sel
                    im_ref_sm = vis.grab_block(im_ref,coord_sel3,[sz_ex]*3)
                    cents_ref3 = get_STD_centers(im_ref_sm, th_seed=th_seed)#list of fits of beads in the ref cube 3
                    im_sm = vis.grab_block(im,coord_sel3,[sz_ex]*3)
                    cents3 = get_STD_centers(im_sm, th_seed=th_seed)#list of fits of beads in the cube 3
                    txyz3 = vis.translation_aling_pts(cents_ref3,cents3,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)
                    #print txyz1,txyz2,txyz3
                    if np.sum(np.abs(txyz3-txyz1))<np.sum(np.abs(txyz3-txyz2)):
                        txyz = (txyz1+txyz3)/2.
                        print(txyz1,txyz3)
                    else:
                        txyz = (txyz2+txyz3)/2.
                        print(txyz2,txyz3)

                txyzs.append(txyz)
                # use this centers as reference for the next hyb
                cents_ref1 = cents1;
                cents_ref2 = cents2;
                ref += 1;
                im_ref = _bead_ims[ref];

        # convert to total drift
        total_drift = {_name: sum(txyzs[:_i+1]) for _i, _name in enumerate(bead_names)}
        if save:
            if not os.path.exists(drift_folder):
                os.makedirs(drift_folder)
            save_cor = drift_folder+os.sep+fovs[fov_id].replace('.dax','_sequential_current_cor.pkl')
            pickle.dump(total_drift,open(save_cor,'wb'))

    return total_drift, repeat, fail_count

# function to generate illumination profiles
def generate_illumination_correction(ims, threshold_percentile=98, gaussian_sigma=40,
                                     save=True, save_name='', save_dir=r'.', make_plot=False):
    '''Take into a list of beads images, report a mean illumination strength image'''
    from scipy import ndimage
    import os
    import pickle as pickle
    from scipy.stats import scoreatpercentile
    # initialize total image
    _ims = ims;
    total_ims = [];
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

def Illumination_correction(ims, correction_channel,
                            correction_folder='C:/Users/puzheng/Documents/Correction_Files', verbose=True):
    '''illumination correction for one list of images
    Inputs:
        ims: list of images, list
        correction_channel: '750','647','561','488','405'
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
    with open(_corr_filename, 'rb') as handle:
        _ic_profile = pickle.load(handle)
    # do correction
    if verbose:
        print("-- Number of images to be corrected:", len(_ims))
    if len(_ims[0].shape) == 2: # if 2D
        _ims = [_im/_ic_profile**1.5 for _im in _ims];
    else: # else, 3D
        _ims = [_im/_ic_profile[np.newaxis,:,:]**1.5 for _im in _ims];

    return _ims

# Function to generate chromatic abbrevation profile
def generate_chromatic_abbrevation_correction(ims, names, master_folder, channels, corr_channel, ref_channel,
                                              fitting_save_subdir=r'Analysis\Beads', seed_th_per=99.92,
                                              correction_folder=r'C:\Users\puzheng\Documents\Correction_Files',
                                              make_plot=False,
                                              save=True, save_folder=r'C:\Users\puzheng\Documents\Correction_Files',
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
    _splitted_ims = ia.get_img_info.split_channels(ims, names, num_channel=len(_channels), buffer_frames=10, DAPI=False)
    _cims = _splitted_ims[_channels.index(str(corr_channel))]
    _rims = _splitted_ims[_channels.index(str(ref_channel))]
    # initialize list of beads centers:
    _ccts, _rcts, _shifts = [], [], []

    # loop through all images and calculate profile
    for _cim,_rim,_name in zip(_cims,_rims,names):
        try:
            # fit correction channel
            _cim = ia.corrections.Illumination_correction(_cim, corr_channel, correction_folder=correction_folder,
                                                        verbose=verbose)[0]
            _cct = ia.corrections.get_STD_centers(_cim, scoreatpercentile(_cim, seed_th_per), quiet=not verbose,
                                        save=save, force=force, save_folder=master_folder+os.sep+fitting_save_subdir,
                                        save_name=_name.split(os.sep)[-1].replace('.dax', '_'+str(corr_channel)+'_fitting.pkl'))
            # fit reference channel
            _rim = ia.corrections.Illumination_correction(_rim, ref_channel, correction_folder=correction_folder,
                                                        verbose=verbose)[0]
            _rct = ia.corrections.get_STD_centers(_rim, scoreatpercentile(_rim, seed_th_per), quiet=not verbose,
                                        save=save, force=force, save_folder=master_folder+os.sep+fitting_save_subdir,
                                        save_name=_name.split(os.sep)[-1].replace('.dax', '_'+str(ref_channel)+'_fitting.pkl'))
            # Align points
            _aligned_cct, _aligned_rct, _shift = ia.visual_tools.beads_alignment_fast(_cct ,_rct, outlier_sigma=1, unique_cutoff=2)
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
            os.mkdir(save_folder);
        _save_file = save_folder + os.sep + "Chromatic_correction_"+str(corr_channel)+'_'+str(ref_channel)+'.pkl';
        if verbose:
            print("-- save chromatic abbrevation correction to file", _save_file)
        pickle.dump(_cc_profiles, open(_save_file, 'wb'))
        if make_plot:
            for _d, _cc in enumerate(_cc_profiles):
                plt.figure();
                plt.imshow(_cc)
                plt.colorbar();
                plt.savefig(_save_file.replace('.pkl', '_'+str(_d)+'.png'))

    return _cc_profiles

# Function to do chromatic abbrevation
def Chromatic_abbrevation_correction(ims, correction_channel,
                                     correction_folder=r'C:/Users/puzheng/Documents/Correction_Files',
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
    _channel_names = ['750','647','561','488','405'];
    if str(correction_channel) not in _channel_names:
        raise ValueError('wrong channel name input! should be among '+str(_channel_names));
    elif str(correction_channel) == str(target_channel):
        return _ims

    # imports
    import pickle as pickle;
    import os;
    import numpy as np;
    from scipy.ndimage.interpolation import map_coordinates

    # load correcton profile
    _correction_file = correction_folder+os.sep+'Chromatic_correction_'+str(correction_channel)+'_'+str(target_channel)+'.pkl';
    if os.path.isfile(_correction_file):
        _cc_profile = pickle.load(open(_correction_file,'rb'))
    else:
        print("- No chromatic correction profile file founded!");
        return None
    # check correction profile dimensions
    _shape_im = np.shape(_ims[0])[-2:];
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

    # loop through images
    _corr_ims = []; # initialize corrected images
    for _im in _ims:

        if len(_im.shape) == 2: # if 2D
            _cim = map_coordinates(_im, _coord.reshape(2,-1))
            _cim = _cim.reshape(_im.shape)
            _corr_ims.append(_cim)
        else: # else, 3D
            _cim = map_coordinates(_im, _coord.reshape(3,-1))
            _cim = _cim.reshape(_im.shape)
            _corr_ims.append(_cim)

    return _corr_ims;



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
	xyz=np.round(-np.array(im_cor.shape)/2.+xyz).astype(int)
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



# correct for illumination _shifts across z layers
def Z_Shift_Correction(im, style='mean', normalization=False, verbose=False):
    '''Function to correct for each layer in z, to make sure they match in term of intensity'''
    if style not in ['mean','median','interpolation']:
        raise ValueError('wrong style input for Z shift correction!');
    _nim = np.zeros(np.shape(im));
    if style == 'mean':
        _norm_factors = [np.mean(_lyr) for _lyr in im];
    elif style == 'median':
        _norm_factors = [np.median(_lyr) for _lyr in im];
    elif stype == 'interpolation':
        _means = np.array([np.mean(_lyr) for _lyr in im]);
        _interpolation = _means;
        _interpolation[1:-1] += _means[:-2];
        _interpolation[1:-1] += _means[2:];
        _interpolation[1:-1] = _interpolation[1:-1]/3;
        _norm_factors = _interpolation;
    # loop through layers
    for _i, _lyr in enumerate(im):
        _nim[_i] = _lyr / _norm_factors[_i];
    # if not normalization
    if not normalization:
        _nim = _nim * np.mean(im);

    return _nim.astype(np.uint16)

def Remove_Hot_Pixels(im, hot_pix_th=0.2, interpolation_style='nearest'):
    '''Function to remove hot pixels by interpolation in each single layer'''
    pass
