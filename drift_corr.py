from . import visual_tools as vis
import numpy as np
import pickle as pickle
import matplotlib.pylab as plt
import os, glob, sys
sys.path.append(r'C:\Users\puzheng\Documents\python-functions\python-functions-library')

def get_STD_beaddrift(ims_beads, analysis_folder, fovs, fov_id,
                      repeat=True, plt_val=False, cutoff_=3, xyz_res_=1,
                      coord_sel=None, sz_ex=100, ref=0, force=False, save=True, quiet=False, th_seed=150):
    """Given a list of bead images This handles the fine bead drift correction.
    If save is true this requires global paramaters analysis_folder,fovs,fov_id
    Inputs:
        ims_beads: list of images, list of ndarray
        analysis_folder: full directory to store analysis files, string
        fovs: names for all field of views, list of strings
        fov_id: the id for field of view to be analysed, int
        """
    # define a sub function to do fitting
    def get_STD_centers(im, plt_val=plt_val):
        '''Fit beads for one image:
        Inputs:
            im: image, ndarray
            plt_val: whether making plot, bool
        Outputs:
            beads: fitted spots with information, n by 4 array'''
        seeds = vis.get_seed_points_base(im,gfilt_size=0.75,filt_size=3,th_seed=th_seed,hot_pix_th=4)
        pfits = vis.fit_seed_points_base_fast(im,seeds,width_z=1.8*1.5/2,width_xy=1.,radius_fit=5,n_max_iter=10,max_dist_th=0.25,quiet=quiet)
        # get coordinates for fitted beads
        if len(pfits) > 0:
            beads = pfits[:,1:4];
        else:
            beads = None;
        # make plot if required
        if plt_val:
            plt.figure()
            plt.imshow(np.max(im,0),interpolation='nearest')
            plt.plot(beads[:,2],beads[:,1],'or')
            plt.show()

        return beads

    # Initialize failed counts
    fail_count = 0;

    # if save, check existing pickle file, if exist, don't repeat
    if save:
        save_cor = analysis_folder+os.sep+fovs[fov_id].replace('.dax','__current_cor.pkl')
        if os.path.exists(save_cor):
            txyzs = pickle.load(open(save_cor,'rb'))
            if len(txyzs)==len(ims_beads):
                repeat=False
    repeat = repeat or force

    # repeat if no fitted data exist
    if repeat:
        # choose reference image
        if ref is None: ref = 0
        im_ref = ims_beads[ref]
        if not quiet:
            print("Fitting reference:", ref)
        if coord_sel is None:
            coord_sel = np.array(im_ref.shape)/2
        coord_sel1 = np.array([0,-sz_ex,-sz_ex]) + coord_sel
        im_ref_sm = vis.grab_block(im_ref,coord_sel1,[sz_ex]*3)
        cents_ref1 = get_STD_centers(im_ref_sm)#list of fits of beads in the ref cube 1
        coord_sel2 = np.array([0,sz_ex,sz_ex]) + coord_sel
        im_ref_sm = vis.grab_block(im_ref,coord_sel2,[sz_ex]*3)
        cents_ref2 = get_STD_centers(im_ref_sm)#list of fits of beads in the ref cube 2

        txyzs = []
        for iim,im in enumerate(ims_beads):
            # if this frame is reference, continue
            if iim == ref:
                txyzs.append(np.array([0.,0.,0.]));
                continue;

            im_sm = vis.grab_block(im,coord_sel1,[sz_ex]*3)
            cents1 = get_STD_centers(im_sm)#list of fits of beads in the cube 1
            im_sm = vis.grab_block(im,coord_sel2,[sz_ex]*3)
            cents2 = get_STD_centers(im_sm)#list of fits of beads in the cube 2
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
                cents_ref3 = get_STD_centers(im_ref_sm)#list of fits of beads in the ref cube 3
                im_sm = vis.grab_block(im,coord_sel3,[sz_ex]*3)
                cents3 = get_STD_centers(im_sm)#list of fits of beads in the cube 3
                txyz3 = vis.translation_aling_pts(cents_ref3,cents3,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False)
                #print txyz1,txyz2,txyz3
                if np.sum(np.abs(txyz3-txyz1))<np.sum(np.abs(txyz3-txyz2)):
                    txyz = (txyz1+txyz3)/2.
                    print(txyz1,txyz3)
                else:
                    txyz = (txyz2+txyz3)/2.
                    print(txyz2,txyz3)

            txyzs.append(txyz)
        if save:
            save_cor = analysis_folder+os.sep+fovs[fov_id].replace('.dax','__current_cor.pkl')
            pickle.dump(txyzs,open(save_cor,'wb'))
    return txyzs, repeat, fail_count




def STD_beaddrift_sequential(ims_beads, analysis_folder, fovs, fov_id,
                      repeat=True, plt_val=False, cutoff_=3, xyz_res_=1,
                      coord_sel=None, sz_ex=100, force=False, save=True, quiet=False, th_seed=150):
    """Given a list of bead images This handles the fine bead drift correction.
    If save is true this requires global paramaters analysis_folder,fovs,fov_id
    Inputs:
        ims_beads: list of images, list of ndarray
        analysis_folder: full directory to store analysis files, string
        fovs: names for all field of views, list of strings
        fov_id: the id for field of view to be analysed, int
        """
    # define a sub function to do fitting
    def get_STD_centers(im, plt_val=plt_val):
        '''Fit beads for one image:
        Inputs:
            im: image, ndarray
            plt_val: whether making plot, bool
        Outputs:
            beads: fitted spots with information, n by 4 array'''
        seeds = vis.get_seed_points_base(im,gfilt_size=0.75,filt_size=3,th_seed=th_seed,hot_pix_th=4)
        pfits = vis.fit_seed_points_base_fast(im,seeds,width_z=1.8*1.5/2,width_xy=1.,radius_fit=5,n_max_iter=10,max_dist_th=0.25,quiet=quiet)
        # get coordinates for fitted beads
        if len(pfits) > 0:
            beads = pfits[:,1:4];
        else:
            beads = None;
        # make plot if required
        if plt_val:
            plt.figure()
            plt.imshow(np.max(im,0),interpolation='nearest')
            plt.plot(beads[:,2],beads[:,1],'or')
            plt.show()

        return beads

    # Initialize failed counts
    fail_count = 0;

    # if save, check existing pickle file, if exist, don't repeat
    if save:
        save_cor = analysis_folder+os.sep+fovs[fov_id].replace('.dax','__current_cor.pkl')
        if os.path.exists(save_cor):
            txyzs = pickle.load(open(save_cor,'rb'))
            if len(txyzs)==len(ims_beads):
                repeat=False
    repeat = repeat or force

    # repeat if no fitted data exist
    if repeat:
        # initialize drifts
        txyzs = [];
        txyzs.append(np.array([0.,0.,0.]));
        # define selected coordinates in each field of view
        if coord_sel is None:
            coord_sel = np.array(ims_beads[0].shape)/2
        coord_sel1 = np.array([0,-sz_ex,-sz_ex]) + coord_sel
        coord_sel2 = np.array([0,sz_ex,sz_ex]) + coord_sel

        # if more than one image provided:
        if len(ims_beads) > 1:

            ref = 0; # initialize reference id
            im_ref = ims_beads[ref]; # initialize reference image
            # start fitting image 0
            im_ref_sm = vis.grab_block(im_ref,coord_sel1,[sz_ex]*3)
            cents_ref1 = get_STD_centers(im_ref_sm)#list of fits of beads in the ref cube 1
            im_ref_sm = vis.grab_block(im_ref,coord_sel2,[sz_ex]*3)
            cents_ref2 = get_STD_centers(im_ref_sm)#list of fits of beads in the ref cube 2


            for iim,im in enumerate(ims_beads[1:]):
                # fit target image
                im_sm = vis.grab_block(im,coord_sel1,[sz_ex]*3)
                cents1 = get_STD_centers(im_sm)#list of fits of beads in the cube 1
                im_sm = vis.grab_block(im,coord_sel2,[sz_ex]*3)
                cents2 = get_STD_centers(im_sm)#list of fits of beads in the cube 2
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
                    cents_ref3 = get_STD_centers(im_ref_sm)#list of fits of beads in the ref cube 3
                    im_sm = vis.grab_block(im,coord_sel3,[sz_ex]*3)
                    cents3 = get_STD_centers(im_sm)#list of fits of beads in the cube 3
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
                im_ref = ims_beads[ref];
        if save:
            save_cor = analysis_folder+os.sep+fovs[fov_id].replace('.dax','__current_cor.pkl')
            pickle.dump(txyzs,open(save_cor,'wb'))
    return txyzs, repeat, fail_count

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

def translate(im,trans):
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

def generate_chromosome(folders, fovs, buffer_frames=10, merging_regions=15, ref_frame=0, fft_dim=100, verbose=True):
	'''Function to generate "chromosomes" by merging first several regions
	Given:
		directory of folders, list of strings
		field of views, list of strings
		buffer_frame designed by z-scan, non-negative int
		merging_regions, number of regions to be merged, non-negative int
		ref_frame: frame used for reference, int
		dimension for FFT, positive int
	Return:
		im_means: list of image objects, which consists of merged images
	'''
	_im_means=[]
	for fov_id, _filename in enumerate(fovs):
		print(_filename);
		# spilt images
		_ims_cy5,_ims_cy3,_names=[],[],[]
		for _folder in folders[:merging_regions]:
			_file= _folder+os.sep+_filename
			if os.path.exists(_file):
				_names += [os.path.basename(_folder)]
				_im = vis.DaxReader(_file).loadMap()
				_ims_cy3 += [_im[2::2][int(buffer_frames/2): -int(buffer_frames/2)]]
				_ims_cy5 += [_im[1::2][int(buffer_frames/2): -int(buffer_frames/2)]]
		_drift_roughs = [fftalign(_ims_cy3[ref_frame], _im, dm=fft_dim) for _im in _ims_cy3]
		if verbose:
			print(_drift_roughs[:6])
		_drift_roughs = -np.array(_drift_roughs)+[_drift_roughs[ref_frame]]
		_ims_signal = [translate(_im__drift_roughs[0],-_im__drift_roughs[1]) for _im__drift_roughs in zip(_ims_cy5[:],_drift_roughs[:])]
		_im_mean = np.mean(_ims_signal, 0)
		_im_means+=[_im_mean]
	return _im_means
