import sys,os,re
sys.path.append(r'C:\Users\puzheng\Documents\python-functions\python-functions-library')
import numpy as np
import pickle as pickle
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy
from scipy.signal import fftconvolve
from scipy.ndimage.filters import maximum_filter,minimum_filter,median_filter,gaussian_filter
from ImageAnalysis3 import corrections, visual_tools

_z_scale = 1.8 * 1.5 / 2;
_x_scale = 1.;
_y_scale = 1.;

def partition_map(list_,map_):
    """
    Inputs
    takes a list [e1,e2,e3,e4,e5,e6] and a map (a list of indices [0,0,1,0,1,2]).  map can be a list of symbols too. ['aa','aa','bb','aa','bb','cc']
    Output
    returns a sorted list of lists, e.g. [[e1, e2,e4],[e3,e5],[e6]]
    """
    list__=np.array(list_)
    map__=np.array(map_)
    return [list(list__[map__==element]) for element in np.unique(map__)]

def gauss_ker(sig_xyz=[2,2,2],sxyz=16,xyz_disp=[0,0,0]):
    '''Create a gaussian kernal, return standard gaussian level within sxyz size and sigma 2,2,2'''
    dim = len(xyz_disp)
    xyz=np.indices([sxyz+1]*dim)
    for i in range(len(xyz.shape)-1):
        sig_xyz=np.expand_dims(sig_xyz,axis=-1)
        xyz_disp=np.expand_dims(xyz_disp,axis=-1)
    im_ker = np.exp(-np.sum(((xyz-xyz_disp-sxyz/2.)/sig_xyz**2)**2,axis=0)/2.)
    return im_ker

def add_source(im_,pos=[0,0,0],h=200,sig=[2,2,2]):
    '''Impose a guassian distribution with given position, height and sigma, onto an existing figure'''
    im=np.array(im_,dtype=float)
    pos_int = np.array(pos,dtype=int)
    xyz_disp = -pos_int+pos
    im_ker = gauss_ker(sig_xyz=sig,sxyz=int(np.max(sig)*20),xyz_disp=xyz_disp)
    im_ker_sz = np.array(im_ker.shape,dtype=int)
    pos_min = pos_int-im_ker_sz/2
    pos_max = pos_min+im_ker_sz
    im_shape = np.array(im.shape)
    def in_im(pos__):
        pos_=np.array(pos__,dtype=int)
        pos_[pos_>=im_shape]=im_shape[pos_>=im_shape]-1
        pos_[pos_<0]=0
        return pos_
    pos_min_ = in_im(pos_min)
    pos_max_ = in_im(pos_max)
    pos_min_ker = pos_min_-pos_min
    pos_max_ker = im_ker_sz+pos_max_-pos_max
    #print zip(pos_min_ker,pos_max_ker),zip(pos_min_,pos_max_),zip(pos_min,pos_max)
    slices_ker = [slice(pm,pM)for pm,pM in zip(pos_min_ker,pos_max_ker)]
    slices_im = [slice(pm,pM)for pm,pM in zip(pos_min_,pos_max_)]
    im[slices_im]+=im_ker[slices_ker]*h
    return im

def subtract_source(im,pfit):
    return add_source(im,pos=pfit[1:4],h=-pfit[0],sig=pfit[-3:])

def plus_source(im,pfit):
    return add_source(im,pos=pfit[1:4],h=pfit[0],sig=pfit[-3:])

def sphere(center,radius,imshape=None):
    """Returns an int array (size: n x len(center)) with the xyz... coords of a sphere(elipsoid) of radius in imshape"""
    radius_=np.array(radius,dtype=float)
    if len(radius_.shape)==0:
        radius_ = np.array([radius]*len(center),dtype=np.int)
    xyz = np.array(np.indices(2*radius_+1),dtype=float)
    radius__=np.array(radius_,dtype=float)
    for i in range(len(xyz.shape)-1):
        radius__=np.expand_dims(radius__,axis=-1)
    xyz_keep = np.array(np.where(np.sum((xyz/radius__-1)**2,axis=0)<1))
    xyz_keep = xyz_keep-np.expand_dims(np.array(radius_,dtype=int),axis=-1)+np.expand_dims(np.array(center,dtype=int),axis=-1)
    xyz_keep = xyz_keep.T
    if imshape is not None:
        xyz_keep=xyz_keep[np.all((xyz_keep>=0)&(xyz_keep<np.expand_dims(imshape,axis=0)),axis=-1)]
    return xyz_keep

def grab_block(im,center,block_sizes):
    dims = im.shape
    slices = []
    def in_dim(c,dim):
        c_ = c
        if c_<0: c_=0
        if c_>dim: c_=dim
        return c_
    for c,block,dim in zip(center,block_sizes,dims):
        block_ = int(block/2)
        c=int(c)
        c_min,c_max = in_dim(c-block_,dim),in_dim(c+block-block_,dim)
        slices.append(slice(c_min,c_max))
    slices.append(Ellipsis)
    return im[slices]

# fit single gaussian
def fitsinglegaussian_fixed_width(data,center,radius=10,n_approx=10,width_zxy=[_z_scale,_x_scale,_y_scale]):
    """Returns (height, x, y,z, width_x, width_y,width_z,bk)
    the gaussian parameters of a 2D distribution found by a fit"""
    data_=np.array(data,dtype=float)
    dims = np.array(data_.shape)
    if center is  not None:
        center_z,center_x,center_y = center
    else:
        xyz = np.array(list(map(np.ravel,np.indices(data_.shape))))
        data__=data_[xyz[0],xyz[1],xyz[2]]
        args_high = np.argsort(data__)[-n_approx:]
        center_z,center_x,center_y = np.median(xyz[:,args_high],axis=-1)

    xyz = sphere([center_z,center_x,center_y],radius,imshape=dims).T
    if len(xyz[0])>0:
        data__=data_[xyz[0],xyz[1],xyz[2]]
        sorted_data = np.sort(data__)#np.sort(np.ravel(data__))
        bk = np.median(sorted_data[:n_approx])
        height = (np.median(sorted_data[-n_approx:])-bk)

        width_z,width_x,width_y = np.array(width_zxy)
        params_ = (height,center_z,center_x,center_y,bk)

        def gaussian(height,center_z, center_x, center_y,
                     bk=0,
                     width_z=width_zxy[0],
                     width_x=width_zxy[1],
                     width_y=width_zxy[2]):
            """Returns a gaussian function with the given parameters"""
            width_x_ = np.abs(width_x)
            width_y_ = np.abs(width_y)
            width_z_ = np.abs(width_z)
            height_ = np.abs(height)
            bk_ = np.abs(bk)
            def gauss(z,x,y):
                g = bk_+height_*np.exp(
                    -(((center_z-z)/width_z_)**2+((center_x-x)/width_x_)**2+
                      ((center_y-y)/width_y_)**2)/2.)
                return g
            return gauss
        def errorfunction(p):
            f=gaussian(*p)(*xyz)
            g=data__
            #err=np.ravel(f-g-g*np.log(f/g))
            err=np.ravel(f-g)
            return err
        p, success = scipy.optimize.leastsq(errorfunction, params_)
        p=np.abs(p)
        p = np.concatenate([p,width_zxy])
        #p[:1:4]+=0.5
        return  p,success
    else:
        return None,None
def fit_seed_points_base(im, centers, width_z=_z_scale, width_xy=_x_scale, radius_fit=5, n_max_iter = 10, max_dist_th=0.25):
    '''Basic function used for multiple gaussian fitting, given image:im, seeding_result:centers '''
    print("Fitting:" +str(len(centers[0]))+" points")
    z,x,y = centers # fitting kernels provided by previous seeding
    if len(x)>0:
        #estimate height
        #gfilt_size=0.75
        #filt_size=3
        #im_plt = gaussian_filter(im,gfilt_size)
        #max_filt = maximum_filter(im_plt,filt_size)
        #min_filt = minimum_filter(im_plt,filt_size)
        #h = max_filt[z,x,y]-min_filt[z,x,y]

        #inds = np.argsort(h)[::-1]
        #z,x,y = z[inds],x[inds],y[inds]
        zxy = np.array([z,x,y],dtype=int).T

        ps = []
        im_subtr = np.array(im,dtype=float)
        for center in zxy:
            p,success = fitsinglegaussian_fixed_width(im_subtr,center,radius=radius_fit,n_approx=10,width_zxy=[width_z,width_xy,width_xy])
            if p is not None: # If got any successful fitting, substract fitted profile
                ps.append(p)
                im_subtr = subtract_source(im_subtr,p)

        im_add = np.array(im_subtr)

        max_dist=np.inf
        n_iter = 0
        while max_dist > max_dist_th:
            ps_1=np.array(ps)
            ps_1=ps_1[np.argsort(ps_1[:,0])[::-1]]
            ps = []
            ps_1_rem=[]
            for p_1 in ps_1:
                center = p_1[1:4]
                im_add = plus_source(im_add,p_1)
                p,success = fitsinglegaussian_fixed_width(im_add,center,radius=radius_fit,n_approx=10,width_zxy=[width_z,width_xy,width_xy])
                if p is not None:
                    ps.append(p)
                    ps_1_rem.append(p_1)
                    im_add = subtract_source(im_add,p)
            ps_2=np.array(ps)
            ps_1_rem=np.array(ps_1_rem)
            dif = ps_1_rem[:,1:4]-ps_2[:,1:4]
            max_dist = np.max(np.sum(dif**2,axis=-1))
            n_iter+=1
            if n_iter>n_max_iter:
                break
        return ps_2
    else:
        return np.array([])
def get_seed_points_base(im,gfilt_size=0.75,filt_size=3,th_seed=300.,hot_pix_th=0,return_h=False):
    im_plt = gaussian_filter(im,gfilt_size)
    max_filt = maximum_filter(im_plt,filt_size)
    min_filt = minimum_filter(im_plt,filt_size)
    im_plt2 = (max_filt==im_plt)&(min_filt!=im_plt)
    z,x,y = np.where(im_plt2)
    keep = (max_filt[z,x,y]-min_filt[z,x,y])>th_seed#/np.array(max_filt[z,x,y],dtype=float)>0.5
    x,y,z = x[keep],y[keep],z[keep]
    h = max_filt[z,x,y]-min_filt[z,x,y]

    #get rid of hot pixels
    if hot_pix_th>0:
        xy_str = [str([x_,y_]) for x_,y_ in zip(x,y)]
        xy_str_,cts_ = np.unique(xy_str,return_counts=True)
        keep = np.array([xy_str__ not in xy_str_[cts_>hot_pix_th] for xy_str__ in xy_str],dtype=bool)
        x,y,z = x[keep],y[keep],z[keep]
        h = h[keep]
    centers = np.array([z,x,y])
    if return_h:
        centers = np.array([z,x,y,h])
    return centers

def fit_seed_points_base_fast(im,centers,width_z=_z_scale,width_xy=_x_scale,radius_fit=5,n_max_iter = 10,max_dist_th=0.25, quiet=False):
    if not quiet:
        print("Fitting:" +str(len(centers[0]))+" points")
    z,x,y = centers
    if len(x)>0:

        zxy = np.array([z,x,y],dtype=int).T

        ps = []
        im_subtr = np.array(im,dtype=float)
        for center in zxy:
            p,success = fitsinglegaussian_fixed_width(im_subtr,center,radius=radius_fit,n_approx=5,width_zxy=[width_z,width_xy,width_xy])
            if p is not None:
                ps.append(p)

        im_add = np.array(im_subtr)

        max_dist=np.inf
        n_iter = 0
        while max_dist>max_dist_th:
            ps_1=np.array(ps)
            ps_1=ps_1[np.argsort(ps_1[:,0])[::-1]]
            ps = []
            ps_1_rem=[]
            for p_1 in ps_1:
                center = p_1[1:4]
                p,success = fitsinglegaussian_fixed_width(im_add,center,radius=5,n_approx=10,width_zxy=[1.8,1.,1.])
                if p is not None:
                    ps.append(p)
                    ps_1_rem.append(p_1)
            ps_2=np.array(ps)
            ps_1_rem=np.array(ps_1_rem)
            dif = ps_1_rem[:,1:4]-ps_2[:,1:4]
            max_dist = np.max(np.sum(dif**2,axis=-1))
            n_iter+=1
            if n_iter>n_max_iter:
                break
        return ps_2
    else:
        return np.array([])

def translation_aling_pts(cents_fix,cents_target,cutoff=2.,xyz_res=1,plt_val=False,return_pts=False):
    """
    This checks all pairs of points in cents_target for counterparts of same distance (+/- cutoff) in cents_fix
    and adds them as posibilities. Then uses multi-dimensional histogram across txyz with resolution xyz_res.
    Then it finds nearest neighbours and returns the median txyz_b within resolution.
    """
    from itertools import combinations
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import cdist
    cents = np.array(cents_fix)
    cents_target = np.array(cents_target)
    dists_target = pdist(cents_target)
    dists = pdist(cents_fix)
    all_pairs = np.array(list(combinations(list(range(len(cents))),2)))
    all_pairs_target = np.array(list(combinations(list(range(len(cents_target))),2)))
    #inds_all = np.arange(len(dists))
    txyzs=[]
    for ind_target in range(len(dists_target)):
        keep_cands = np.abs(dists-dists_target[ind_target])<cutoff
        good_pairs = all_pairs[keep_cands][:]
        p1 = cents[good_pairs[:,0]]
        p2 = cents[good_pairs[:,1]]
        p1T = cents_target[all_pairs_target[ind_target,0]]
        p2T = cents_target[all_pairs_target[ind_target,1]]
        txyzs.extend(p1[:]-[p1T])
        txyzs.extend(p1[:]-[p2T])
    bin_txyz = np.array((np.max(txyzs,axis=0)-np.min(txyzs,axis=0))/float(xyz_res),dtype=int)

    hst_res = np.histogramdd(np.array(txyzs),bins=bin_txyz)
    ibest = np.unravel_index(np.argmax(hst_res[0]),hst_res[0].shape)
    txyz_f = [hst[ib]for hst,ib in zip(hst_res[1],ibest)]
    txyz_f = np.array(txyz_f)
    inds_closestT = np.argmin(cdist(cents,cents_target + txyz_f),axis=1)
    inds_closestF=np.arange(len(inds_closestT))
    keep = np.sqrt(np.sum((cents_target[inds_closestT]+ txyz_f-cents[inds_closestF])**2,axis=-1))<2*xyz_res
    inds_closestT=inds_closestT[keep]
    inds_closestF=inds_closestF[keep]

    txyz_b = np.median(cents_target[inds_closestT]-cents[inds_closestF],axis=0)
    if plt_val:
        plt.figure()
        plt.plot(cents[inds_closestF].T[0],cents[inds_closestF].T[1],'go')
        plt.plot(cents_target[inds_closestT].T[0]-txyz_b[0],cents_target[inds_closestT].T[1]-txyz_b[1],'ro')
        plt.figure()
        dists = np.sqrt(np.sum((cents_target[inds_closestT]-cents[inds_closestF])**2,axis=-1))
        plt.hist(dists)
        plt.show()
    if return_pts:
        return txyz_b,cents[inds_closestF],cents_target[inds_closestT]
    return txyz_b

# fast alignment of fitted items which are bright and sparse (like beads)
def beads_alignment_fast(beads, ref_beads, unique_cutoff=2., check_outlier=True, outlier_sigma=1., verbose=True):
    '''beads_alignment_fast, for finding pairs of beads when they are sparse
    Inputs:
        beads: ndarray of beads coordnates, num_beads by [z,x,y], n-by-3 numpy ndarray
        ref_beads: similar coorndiates for beads in reference frame, n-by-3 numpy ndarray
        unique_cutoff: a threshold that assuming there are only unique pairs within it, float
        check_outlier: whether using Delaunay triangulation neighbors to check
        outlier_sigma: times for sigma that determine threshold in checking outlier, positive float
        verbose: whether say something during alignment, bool
    Outputs:
        _paired_beads: beads that find their pairs in ref frame, n-by-3 numpy array
        _paired_ref_beads: ref_beads that find their pairs (sorted), n-by-3 numpy array
        _shifts: 3d shift of beads (bead - ref_bead), n-by-3 numpy array
        '''
    # initialize
    _paired_beads, _paired_ref_beads, _shifts = [], [], []
    # loop through all beads in ref frame
    for _rb in ref_beads:
        _competing_ref_beads = ref_beads[np.sqrt(np.sum((ref_beads - _rb)**2,1)) < unique_cutoff]
        if len(_competing_ref_beads) > 1: # in this case, other ref_bead exist within cutoff
            continue;
        else:
            _candidate_beads = beads[np.sqrt(np.sum((beads - _rb)**2,1)) < unique_cutoff]
            if len(_candidate_beads) == 1: # if unique pairs identified
                _paired_beads.append(_candidate_beads[0])
                _paired_ref_beads.append(_rb);
                _shifts.append(_candidate_beads[0] - _rb)
    # covert to numpy array
    _paired_beads = np.array(_paired_beads)
    _paired_ref_beads = np.array(_paired_ref_beads)
    _shifts = np.array(_shifts)
    # remove suspicious shifts
    for _j in range(_shifts.shape[1]):
        _shift_keeps = np.abs(_shifts)[:,_j] < np.mean(np.abs(_shifts)[:,_j])+outlier_sigma*np.std(np.abs(_shifts)[:,_j])
        # filter beads and shifts
        _paired_beads = _paired_beads[_shift_keeps]
        _paired_ref_beads = _paired_ref_beads[_shift_keeps]
        _shifts = _shifts[_shift_keeps]



    # check outlier
    if check_outlier:
        from scipy.spatial import Delaunay
        from mpl_toolkits.mplot3d import Axes3D
        # initialize list for shifts calculated by neighboring points
        _alter_shifts = [];
        # calculate Delaunay triangulation for ref_beads
        _tri = Delaunay(_paired_ref_beads);
        # loop through all beads
        for _i in range(_paired_ref_beads.shape[0]):
            # initialize diff, which used to judge whether keep this
            _keep = True;
            # extract shift
            _shift = _shifts[_i];
            # initialize neighboring point ids
            _neighbor_ids = []
            # find neighbors for this point
            for _simplex in _tri.simplices.copy():
                if _i in _simplex:
                    _neighbor_ids.append(_simplex);
            _neighbor_ids = np.array(np.unique(_neighbor_ids).astype(np.int));
            _neighbor_ids = _neighbor_ids[_neighbor_ids != _i]; # remove itself
            _neighbor_ids = _neighbor_ids[_neighbor_ids != -1]; # remove error
            # calculate alternative shift
            _neighbors = _paired_ref_beads[_neighbor_ids,:]
            _neighbor_shifts = _shifts[_neighbor_ids,:]
            _neighbor_weights = 1/np.sqrt(np.sum((_neighbors-_paired_ref_beads[_i])**2,1))
            _alter_shift = np.dot(_neighbor_shifts.T, _neighbor_weights) / np.sum(_neighbor_weights)
            _alter_shifts.append(_alter_shift);
            #print _i,  _alter_shift, _shift
        # differences between shifts and alternative shifts
        _diff = [np.linalg.norm(_shift-_alter_shift) for _shift,_alter_shift in zip(_shifts, _alter_shifts)];
        # determine whether keep this:
        print('-- differences in original drift and neighboring dirft:', np.mean(_diff), np.std(_diff))
        _keeps = np.array(_diff < np.mean(_diff)+np.std(_diff)*outlier_sigma, dtype=np.bool)
        # filter beads and shifts
        _paired_beads = _paired_beads[_keeps]
        _paired_ref_beads = _paired_ref_beads[_keeps]
        _shifts = _shifts[_keeps]

    return np.array(_paired_beads), np.array(_paired_ref_beads), np.array(_shifts)


class imshow_mark_3d_v2:
    def master_reset(self):
        #self.dic_min_max = {}
        self.class_ids = []
        self.draw_x,self.draw_y,self.draw_z=[],[],[]
        self.coords = list(zip(self.draw_x,self.draw_y,self.draw_z))
        #load vars
        self.load_coords()
        self.set_image()
    def __init__(self,ims,fig=None,image_names=None,rescz=1.,min_max_default = [None,None],save_file=None,paramaters={}):
        #internalize
        #seeding paramaters
        self.gfilt_size = paramaters.get('gfilt_size',0.75)#first gaussian blur with radius # to avoid false local max from camera fluc
        self.filt_size = paramaters.get('filt_size',3)#local maxima and minima are computed on blocks of size #
        self.th_seed = paramaters.get('th_seed',300.)#keep points when difference between local minima and maxima is more than #
        self.hot_pix_th = paramaters.get('hot_pix_th',0)
        #fitting paramaters
        self.width_z = paramaters.get('width_z',1.8*1.5)#fixed width in z # 1.8 presuposes isotropic pixel size
        self.width_xy = paramaters.get('width_xy',1.)#fixed width in xy
        self.radius_fit = paramaters.get('radius_fit',5)#neibouring of fitting for each seed point

        self.paramaters=paramaters

        self.ims=ims
        self.rescz = rescz
        if image_names is None:
            self.image_names = ['Image '+str(i+1) for i in range(len(ims))]
        else:
            self.image_names = image_names
        self.save_file = save_file
        #define extra vars
        self.dic_min_max = {}
        self.class_ids = []
        self.draw_x,self.draw_y,self.draw_z=[],[],[]
        self.coords = list(zip(self.draw_x,self.draw_y,self.draw_z))
        self.delete_mode = False
        #load vars
        self.load_coords()
        #construct images
        self.index_im = 0
        self.im_ = self.ims[self.index_im]
        self.im_xy = np.max(self.im_,axis=0)
        self.im_z = np.max(self.im_,axis=1)
        im_z_len = self.im_z.shape[0]
        indz=np.array(np.round(np.arange(0,im_z_len,self.rescz)),dtype=int)
        self.im_z = self.im_z[indz[indz<im_z_len],...]
        #setup plots
        if fig is None:
            self.f=plt.figure()
        else:
            self.f=fig
        self.ax1,self.ax2 = ImageGrid(self.f, 111, nrows_ncols=(2, 1), axes_pad=0.1)
        self.lxy,=self.ax1.plot(self.draw_x, self.draw_y, 'o',
                              markersize=12,markeredgewidth=1,markeredgecolor='y',markerfacecolor='None')
        self.lz,=self.ax2.plot(self.draw_x, self.draw_z, 'o',
                      markersize=12,markeredgewidth=1,markeredgecolor='y',markerfacecolor='None')
        self.imshow_xy = self.ax1.imshow(self.im_xy,interpolation='nearest',cmap='gray')
        self.imshow_z = self.ax2.imshow(self.im_z,interpolation='nearest',cmap='gray')

        self.min_,self.max_ = min_max_default
        if self.min_ is None: self.min_ = np.min(self.im_)
        if self.max_ is None: self.max_ = np.max(self.im_)
        self.imshow_xy.set_clim(self.min_,self.max_)
        self.imshow_z.set_clim(self.min_,self.max_)

        self.ax1.callbacks.connect('ylim_changed', self.xy_on_lims_change)
        self.ax2.callbacks.connect('ylim_changed', self.z_on_lims_change)
        self.f.suptitle(self.image_names[self.index_im])
        #connect mouse and keyboard
        cid = self.f.canvas.mpl_connect('button_press_event', self.onclick)
        cid2 = self.f.canvas.mpl_connect('key_press_event', self.press)
        cid3 = self.f.canvas.mpl_connect('key_release_event', self.release)
        self.set_image()
        if fig is None:
            plt.show()
    def onclick(self,event):
        if event.button==3:
            #print "click"
            if event.inaxes is self.ax1:
                if self.delete_mode:
                    z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
                    x_,y_,z_ = list(map(np.array,[self.draw_x,self.draw_y,self.draw_z]))
                    #print x_min,x_max,y_min,y_max,z_min,z_max
                    #print x_,y_,z_
                    keep_in_window = (x_>y_min)&(x_<y_max)&(y_>x_min)&(y_<x_max)&(z_>z_min)&(z_<z_max)
                    keep_class = (np.array(self.class_ids)==self.index_im)&(np.isnan(self.draw_x)==False)
                    keep = keep_in_window&keep_class
                    if np.sum(keep)>0:
                        keep_ind = np.arange(len(keep))[keep]
                        coords_xy_class = list(zip(np.array(self.draw_x)[keep],
                                              np.array(self.draw_y)[keep]))
                        difs = np.array(coords_xy_class)-np.array([[event.xdata,event.ydata]])
                        ind_= np.argmin(np.sum(np.abs(difs),axis=-1))
                        self.draw_x.pop(keep_ind[ind_])
                        self.draw_y.pop(keep_ind[ind_])
                        self.draw_z.pop(keep_ind[ind_])
                        self.class_ids.pop(keep_ind[ind_])
                else:
                    if event.xdata is not None and event.ydata is not None:
                        self.draw_x.append(event.xdata)
                        self.draw_y.append(event.ydata)
                        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
                        self.draw_z.append((z_min+z_max)/2.)
                        self.class_ids.append(self.index_im)
            if event.inaxes is self.ax2:
                if event.xdata is not None and event.ydata is not None:
                    z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
                    x_,y_,z_ = list(map(np.array,[self.draw_x,self.draw_y,self.draw_z]))
                    keep_in_window = (x_>y_min)&(x_<y_max)&(y_>x_min)&(y_<x_max)&(z_>z_min)&(z_<z_max)
                    keep_class = (np.array(self.class_ids)==self.index_im)&(np.isnan(self.draw_x)==False)
                    keep = keep_in_window&keep_class
                    if np.sum(keep)>0:
                        keep_ind = np.arange(len(keep))[keep]
                        coords_x = np.array(self.draw_x)[keep]
                        ind_ = np.argmin(np.abs(coords_x-event.xdata))
                        self.draw_z[keep_ind[ind_]]=event.ydata
            self.update_point_plot()
    def press(self,event):
        if event.key== 'd':
            self.index_im = (self.index_im+1)%len(self.ims)
            self.set_image()
        if event.key== 'a':
            self.index_im = (self.index_im-1)%len(self.ims)
            self.set_image()
        if event.key=='s':
            self.save_ims()
        if event.key== 'x':
            self.auto_scale()
        if event.key== 't':
            self.get_seed_points()
        if event.key== 'n':
            self.handle_in_nucleus()
        if event.key== 'q':
            prev_im = self.index_im
            for self.index_im in range(len(self.ims)):
                self.set_image()
                self.get_seed_points()
                self.fit_seed_points()
            self.index_im = prev_im
            self.set_image()
        if event.key== 'y':
            self.fit_seed_points()
        if event.key == 'delete':
            self.draw_x.pop(-1)
            self.draw_y.pop(-1)
            self.draw_z.pop(-1)
            self.class_ids.pop(-1)
            self.update_point_plot()
        if event.key == 'shift':
            self.delete_mode = True
    def release(self, event):
        if event.key == 'shift':
            self.delete_mode = False
    def populate_draw_xyz(self,flip=False):
        if len(self.coords)>0:
            self.draw_x,self.draw_y,self.draw_z = list(zip(*self.coords))
            if flip: self.draw_x,self.draw_y,self.draw_z =  list(map(list,[self.draw_y,self.draw_x,self.draw_z]))
            else: self.draw_x,self.draw_y,self.draw_z =  list(map(list,[self.draw_x,self.draw_y,self.draw_z]))
        else:
            self.draw_x,self.draw_y,self.draw_z = [],[],[]
    def create_text(self):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        self.texts = []
        i_ims = np.zeros(len(self.ims),dtype=int)
        for (xyz,c_id) in zip(self.coords,self.class_ids):
            i_ims[c_id]+=1
            if c_id==self.index_im:
                if not np.isnan(xyz[0]):
                    if z_min<xyz[2] and z_max>xyz[2] and y_min<xyz[0] and y_max>xyz[0] and x_min<xyz[1] and x_max>xyz[1]:
                        text_ = str(i_ims[c_id])
                        color_='r'
                        if hasattr(self,'dec_text'):
                            key_dec = tuple(list(np.array(xyz,dtype=int))+[c_id])
                            if key_dec in self.dec_text:
                                text_=self.dec_text[key_dec]['text']
                                color_='b'
                        self.texts.append(self.ax1.text(xyz[0],xyz[1],text_,color=color_))
                        self.texts.append(self.ax2.text(xyz[0],xyz[2],text_,color=color_))
    def update_point_plot(self):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()

        self.coords = list(zip(self.draw_x,self.draw_y,self.draw_z))
        x_,y_,z_ = list(map(np.array,[self.draw_x,self.draw_y,self.draw_z]))
        #print x_min,x_max,y_min,y_max,z_min,z_max
        #print x_,y_,z_
        keep_class = np.array(self.class_ids)==self.index_im
        keep_in_window = (x_>y_min)&(x_<y_max)&(y_>x_min)&(y_<x_max)&(z_>z_min)&(z_<z_max)
        keep = keep_class&keep_in_window
        self.lxy.set_xdata(x_[keep])
        self.lxy.set_ydata(y_[keep])
        self.lz.set_xdata(x_[keep])
        self.lz.set_ydata(z_[keep])
        self.save_coords()
        self.remove_text()
        self.create_text()
        self.f.canvas.draw()
    def remove_text(self):
        if not hasattr(self,'texts'): self.texts = []
        for txt in self.texts:
            txt.remove()
    def load_coords(self):
        save_file = self.save_file
        if save_file is not None and os.path.exists(save_file):
            fid = open(save_file,'r')
            save_dic = pickle.load(fid, encoding='latin1')
            self.coords,self.class_ids = save_dic['coords'],save_dic['class_ids']
            if 'pfits' in save_dic:
                self.pfits_save = save_dic['pfits']
            if 'dec_text' in save_dic:
                self.dec_text=save_dic['dec_text']
            fid.close()
            self.populate_draw_xyz()#coords to plot list
    def save_coords(self):
        save_file = self.save_file
        if save_file is not None:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            fid = open(save_file,'w')
            self.pfits_save = getattr(self,'pfits_save',{})
            self.dec_text = getattr(self,'dec_text',{})
            save_dic = {'coords':self.coords,'class_ids':self.class_ids,'pfits':self.pfits_save,'dec_text':self.dec_text}
            pickle.dump(save_dic,fid)
            fid.close()
    def auto_scale(self):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        im_chop = self.im_[z_min:z_max,x_min:x_max,y_min:y_max,...]
        min_,max_ = np.min(im_chop),np.max(im_chop)
        self.imshow_xy.set_clim(min_,max_)
        self.imshow_z.set_clim(min_,max_)
        self.dic_min_max[self.index_im] = [min_,max_]
        self.f.canvas.draw()
    def del_ext(self,str_):
        "Deletes extention"
        if os.path.basename(str_).count('.')>0:
            return '.'.join(str_.split('.')[:-1])
        else:
            return str_
    def save_ims(self):
        import scipy.misc
        save_file = self.save_file
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        for index_im,im_ in enumerate(self.ims):
            im_chop = im_[self.get_z_ind(),x_min:x_max,y_min:y_max,...]
            im_xy = np.max(im_chop,axis=0)
            im_z = np.max(im_chop,axis=1)

            if index_im in self.dic_min_max:
                min_,max_ = self.dic_min_max[index_im]
                im_xy = minmax(im_xy,min_=min_,max_=max_)
                im_z = minmax(im_z,min_=min_,max_=max_)
            else:
                min_,max_ = self.min_,self.max_
                im_xy = minmax(im_xy,min_=min_,max_=max_)
                im_z = minmax(im_z,min_=min_,max_=max_)
            if save_file is not None:
                if not os.path.exists(os.path.dirname(save_file)):
                    os.makedirs(os.path.dirname(save_file))
                save_image = self.del_ext(save_file)+'_'+self.image_names[index_im]
                scipy.misc.imsave(save_image+'_xy.png', im_xy)
                scipy.misc.imsave(save_image+'_z.png', im_z)

    def set_image(self):
        self.im_ = self.ims[self.index_im]
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        self.im_sm = self.im_[z_min:z_max,x_min:x_max,y_min:y_max]
        self.im_xy = np.max(self.im_[z_min:z_max,:,...],axis=0)
        self.imshow_xy.set_data(self.im_xy)

        self.im_z = np.max(self.im_[:,x_min:x_max,...],axis=1)
        self.im_z = self.im_z[self.get_z_ind(),:]
        self.imshow_z.set_data(self.im_z)

        if self.index_im in self.dic_min_max:
            min_,max_ = self.dic_min_max[self.index_im]
            self.imshow_xy.set_clim(min_,max_)
            self.imshow_z.set_clim(min_,max_)
        self.update_point_plot()
        self.f.suptitle(self.image_names[self.index_im])
        self.f.canvas.draw()
    def get_limits(self):
        y_min,y_max = self.ax1.get_xlim()
        x_min,x_max = self.ax1.get_ylim()[::-1]
        x_min = max(int(x_min),0)
        x_max = min(int(x_max),self.im_.shape[1])
        y_min = max(int(y_min),0)
        y_max = min(int(y_max),self.im_.shape[2])

        z_min,z_max = np.array(self.ax2.get_ylim()[::-1])*self.rescz
        z_min = max(int(z_min),0)
        z_max = min(int(z_max),self.im_.shape[0])
        return z_min,z_max,x_min,x_max,y_min,y_max
    def get_z_ind(self):
        im_z_len = self.im_z.shape[0]
        indz=np.array(np.round(np.arange(0,im_z_len,self.rescz)),dtype=int)
        return indz[indz<im_z_len]
    def xy_on_lims_change(self,ax):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        self.im_sm = self.im_[z_min:z_max,x_min:x_max,y_min:y_max]
        self.im_z = np.max(self.im_[:,x_min:x_max,...],axis=1)
        self.im_z = self.im_z[self.get_z_ind(),:]
        self.imshow_z.set_data(self.im_z)
        self.update_point_plot()
    def z_on_lims_change(self,ax):
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        self.im_sm = self.im_[z_min:z_max,x_min:x_max,y_min:y_max]
        self.im_xy = np.max(self.im_[z_min:z_max,:,...],axis=0)
        self.imshow_xy.set_data(self.im_xy)
        self.update_point_plot()
    def fit_seed_points(self):
        #get default paramaters from self
        width_z = self.width_z
        width_xy = self.width_xy
        radius_fit = self.radius_fit
        im = self.im_sm
        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        y_,x_,z_ = list(map(np.array,[self.draw_x,self.draw_y,self.draw_z]))
        keep_class = np.array(self.class_ids)==self.index_im
        keep_in_window = (x_>x_min)&(x_<x_max)&(y_>y_min)&(y_<y_max)&(z_>z_min)&(z_<z_max)
        keep = keep_class&keep_in_window
        xyzguess = np.array([z_[keep]-z_min,x_[keep]-x_min,y_[keep]-y_min],dtype=int)

        self.pfits = fit_seed_points_base(im,xyzguess,width_z=width_z,width_xy=width_xy,radius_fit=3,n_max_iter = 15,max_dist_th=0.25)
        if len(self.pfits>0):
            self.pfits[:,1:4]+=[[z_min,x_min,y_min]]
            #update graph and points
            keep = np.array(self.class_ids)!=self.index_im
            self.class_ids,self.draw_z,self.draw_x,self.draw_y = [list(np.array(x)[keep]) for x in [self.class_ids,self.draw_z,self.draw_x,self.draw_y]]
            if not hasattr(self,'pfits_save'):
                self.pfits_save={}
            self.pfits_save[self.index_im]=self.pfits
            centers_0,centers_1,centers_2 = self.pfits[:,1:4].T
            self.draw_z.extend(centers_0)
            self.draw_x.extend(centers_2)
            self.draw_y.extend(centers_1)
            self.class_ids.extend([self.index_im]*len(centers_0))
        self.update_point_plot()
    def get_seed_points(self):
        #get default paramaters from self
        gfilt_size = self.gfilt_size
        filt_size = self.filt_size
        th_seed = self.th_seed
        hot_pix_th = self.hot_pix_th

        im = self.im_sm

        centers = get_seed_points_base(im,gfilt_size=gfilt_size,filt_size=filt_size,th_seed=th_seed,hot_pix_th=hot_pix_th)

        z_min,z_max,x_min,x_max,y_min,y_max = self.get_limits()
        keep = np.array(self.class_ids)!=self.index_im
        self.class_ids,self.draw_z,self.draw_x,self.draw_y = [list(np.array(x)[keep]) for x in [self.class_ids,self.draw_z,self.draw_x,self.draw_y]]
        self.draw_z.extend(centers[0]+z_min)
        self.draw_x.extend(centers[2]+y_min)
        self.draw_y.extend(centers[1]+x_min)
        self.class_ids.extend([self.index_im]*len(centers[0]))
        self.update_point_plot()
    def handle_in_nucleus(self):
        if hasattr(self,'nucl_x'):
            i_im = self.index_im
            class_ids = np.array(self.class_ids)
            Y,X,Z = np.array(self.draw_x,dtype=int),np.array(self.draw_y,dtype=int),np.array(self.draw_z,dtype=int)
            keep = class_ids==i_im
            Y,X,Z=Y[keep],X[keep],Z[keep]
            nucl_ = np.array([self.nucl_x,self.nucl_y,self.nucl_z],dtype=int).T
            draw_x,draw_y,draw_z=[],[],[]
            for x,y,z in zip(X,Y,Z):
                if np.any(np.sum(np.abs(nucl_-[[x,y,z]]),axis=-1)==0):
                    draw_z.append(z)
                    draw_x.append(y)
                    draw_y.append(x)
            keep = np.array(self.class_ids)!=self.index_im
            self.class_ids,self.draw_z,self.draw_x,self.draw_y = [list(np.array(x)[keep]) for x in [self.class_ids,self.draw_z,self.draw_x,self.draw_y]]
            self.draw_z.extend(draw_z)
            self.draw_x.extend(draw_x)
            self.draw_y.extend(draw_y)
            self.class_ids.extend([self.index_im]*len(draw_x))
            self.update_point_plot()

class Reader:

    # Close the file on cleanup.
    def __del__(self):
        if self.fileptr:
            self.fileptr.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        if self.fileptr:
            self.fileptr.close()

    # Average multiple frames in a movie.
    def averageFrames(self, start = False, end = False, verbose = False):
        if (not start):
            start = 0
        if (not end):
            end = self.number_frames

        length = end - start
        average = np.zeros((self.image_width, self.image_height), np.float)
        for i in range(length):
            if verbose and ((i%10)==0):
                print(" processing frame:", i, " of", self.number_frames)
            average += self.loadAFrame(i + start)

        average = average/float(length)
        return average

    # returns the film name
    def filmFilename(self):
        return self.filename

    # returns the film size
    def filmSize(self):
        return [self.image_width, self.image_height, self.number_frames]

    # returns the picture x,y location, if available
    def filmLocation(self):
        if hasattr(self, "stage_x"):
            return [self.stage_x, self.stage_y]
        else:
            return [0.0, 0.0]

    # returns the film focus lock target
    def lockTarget(self):
        if hasattr(self, "lock_target"):
            return self.lock_target
        else:
            return 0.0

    # returns the scale used to display the film when
    # the picture was taken.
    def filmScale(self):
        if hasattr(self, "scalemin") and hasattr(self, "scalemax"):
            return [self.scalemin, self.scalemax]
        else:
            return [100, 2000]

# Dax reader class. This is a Zhuang lab custom format.
#

class DaxReader(Reader):
    # dax specific initialization
    def __init__(self, filename, verbose = 0):
        import os,re
        # save the filenames
        self.filename = filename
        dirname = os.path.dirname(filename)
        if (len(dirname) > 0):
            dirname = dirname + "/"
        self.inf_filename = dirname + os.path.splitext(os.path.basename(filename))[0] + ".inf"

        # defaults
        self.image_height = None
        self.image_width = None

        # extract the movie information from the associated inf file
        size_re = re.compile(r'frame dimensions = ([\d]+) x ([\d]+)')
        length_re = re.compile(r'number of frames = ([\d]+)')
        endian_re = re.compile(r' (big|little) endian')
        stagex_re = re.compile(r'Stage X = ([\d\.\-]+)')
        stagey_re = re.compile(r'Stage Y = ([\d\.\-]+)')
        lock_target_re = re.compile(r'Lock Target = ([\d\.\-]+)')
        scalemax_re = re.compile(r'scalemax = ([\d\.\-]+)')
        scalemin_re = re.compile(r'scalemin = ([\d\.\-]+)')

        inf_file = open(self.inf_filename, "r")
        while 1:
            line = inf_file.readline()
            if not line: break
            m = size_re.match(line)
            if m:
                self.image_height = int(m.group(1))
                self.image_width = int(m.group(2))
            m = length_re.match(line)
            if m:
                self.number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    self.bigendian = 1
                else:
                    self.bigendian = 0
            m = stagex_re.match(line)
            if m:
                self.stage_x = float(m.group(1))
            m = stagey_re.match(line)
            if m:
                self.stage_y = float(m.group(1))
            m = lock_target_re.match(line)
            if m:
                self.lock_target = float(m.group(1))
            m = scalemax_re.match(line)
            if m:
                self.scalemax = int(m.group(1))
            m = scalemin_re.match(line)
            if m:
                self.scalemin = int(m.group(1))

        inf_file.close()

        # set defaults, probably correct, but warn the user
        # that they couldn't be determined from the inf file.
        if not self.image_height:
            print("Could not determine image size, assuming 256x256.")
            self.image_height = 256
            self.image_width = 256

        # open the dax file
        if os.path.exists(filename):
            self.fileptr = open(filename, "rb")
        else:
            self.fileptr = 0
            if verbose:
                print("dax data not found", filename)

    # Create and return a memory map the dax file
    def loadMap(self):
        if os.path.exists(self.filename):
            if self.bigendian:
                self.image_map = np.memmap(self.filename, dtype='>u2', mode='r', shape=(self.number_frames,self.image_width, self.image_height))
            else:
                self.image_map = np.memmap(self.filename, dtype='uint16', mode='r', shape=(self.number_frames,self.image_width, self.image_height))
        return self.image_map

    # load a frame & return it as a np array
    def loadAFrame(self, frame_number):
        if self.fileptr:
            assert frame_number >= 0, "frame_number must be greater than or equal to 0"
            assert frame_number < self.number_frames, "frame number must be less than " + str(self.number_frames)
            self.fileptr.seek(frame_number * self.image_height * self.image_width * 2)
            image_data = np.fromfile(self.fileptr, dtype='uint16', count = self.image_height * self.image_width)
            image_data = np.transpose(np.reshape(image_data, [self.image_width, self.image_height]))
            if self.bigendian:
                image_data.byteswap(True)
            return image_data
    # load full movie and retun it as a np array
    def loadAll(self):
        image_data = np.fromfile(self.fileptr, dtype='uint16', count = -1)
        image_data = np.swapaxes(np.reshape(image_data, [self.number_frames,self.image_width, self.image_height]),1,2)
        if self.bigendian:
            image_data.byteswap(True)
        return image_data

## segmentation with DAPI
def DAPI_thresholding_segmentation(ims, names,
                                  cap_percentile=0.5,
                                  image3D=False,
                                  illumination_correction=True,
                                  denoise='bilateral',
                                  obj_size_limit=1250,
                                  threshold_percentile=88,
                                  remove_fov_boundary=40,
                                  segment_method='random_walker',
                                  make_plot=False, verbose=True):
    '''cell segmentation for DAPI images
    Inputs:
        ims: list of images
        names: list of names, same length as ims
        cap_percentile: removing top and bottom percentile in each image, float from 0-100
        image3D: whether process images with 3D or 2D, bool
        illumination_correction: whether correct illumination for each field of view, bool
        denoise: method of denoise, 'bilateral'/'gaussian'/'None'
        obj_size_limit: smallest object size allowed as nucleus, int (default:1000 for 2d, 35000 for 3D)
        threshold_percentile: percentile for thresholding, float from 0-100
        remove_fov_boundary: if certain label is too close to fov boundary, remove, float
        segmentation_method: method of segmentation, 'random_walker'/'watershed'
        make_plot: whether making plots for checking purpose, bool
        verbose: whether say something during the process, bool
    Output:
        _seg_labels: list of labels, same dimension as ims, list of bool matrix
        '''
    # check whether input is a list of images or just one image
    if isinstance(ims, list):
        if verbose:
            print("Start segmenting list of images");
        _ims = ims;
        _names = names;
    else:
        if verbose:
            print("Start segmenting one image");
        _ims = [ims];
        _names = [names];

    # check input length
    if len(_names) != len(_ims):
        raise ValueError('input images and names length not compatible!');

    # 2D or 3D:
    if not image3D:
        if verbose:
            print("Converting images to 2D");
        _ims = [np.mean(_im,0).astype(np.float) for _im in _ims]; # do z-stack if not 3D
        print(_ims[0].shape)

    # illumination correction
    if illumination_correction:
        #import corrections
        from ImageAnalysis3.corrections import Illumination_correction
        correction_channel = 405 # DAPI is in 405 channel
        if verbose:
            print("Correcting illuminations");
        _ims = Illumination_correction(_ims, correction_channel);

    # normalize image
    from scipy import stats
    _limits = [stats.scoreatpercentile(_im, (cap_percentile, 100.-cap_percentile)).astype(np.float) for _im in _ims];
    _norm_ims = [(_im-np.min(_limit))/(np.max(_limit)-np.min(_limit)) for _im,_limit in zip(_ims, _limits)]
    for _im in _norm_ims: # cap image
        _im[_im > 1] = 1;
        _im[_im < 0] = 0;
    if verbose:
        print("Normalize image intensities to real number within 0-1");

    # denoise
    if denoise:
        from skimage import restoration
        if verbose:
            print("Denoising images");
        if denoise == 'bilateral':
            if verbose:
                print("- denoising with bilateral")
            _norm_ims = [restoration.denoise_bilateral(_im, multichannel=image3D) for _im in _norm_ims]
        if denoise == 'wavelet':
            if verbose:
                print("- denoising with wavelet")
            _norm_ims = [restoration.denoise_nl_means(_im, multichannel=False) for _im in _norm_ims]

    # dialate and erosion
    from scipy import ndimage
    from skimage import morphology
    if verbose:
        print("Dialation and erosion for images");
        print("- shape of image:",_norm_ims[0].shape)
    if image3D:
        _dialation_ims = [ndimage.grey_dilation(_im, structure=morphology.ball(3)) for _im in _norm_ims];
        _erosion_ims = [ndimage.grey_erosion(_im,structure=morphology.ball(4)) for _im in _dialation_ims];
    else:
        _dialation_ims = [ndimage.grey_dilation(_im, structure=morphology.disk(3)) for _im in _norm_ims];
        _erosion_ims = [ndimage.grey_erosion(_im,structure=morphology.disk(4)) for _im in _dialation_ims];
    if make_plot:
        if image3D:
            imshow_mark_3d_v2(_dialation_ims+_erosion_ims, image_names=_names+_names);
        else:
            plt.figure();
            plt.imshow(_dialation_ims[0]); plt.colorbar();plt.show();
            plt.figure();
            plt.imshow(_erosion_ims[0]); plt.colorbar();plt.show();

    # thresholding and filling holes
    _bims = [_im > stats.scoreatpercentile(_im, (threshold_percentile)) for _im in _erosion_ims];
    _bims = [ndimage.binary_fill_holes(_bim) for _bim in _bims];

    # acquiring contours
    if image3D:
        _open_objects = [morphology.opening(_bim, morphology.ball(3)) for _bim in _bims];
        _close_objects = [morphology.closing(_open, morphology.ball(3)) for _open in _open_objects]
        _close_objects = [morphology.remove_small_objects(_close, obj_size_limit*33) for _close in _close_objects];
    else:
        _open_objects = [morphology.opening(_bim, morphology.disk(3)) for _bim in _bims];
        _close_objects = [morphology.closing(_open, morphology.disk(3)) for _open in _open_objects]
        _close_objects = [morphology.remove_small_objects(_close, obj_size_limit) for _close in _close_objects];

    # labelling for erosion images, used for further segmentation
    _bboxes = [ndimage.find_objects(_close) for _close in _close_objects];
    _masks = [_close[_bbox[0]] for _bbox, _close in zip(_bboxes, _close_objects)];
    _roi_images = [_im[_bbox[0]] for _bbox, _im in zip(_bboxes, _norm_ims)];
    if make_plot and not image3D:
        for _roi, _mask in zip(_roi_images, _masks):
            plt.figure()
            plt.imshow(_roi, cmap='gray'); plt.colorbar();
            plt.contour(_mask);
            plt.show()
    _labels, _nums = [],[]; # initialize labels
    for _close in _close_objects:
        _label, _num = ndimage.label(_close);
        # remove labels too close to field of view boundary
        for i in range(_num-1):
            _center = np.round(ndimage.measurements.center_of_mass(_label==i+1));
            if _center[-1] < remove_fov_boundary or _center[-1] > _label.shape[-1] - remove_fov_boundary or _center[-2] < remove_fov_boundary or _center[-2] > _label.shape[-2] - remove_fov_boundary:
                _label[_label==i+1] = -1;
                _label[_label > i+1] -= 1;
        _label[_label==0] = -1;
        _labels.append(_label);
        _nums.append(_num);

    # Segmentation, using previous labelling as seed
    if segment_method == 'random_walker':
        if verbose:
            print("Segmentation by random_walker");
        from skimage.segmentation import random_walker
        _seg_labels = [random_walker(_im, _label, beta=1000, mode='bf') for _im, _label in zip(_norm_ims, _labels)];
    elif segment_method == 'watershed':
        if verbose:
            print("Segmentation by watershed");
        _seg_labels = [ndimage.watershed_ift(_im.astype(np.uint8), _label.astype(np.int)) for _im, _label in zip(_norm_ims, _labels)];

    if make_plot:
        if image3D:
            imshow_mark_3d_v2(_seg_labels,image_names=_names);
        else:
            for _seg_label in _seg_labels:
                plt.figure();
                plt.imshow(_seg_label)
                plt.colorbar();plt.show();

    # return segmentation results
    return _seg_labels;

# segmentation with convolution of DAPI images
def DAPI_convolution_segmentation(ims, names,
                                  cap_percentile=0.5,
                                  image3D=False,
                                  illumination_correction=True,
                                  denoise='bilateral',
                                  zoom_ratio = 8,
                                  conv_kernel_dims = [2,1],
                                  conv_threshold = 0.1,
                                  obj_size_limit=1300,
                                  total_size_limit = 275,
                                  remove_fov_boundary = 6,
                                  segment_method='random_walker',
                                  make_plot=False, verbose=True):
    '''cell segmentation for DAPI images with pooling and convolution layers
    Inputs:
        ims: list of images
        names: list of names, same length as ims
        cap_percentile: removing top and bottom percentile in each image, float from 0-100
        image3D: whether process images with 3D or 2D, bool
        illumination_correction: whether correct illumination for each field of view, bool
        denoise: method of denoise, 'bilateral'/'gaussian'/'None'
        zoom_ratio: ratio to zoom out (as pooling layer), float > 1
        conv_kernel: convolution laplacial-like kernel with [inner_dim, outer_dim], array-like
        obj_size_limit: smallest object size allowed as nucleus, int (default:1300 for 2d, 35000 for 3D)
        total_size_limit: upper limit for all dimensions, int (default: 300 for 2d, 350 for 3D)
        remove_fov_boundary: if certain label is too close to fov boundary, remove, float
        segmentation_method: method of segmentation, 'random_walker'/'watershed'
        make_plot: whether making plots for checking purpose, bool
        verbose: whether say something during the process, bool
    Output:
        _seg_labels: list of labels, same dimension as ims, list of bool matrix
        '''
    # Imports
    from scipy import ndimage
    from skimage import morphology
    from scipy import stats
    from skimage import restoration
    from ImageAnalysis3.corrections import Illumination_correction

    # check whether input is a list of images or just one image
    if isinstance(ims, list):
        if verbose:
            print("Start segmenting list of images");
        _ims = ims;
        _names = names;
    else:
        if verbose:
            print("Start segmenting one image");
        _ims = [ims];
        _names = [names];

    # check input length
    if len(_names) != len(_ims):
        raise ValueError('input images and names length not compatible!');

    # 2D or 3D:
    if not image3D:
        if verbose:
            print("Converting images to 2D");
        _ims = [np.mean(_im,0).astype(np.float) for _im in _ims]; # do z-stack if not 3D
        print(_ims[0].shape)

    # illumination correction
    if illumination_correction:
        #import corrections
        correction_channel = 405 # DAPI is in 405 channel
        if verbose:
            print("Correcting illuminations");
        _ims = Illumination_correction(_ims, correction_channel);

    # normalize image
    _limits = [stats.scoreatpercentile(_im, (cap_percentile, 100.-cap_percentile)).astype(np.float) for _im in _ims];
    _norm_ims = [(_im-np.min(_limit))/(np.max(_limit)-np.min(_limit)) for _im,_limit in zip(_ims, _limits)]
    for _im in _norm_ims: # cap image
        _im[_im > 1] = 1;
        _im[_im < 0] = 0;
    if verbose:
        print("Normalize image intensities to real number within 0-1");

    # denoise
    if denoise:
        if verbose:
            print("Denoising images");
        if denoise == 'bilateral':
            if verbose:
                print("- denoising with bilateral")
            _norm_ims = [restoration.denoise_bilateral(_im, multichannel=image3D) for _im in _norm_ims]
        if denoise == 'wavelet':
            if verbose:
                print("- denoising with wavelet")
            _norm_ims = [restoration.denoise_nl_means(_im, multichannel=False) for _im in _norm_ims]

    # zoom out (pooling layer)
    if verbose:
        print("Pooling images");
    _zims = [restoration.denoise_nl_means(ndimage.zoom(_norm_im, 1./zoom_ratio)) for _norm_im in _norm_ims];

    ## convolution
    if verbose:
        print("Convolution with pooled images")
    _inner, _outer = conv_kernel_dims[0], conv_kernel_dims[-1];
    _conv_zims = []; # initialize
    if image3D: # if 3D:
        _kernel = np.zeros([_inner+2*_outer]*(2+int(image3D)))
        _kernel[:,_outer:_outer+_inner,_outer:_outer+_inner] = -1;
        _kernel[_outer:_outer+_inner,:,_outer:_outer+_inner] = -1;
        _kernel[_outer:_outer+_inner,_outer:_outer+_inner,:] = -1;
        _kernel[_outer:_outer+_inner,_outer:_outer+_inner]= (2+int(image3D)) * 2 * _outer / float(_inner)
        # convolution
        for _zim in _zims:
            _conv_zim = ndimage.convolve(_zim, _kernel)
            _conv_zim[0,:,:] = 0;
            _conv_zim[-1,:,:] = 0;
            _conv_zim[:,0,:] = 0;
            _conv_zim[:,-1,:] = 0;
            _conv_zim[:,:,0] = 0;
            _conv_zim[:,:,-1] = 0;
            _conv_zims.append(_conv_zim);
    else: # if 2D
        _kernel = np.zeros([_inner+2*_outer]*2)
        _kernel[_outer:_outer+_inner,:] = -1;
        _kernel[:,_outer:_outer+_inner] = -1;
        _kernel[_outer:_outer+_inner,_outer:_outer+_inner]= 4 * _outer / float(_inner)
        # convolution
        for _zim in _zims:
            _conv_zim = ndimage.convolve(_zim, _kernel)
            _conv_zim[0,:] = 0;
            _conv_zim[-1,:] = 0;
            _conv_zim[:,0] = 0;
            _conv_zim[:,-1] = 0;
            _conv_zims.append(_conv_zim);
    # thresholding convoluted images to get seeds
    if verbose:
        print("Thresholding convoluted images")
    _bims = [ndimage.zoom(_conv_zim, zoom_ratio) for _conv_zim in _conv_zims] # zoom in again (reverse pooling)
    _bims = [_bim > conv_threshold for _bim in _bims] # thresholding convoluted images
    if image3D:
        _bims = [ndimage.binary_fill_holes(_bim, structure=morphology.ball(2)) for _bim in _bims];
    else:
        _bims = [ndimage.binary_fill_holes(_bim, structure=morphology.disk(2)) for _bim in _bims];


    # acquiring contours
    if image3D:
        _open_objects = [morphology.opening(_bim, morphology.ball(3)) for _bim in _bims];
        _close_objects = [morphology.closing(_open, morphology.ball(3)) for _open in _open_objects]
        _close_objects = [morphology.remove_small_objects(_close, obj_size_limit*32) for _close in _close_objects];
    else:
        _open_objects = [morphology.opening(_bim, morphology.disk(3)) for _bim in _bims];
        _close_objects = [morphology.closing(_open, morphology.disk(3)) for _open in _open_objects]
        _close_objects = [morphology.remove_small_objects(_close, obj_size_limit) for _close in _close_objects];

    # labelling for erosion images, used for further segmentation
    _bboxes = [ndimage.find_objects(_close) for _close in _close_objects];
    _masks = [_close[_bbox[0]] for _bbox, _close in zip(_bboxes, _close_objects)];
    _roi_images = [_im[_bbox[0]] for _bbox, _im in zip(_bboxes, _norm_ims)];
    if make_plot and not image3D:
        for _roi, _mask,_name in zip(_roi_images, _masks, _names):
            plt.figure()
            plt.imshow(_roi, cmap='gray'); plt.colorbar();
            plt.contour(_mask);
            plt.title(_name)
            plt.show()
    _labels, _nums = [],[]; # initialize labels
    for _close,_name in zip(_close_objects, _names):
        _label, _num = ndimage.label(_close);
        _label[_label==0] = -1;
        # remove labels too close to field of view boundary
        i = 1;
        while i < np.max(_label):
            _total_size = 0; # initialzie total size
            for _d in range(2+int(image3D)):
                _total_size += np.sum(np.sum(_label == i ,_d)>0)
            _center = np.round(ndimage.measurements.center_of_mass(_label==i));
            for _coord,_dim in zip(_center[-2:], _label.shape[-2:]):
                if _coord < remove_fov_boundary*zoom_ratio or _coord > _dim - remove_fov_boundary*zoom_ratio:
                    print("-- remove a label near boundary in", _name, i)
                    _label[_label == i] = -1;
                    _label[_label > i] -= 1;
                    break;
                elif _total_size > total_size_limit + 50 * int(image3D):
                    print("-- remove a label with total size: "+str(_total_size) + ', '+str(i), 'in', _name)
                    _label[_label == i] = -1;
                    _label[_label > i] -= 1;
                    break
            i += 1;
        _labels.append(_label);
        _nums.append(i);

    # Segmentation, using previous labelling as seed
    if segment_method == 'random_walker':
        if verbose:
            print("Segmentation by random_walker");
        from skimage.segmentation import random_walker
        _seg_labels = [random_walker(_im, _label, beta=1000, mode='bf') for _im, _label in zip(_ims, _labels)];
    elif segment_method == 'watershed':
        if verbose:
            print("Segmentation by watershed");
        _seg_labels = [ndimage.watershed_ift(_im.astype(np.uint8), _label.astype(np.int)) for _im, _label in zip(_ims, _labels)];

    if make_plot:
        if image3D:
            imshow_mark_3d_v2(_seg_labels,image_names=_names);
        else:
            for _seg_label, _name in zip(_seg_labels, _names):
                plt.figure();
                plt.imshow(_seg_label)
                plt.title(_name)
                plt.colorbar();plt.show();

    # return segmentation results
    return _seg_labels;

# merge images to generate "chromosome"
def generate_chromosome_from_dic(im_dic, merging_channel, color_dic,  bead_label='beads',
                                 merge_num=10, ref_frame=0, fft_dim=125, verbose=True):
    '''Function to generate "chromosomes" by merging first several regions
    Given:
        im_dic: dictionary of images loaded by get_img_info.split_channels_by_image, dic
        merging_channel: use which channel to merge as chromosome, -1 means all channels except beads, int
        color_dic: dictionary of color usage loaded by get_img_info.Load_Color_Usage, dic
        merge_num: number of images to be merged, int (default: 10)
        ref_frame: which frame is used as reference, non-negative int (default: 0)
        fft_dim: dimension for FFT, positive int (default: 125)
        verbose: say something!, bool (default: True)
    Return:
        _mean_im: merged image, 3d-array
        _rough_dfts: drifts calculated by FFT, list of 1d-arrays
    '''
    import numpy as np;
    import os
    from ImageAnalysis3.corrections import fast_translate, fftalign
    # initialize mean_image as chromosome
    _mean_im=[]
    _rough_dfts = [];
    # get ref frame
    _ref_name = sorted(list(im_dic.items()), key=lambda k_v: int(k_v[0].split('H')[1].split('R')[0]))[ref_frame][0]
    _ref_ims = sorted(list(im_dic.items()), key=lambda k_v1: int(k_v1[0].split('H')[1].split('R')[0]))[ref_frame][1]
    if bead_label not in color_dic[_ref_name.split(os.sep)[0]]:
        raise ValueError('wrong ref frame, no beads exist in this hybe.')
    for _i, _label in enumerate(color_dic[_ref_name.split(os.sep)[0]]):
        # check bead label
        if bead_label == _label:
            _ref_bead = _ref_ims[_i]
            break;
    # loop through all images for this field of view
    for _name, _ims in sorted(list(im_dic.items()), key=lambda k_v2: int(k_v2[0].split('H')[1].split('R')[0])):
        if len(_rough_dfts) >= merge_num: # stop if images more than merge_num are already calclulated.
            break;
        if _name == _ref_name: # pass the ref frame
            continue
        if bead_label in color_dic[_name.split(os.sep)[0]]:
            #if verbose:
            #    print "processing image:", _name
            # extract bead image
            for _i, _label in enumerate(color_dic[_name.split(os.sep)[0]]):
                # check bead label
                if bead_label == _label:
                    _bead = _ims[_i]
                    break;
            # calculate drift fastly with FFT
            _rough_dft = fftalign(_ref_bead, _bead);
            _rough_dfts.append(_rough_dft)
            # roughly align image and save
            if merging_channel >=0 and merging_channel < len(_ims): # if merging_channel is provided properly
                _corr_im = translate(_ims[merging_channel],-_rough_dft);
                _mean_im.append(_corr_im);
            else: # if merging_channel is not provided etc:
                for _i, _label in enumerate(color_dic[_name.split(os.sep)[0]]):
                    if bead_label != _label and _label != '':
                        _corr_im = fast_translate(_ims[_i],-_rough_dft);
                        _mean_im.append(_corr_im)
    if verbose:
        print('- number of images to calculate mean: '+str(len(_mean_im))+'\n- number of FFT drift corrections: '+str(len(_rough_dfts)))
        print("- drifts are: \n", _rough_dfts)
    _mean_im = np.mean(_mean_im,0);

    return _mean_im, _rough_dfts

# crop cells based on DAPI segmentation result
def crop_cell(im, segmentation_label, drift=None, extend_dim=20, overlap_threshold = 0.1, verbose=True):
    '''basic function to crop image into small ones according to segmentation_label
    Inputs:
        im: single nd-image, numpy.ndarray
        segmentation_label: 2D or 3D segmentaiton label, each cell has unique id, numpy.ndarray (if None, no drift applied)
        drift: whether applying drift to the cropping, should be relative drift to frame with DAPI, 1darray (default: None)
        extend_dim: dimension that expand for cropping, int (default: 30)
        overlap_threshold: upper limit of how much the cropped image include other labels, float<1 (default: 0.1)
        verbose: say something during processing!, bool (default: True)
    Outputs:
        _crop_ims: list of images that has been cropped
    '''
    # imports
    from scipy.ndimage.interpolation import shift
    # check dimension
    _im_dim = np.shape(im);
    _label_dim = np.shape(segmentation_label)
    if drift is not None:
        if len(drift) != len(im.shape):
            raise ValueError('drift dimension and image dimension doesnt match!');
    # initialize cropped image list
    _crop_ims = [];
    for _l in range(np.max(segmentation_label)):
        #print _l
        if len(_label_dim) == 3: # 3D
            _limits = np.zeros([len(_label_dim),2]); # initialize matrix to save cropping limit
            _binary_label = segmentation_label == _l+1 # extract binary image
            for _m in range(len(_label_dim)):
                _1d_label = _binary_label.sum(_m) > 0;
                _has_label=False;
                for _n in range(len(_1d_label)):
                    if _1d_label[_n] and not _has_label:
                        _limits[_m,0] = max(_n-extend_dim, 0);
                        _has_label = True;
                    elif not _1d_label[_n] and _has_label:
                        _limits[_m,1] = min(_n+extend_dim, _im_dim[_m]);
                        _has_label = False;
                if _has_label:
                    _limits[_m,1] = _im_dim[_m];
            # crop image and save to _crop_ims
            if drift is None:
                _crop_ims.append(im[_limits[0,0]:_limits[0,1], _limits[2,0]:_limits[2,1], _limits[1,0]:_limits[1,1]])
            else: # do drift correction first and crop
                # define a new drift limits to do cropping
                _drift_limits = np.zeros(_limits.shape)
                for _m, _dim, _d in zip(list(range(len(_label_dim))), _im_dim[-len(_label_dim):], drift[[0,2,1]]):
                    _drift_limits[_m, 0] = max(_limits[_m, 0]-np.ceil(np.max(np.abs(_d))), 0);
                    _drift_limits[_m, 1] = min(_limits[_m, 1]+np.ceil(np.max(np.abs(_d))), _dim);
                #print _drift_limits
                # crop image for pre-correction
                _pre_im = im[_drift_limits[0,0]:_drift_limits[0,1],_drift_limits[2,0]:_drift_limits[2,1],_drift_limits[1,0]:_drift_limits[1,1]]
                # drift correction
                _post_im = shift(_pre_im, -drift);
                # re-crop
                _limit_diffs = _limits - _drift_limits;
                for _m in range(len(_label_dim)):
                    if _limit_diffs[_m,1] == 0:
                        _limit_diffs[_m,1] = _limits[_m,1] - _limits[_m,0]
                _limit_diffs = _limit_diffs.astype(np.int);
                #print _limit_diffs
                _crop_ims.append(_post_im[_limit_diffs[0,0]:_limit_diffs[0,0]+_limits[0,1]-_limits[0,0],\
                                          _limit_diffs[2,0]:_limit_diffs[2,0]+_limits[2,1]-_limits[2,0],\
                                          _limit_diffs[1,0]:_limit_diffs[1,0]+_limits[1,1]-_limits[1,0]])

        else: # 2D
            _limits = np.zeros([len(_label_dim),2], dtype=np.int); # initialize matrix to save cropping limit
            _binary_label = segmentation_label == _l+1 # extract binary image
            for _m in range(len(_label_dim)):
                _1d_label = _binary_label.sum(_m) > 0;
                _has_label=False;
                for _n in range(len(_1d_label)):
                    if _1d_label[_n] and not _has_label:
                        _limits[_m,0] = max(_n-extend_dim, 0);
                        _has_label = True;
                    elif not _1d_label[_n] and _has_label:
                        _limits[_m,1] = min(_n+extend_dim, _im_dim[1+_m]);
                        _has_label = False;
                if _has_label: # if label touch boundary
                    _limits[_m,1] = _im_dim[1+_m];
            #print _limits
            # crop image and save to _crop_ims
            if drift is None:
                _crop_ims.append(im[:,_limits[1,0]:_limits[1,1],_limits[0,0]:_limits[0,1]])
            else: # do drift correction first and crop
                # define a new drift limits to do cropping
                _drift_limits = np.zeros(_limits.shape, dtype=np.int)
                for _m, _dim in zip(list(range(len(_label_dim))), _im_dim[-len(_label_dim):]):
                    _drift_limits[_m, 0] = max(_limits[_m, 0]-np.ceil(np.abs(drift[2-_m])), 0);
                    _drift_limits[_m, 1] = min(_limits[_m, 1]+np.ceil(np.abs(drift[2-_m])), _dim);
                #print _drift_limits
                # crop image for pre-correction
                _pre_im = im[:,_drift_limits[1,0]:_drift_limits[1,1],_drift_limits[0,0]:_drift_limits[0,1]]
                # drift correction
                _post_im = shift(_pre_im, -drift);
                # re-crop
                _limit_diffs = (_limits - _drift_limits).astype(np.int);
                #print _limit_diffs
                _crop_ims.append(_post_im[:,_limit_diffs[1,0]:_limit_diffs[1,0]+_limits[1,1]-_limits[1,0],_limit_diffs[0,0]:_limit_diffs[0,0]+_limits[0,1]-_limits[0,0]])
    return _crop_ims


# get limitied points of seed within radius of a center
def get_seed_in_distance(im, center,
                         num_seeds=-1, seed_radius=15, gfilt_size=0.5, filt_size=3, th_seed_percentile=90.,
                         hot_pix_th=0, return_h=False):
    '''Get seed points with in a distance to a center coordinate
    Inputs:
        im: image, 3D-array
        center: center coordinate to get seeds nearby, 1d array / list of 3
        num_seed: maximum number of seeds kept within radius, -1 means keep all, int (default: -1)
        seed_radius: distance of seed points to center, float (default: 15)
        gfilt_size: sigma of gaussian filter applied to image before seeding, float (default: 0.5)
        filt_size: getting local maximum and minimum within in size, int (default: 3)
        th_seed_percentile: intensity percentile of whole image that used as seeding threshold, float (default: 90.)
        hot_pix_th: thereshold for hot pixels, int (default: 0, not removing hot pixel)
        return_h: whether return height of seeds, bool (default: False)
    Outputs:
        _seeds: z,x,y coordinates of seeds, 3 by n matrix
            n = num_seed
            if return height is true, return h,z,x,y instead.
        '''
    from scipy.stats import scoreatpercentile
    from scipy.spatial.distance import cdist

    # check input
    if len(center) != 3:
        raise ValueError('wrong input dimension of center!');
    _dim = np.shape(im);
    _center = np.array(center, dtype=np.float);
    _limits = np.zeros([2,3], dtype=np.int);
    _limits[0] = np.array([np.max([x,y]) for x,y in zip(np.zeros(3), _center-seed_radius)])
    _limits[1] = np.array([np.min([x,y]) for x,y in zip(_dim, _center+seed_radius)])
    _th_seed = scoreatpercentile(im, th_seed_percentile);
    # crop image
    _cim = im[_limits[0,0]:_limits[1,0],_limits[0,1]:_limits[1,1],_limits[0,2]:_limits[1,2]];
    # get candidate seeds
    _cand_seeds = get_seed_points_base(_cim, gfilt_size=gfilt_size, filt_size=filt_size,
                                       th_seed=_th_seed, hot_pix_th=hot_pix_th, return_h=True);
    _local_center = _center - _limits[0];
    # keep seed within distance
    _distance = cdist(_cand_seeds[:3].transpose(), _local_center[np.newaxis,:3]).transpose()[0]
    _keep = _distance < seed_radius;
    _seeds = _cand_seeds[:,_keep]
    _seeds[:3,:] += _limits[0][:,np.newaxis]
    # if limited seeds reported, report top n
    if num_seeds > 0:
        _intensity_order = np.argsort(_seeds[-1]);
        _seeds = _seeds[:,np.flipud(_intensity_order[-num_seeds:])]
    # if not return height, remove height
    if not return_h:
        _seeds = _seeds[:3];

    return _seeds

# fit single gaussian with varying width given prior
def fit_single_gaussian(data, center_zxy, width_zxy=[1.35, 1., 1.], radius=10, n_approx=10,
                        height_sensitivity=100.,
                        expect_intensity=800.,
                        expect_weight = 1000. ):
    """ Function to fit single gaussian with given prior
    Inputs:
        data: image, 3d-array
        center_zxy: center coordinate of seed, 1darray or list of 3
        width_zxy: prior width of gaussian fit, 1darray or list of 3 (default: [1.35,1,1])
        radius: radius that allowed for fitting, float (default: 10)
        n_approx: number of pixels used for approximation, int (default: 10)
        height_sensitivity: grant height parameter extra sensitivity compared to others, float (default: 100)
        expect_intensity: lower limit of penalty function applied to fitting, float (default: 1000)
        expect_weight: L1 norm penalty function applied to widths, float (default: 1000)
    Outputs:
        p.x, p.success: parameters and whether success
        Returns (height, x, y,z, width_x, width_y,width_z,bk)
        the gaussian parameters of a 2D distribution found by a fit"""

    data_=np.array(data,dtype=float)
    dims = np.array(data_.shape)

    if center_zxy:
        center_z,center_x,center_y = center_zxy
    else:
        zxy = np.array(list(map(np.ravel,np.indices(data_.shape))))
        data__=data_[zxy[0],zxy[1],zxy[2]]
        args_high = np.argsort(data__)[-n_approx:]
        center_z,center_x,center_y = np.median(zxy[:,args_high],axis=-1)

    zxy = sphere([center_z,center_x,center_y],radius,imshape=dims).T
    if len(zxy[0])>0:
        data__=data_[zxy[0],zxy[1],zxy[2]]
        sorted_data = np.sort(data__)#np.sort(np.ravel(data__))
        bk = np.median(sorted_data[:n_approx])
        if bk < 0:
            bk = 0;
        height = (np.median(sorted_data[-n_approx:])-bk) / height_sensitivity
        width_z,width_x,width_y = np.array(width_zxy)
        params_ = (height,center_z,center_x,center_y,bk, width_z,width_x,width_y)

        def gaussian(height,center_z, center_x, center_y,
                     bk=0,
                     width_z=width_zxy[0],
                     width_x=width_zxy[1],
                     width_y=width_zxy[2]):
            """Returns a gaussian function with the given parameters"""
            width_x_ = np.abs(width_x)
            width_y_ = np.abs(width_y)
            width_z_ = np.abs(width_z)
            height_ = np.abs(height)
            bk_ = np.abs(bk)
            def gauss(z,x,y):
                g = bk_ + height_ * height_sensitivity *np.exp(
                    -(((center_z-z)/width_z_)**2+((center_x-x)/width_x_)**2+
                      ((center_y-y)/width_y_)**2)/2.)
                return g
            return gauss
        def errorfunction(p):
            f = gaussian(*p)(*zxy)
            g = data__
            #err=np.ravel(f-g-g*np.log(f/g))
            err=np.ravel(f-g) + expect_weight * (np.linalg.norm(p[-3:]-width_zxy, 1) + 0.2 * (p[0]*height_sensitivity/expect_intensity)*int(p[0]<expect_intensity));
            return err
        p = scipy.optimize.least_squares(errorfunction,  params_, bounds=(0, np.inf))
        p.x[0] *= height_sensitivity

        return  p.x, p.success
    else:
        return None,None

# Multi gaussian fitting
def fit_multi_gaussian(im, seeds, width_zxy = [1.35,1,1], fit_radius=5,
                       height_sensitivity=100., expect_intensity=800., expect_weight=1000.,
                       n_max_iter = 10, max_dist_th=0.25, return_im=False, verbose=True):
    """ Function to fit multiple gaussians (with given prior)
    Inputs:
        im: image, 3d-array
        center_zxy: center coordinate of seed, 1darray or list of 3
        width_zxy: prior width of gaussian fit, 1darray or list of 3 (default: [1.35,1,1])
        fit_radius: radius that allowed for fitting, float (default: 10)
        height_sensitivity: grant height parameter extra sensitivity compared to others, float (default: 100)
        expect_intensity: lower limit of penalty function applied to fitting, float (default: 1000)
        expect_weight: L1 norm penalty function applied to widths, float (default: 1000)
        n_max_iter: max iteration count for re-fit existing points, int (default: 10)
        max_dist_th: maximum allowed distance between original fit and re-fit, float (default: 0.25)
        return_im: whether return images of every single fitting, bool (default: False)
        verbose: whether say something, bool (default: True)
    Outputs:
        p: parameters
        Returns (height, x, y,z, width_x, width_y,width_z,bk)
        the gaussian parameters of a 2D distribution found by a fit"""

    if verbose:
        print("Fitting:" +str(len(seeds[0]))+" points")
    # transform seeds input
    _seeds = seeds.transpose() # fitting kernels provided by previous seeding
    if len(_seeds) > 0:
        # initialize
        ps = []
        sub_ims = []
        im_subtr = np.array(im,dtype=np.float)

        # loop through seeds
        for _seed in _seeds:
            p, success = fit_single_gaussian(im_subtr,_seed,
                                                                      height_sensitivity=height_sensitivity,
                                                                      expect_intensity=expect_intensity,
                                                                      expect_weight=expect_weight,
                                                                      radius=fit_radius,
                                                                      width_zxy=width_zxy)
            if p is not None and success: # If got any successful fitting, substract fitted profile
                ps.append(p)
                sub_ims.append(im_subtr)
                im_subtr = subtract_source(im_subtr,p)

        # recheck fitting
        im_add = np.array(im_subtr)
        max_dist=np.inf
        n_iter = 0
        while max_dist > max_dist_th:
            ps_1=np.array(ps)
            ps_1=ps_1[np.argsort(ps_1[:,0])[::-1]]
            ps = []
            ps_1_rem=[]
            for p_1 in ps_1:
                _seed = p_1[1:4]
                im_add = plus_source(im_add,p_1)
                p,success = fit_single_gaussian(im_subtr,_seed,
                                                                      height_sensitivity=height_sensitivity,
                                                                      expect_intensity=expect_intensity,
                                                                      expect_weight=expect_weight,
                                                                      radius=fit_radius,
                                                                      width_zxy=width_zxy)
                if p is not None:
                    ps.append(p)
                    ps_1_rem.append(p_1)
                    im_add = subtract_source(im_add,p)
            ps_2=np.array(ps)
            ps_1_rem=np.array(ps_1_rem)
            dif = ps_1_rem[:,1:4]-ps_2[:,1:4]
            max_dist = np.max(np.sum(dif**2,axis=-1))
            n_iter+=1
            if n_iter>n_max_iter:
                break
        if return_im:
            return ps_2, sub_ims
        else:
            return ps_2
    else:
        return np.array([])
