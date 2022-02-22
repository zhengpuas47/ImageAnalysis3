# global variables
from .. import _correction_folder, _temp_folder, _distance_zxy, _sigma_zxy, _allowed_colors, _image_size
# some shared parameters
_seed_th={
    '750': 400,
    '647': 600,
    '561': 400,
}

## load sub packages
# sub-package for fitting spots
from . import fitting 
# sub-package for picking spots
from . import picking
# sub-package for scoring spots
from . import scoring
# sub-package for checking spots
from . import checking
# matching DNA RNA
from . import matching
# translating, warpping spots
from . import translating
# relabelling analysis
from . import relabelling

# default params
_3d_spot_infos = ['height', 'z', 'x', 'y', 'background', 'sigma_z', 'sigma_x', 'sigma_y', 'sin_t', 'sin_p', 'eps']
_3d_infos = ['z', 'x', 'y']
_spot_coord_inds = [_3d_spot_infos.index(_info) for _info in _3d_infos]