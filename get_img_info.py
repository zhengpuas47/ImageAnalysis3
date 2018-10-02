import sys,glob,os
import numpy as np
sys.path.append(r'C:\Users\puzheng\Documents\python-functions\python-functions-library')

def get_hybe(folder):
	#this sorts by the region number Rx.
	try: return int(os.path.basename(folder).split('H')[-1].split('R')[0])
	except: return np.inf

def get_folders(master_folder, feature='H', verbose=True):
	'''Function to get all subfolders
	given master_folder directory
		feature: 'H' for hybes, 'B' for beads
	returns folders, field_of_views'''
	folders = [folder for folder in glob.glob(master_folder+os.sep+'*') if os.path.basename(folder)[0]==feature] # get folders start with 'H'
	folders = list(np.array(folders)[np.argsort(list(map(get_hybe,folders)))])
	fovs = list(map(os.path.basename,glob.glob(folders[0]+os.sep+'*.dax')))
	if verbose:
		print("Get Folder Names: (ia.get_img_info.get_folders)")
		print("- Number of folders:", len(folders));
		print("- Number of field of views:", len(fovs));
	return folders, fovs;

def get_img_fov(folders, fovs, fov_id=0, verbose=True):
	'''Function to load a certain field_of_view
	Inputs:
		folders: folder names for each hybe, list of strings
		fovs: field of view names, list of strings
		fov_id: field_of_views id, int
	Outputs:
		List of dax-image items
		List of names compatible to dax-images
	'''
	from . import visual_tools as vis
	if not isinstance(fov_id, int):
		raise ValueError('Wrong fov_id input type!')
	if verbose:
		print("Get images of a fov (ia.get_img_info.get_img_fov)")
	_fov = fovs[fov_id]
	_names, _ims = [],[]
	if verbose:
		print("- loading field of view:", _fov);
	# load images
	for _folder in folders[:]:
		_filename= _folder+os.sep+_fov
		if os.path.exists(_filename):
			_names += [os.path.basename(_folder)+os.sep+_fov]
			_ims += [vis.DaxReader(_filename).loadMap()]

	if verbose:
		print("- number of images loaded:", len(_ims))

	return _ims, _names

def get_img_hyb(folders, fovs, hyb_id=0, verbose=True):
	'''Function to load images for a certain hyb
	Inputs:
		folders: folder names for each hybe, list of strings
		fovs: field of view names, list of strings
		hyb_id: field_of_views id, int
	Outputs:
		List of dax-image items
		List of names compatible to dax-images
	'''
	from . import visual_tools as vis
	# check input style
	if not isinstance(hyb_id, int):
		raise ValueError('Wrong hyb_id input type!')
	# initialize
	_names, _ims = [],[]
	_folder = folders[hyb_id];
	print("--loading images from folder:", _folder)
	# read images
	for _fov in fovs:
		_filename = _folder+os.sep+_fov
		if os.path.exists(_filename):
			_names += [os.path.basename(_folder)+os.sep+_fov];
			_ims += [vis.DaxReader(_filename).loadMap()];

	print("-- number of images loaded:", len(_ims))

	return _ims, _names

## Load file Color_Usage in dataset folder
def Load_Color_Usage(master_folder, color_filename='Color_Usage', color_format='csv',
					 DAPI_hyb_name="H0R0", return_color=True):
	'''Function to load standard Color_Usage file:
	Inputs:
		master_folder: master directory of this dataset, path(string);
		color_filename: filename and possible sub-path for color file, string;
		color_format: format of color file, csv or txt
	Outputs:
		color_usage: dictionary of color usage, folder_name -> list of region ID
		dapi: whether dapi is used, bool
		'''
	# initialize as default
	_color_usage = {};

	# process with csv format
	if color_format == 'csv':
		_full_name = master_folder+os.sep+color_filename+"."+'csv';
		print("- Importing csv file:", _full_name);
		import csv
		with open(_full_name, 'r') as handle:
			_reader = csv.reader(handle)
			_header = next(_reader);
			print("- header:", _header)
			for _content in _reader:
				while _content[-1] == '':
					_content = _content[:-1]
				_hyb = _content.pop(0);
				_color_usage[_hyb] = _content
	# process with txt format (\t splitted)
	elif color_format == 'txt':
		_full_name = master_folder+os.sep+color_filename+"."+'txt';
		print("- Importing txt file:", _full_name);
		with open(_full_name, 'r') as handle:
			_line = handle.readline().rstrip();
			_header = _line.split('\t')
			print("-- header:", _header)
			for _line in handle.readlines():
				_content = _line.rstrip().split('\t');
				while _content[-1] == '':
					_content = _content[:-1]
				_hyb = _content.pop(0);
				_color_usage[_hyb] = _content
	# detect dapi
	if DAPI_hyb_name in _color_usage:
		print(f"-- Hyb {DAPI_hyb_name} exists in this data")
		if 'dapi' in _color_usage[DAPI_hyb_name] or 'DAPI' in _color_usage[DAPI_hyb_name]:
			print("-- DAPI exists")
			_dapi = True
		else:
			_dapi = False
	else:
		_dapi = False
	if return_color:
		_colors = [int(_c) for _c in _header[1:]]
		return _color_usage, _dapi, _colors
	return _color_usage, _dapi

# function for finding bead_channel given color_usage profile
def find_bead_channel(__color_dic, __bead_mark='beads'):
	'''Given a color_dic loaded from Color_Usage file, return bead channel if applicable'''
	__bead_channels = []
	for __name, __info in sorted(list(__color_dic.items()), key=lambda k_v:int(k_v[0].split('H')[1].split('R')[0])):
		__bead_channels.append(__info.index(__bead_mark))
	__unique_channel = np.unique(__bead_channels);
	if len(__unique_channel) == 1:
		return __unique_channel[0];
	else:
		raise ValueError("-- bead channel not unique:", __unique_channel)
		return __unique_channel

# function for finding DAPI channel given color_usage profile
def find_dapi_channel(__color_dic, __dapi_mark='DAPI'):
	'''Given a color_dic loaded from Color_Usage file, return bead channel if applicable'''
	__dapi_channels = []
	for __name, __info in sorted(list(__color_dic.items()), key=lambda k_v:int(k_v[0].split('H')[1].split('R')[0])):
		if __dapi_mark in __info:
			__dapi_channels.append(__info.index(__dapi_mark))
	__unique_channel = np.unique(__dapi_channels);
	if len(__unique_channel) == 1:
		return __unique_channel[0];
	else:
		raise ValueError("-- dapi channel not unique:", __unique_channel)
		return __unique_channel

# load encoding scheme for decoding
def Load_Encoding_Scheme(master_folder, encoding_filename='Encoding_Scheme', encoding_format='csv',
					     return_info=True, verbose=True):
	'''Load encoding scheme from csv file
	Inputs:
		master_folder: master directory of this dataset, path(string);
		encoding_filename: filename and possible sub-path for color file, string;
		encoding_format: format of color file, csv or txt
	Outputs:
		_encoding_scheme: dictionary of encoding scheme, list of folder_name -> encoding matrix
		(optional)
		_num_hyb: number of hybridization per group, int
		_num_reg: number of regions per group, int
		_num_color: number of colors used in parallel, int'''
	# initialize
	_hyb_names = [];
	_encodings = [];
	_num_hyb,_num_reg,_num_color,_num_group = None,None,None,None
	# process with csv format
	if encoding_format == 'csv':
		_full_name = master_folder+os.sep+encoding_filename+"."+'csv';
		if verbose:
			print("- Importing csv file:", _full_name);
		import csv
		with open(_full_name, 'r') as handle:
			_reader = csv.reader(handle)
			_header = next(_reader);
			for _content in _reader:
				_hyb = _content.pop(0);
				for _i,_n in enumerate(_content):
					if _n == '':
						_content[_i] = -1
				if str(_hyb) == 'num_hyb':
					_num_hyb = int(_content[0]);
				elif str(_hyb) == 'num_reg':
					_num_reg = int(_content[0]);
				elif str(_hyb) == 'num_color':
					_num_color = int(_content[0]);
				else:
					_hyb_names.append(_hyb);
					_encodings.append(_content);
	# process with txt format (\t splitted)
	elif encoding_format == 'txt':
		_full_name = master_folder+os.sep+encoding_filename+"."+'txt';
		if verbose:
			print("- Importing txt file:", _full_name);
		with open(_full_name, 'r') as handle:
			_line = handle.readline().rstrip();
			_header = _line.split('\t')
			for _line in handle.readlines():
				_content = _line.rstrip().split('\t');
				_hyb = _content.pop(0);
				for _i,_n in enumerate(_content):
					if _n == '':
						_content[_i] = -1
				if str(_hyb) == 'num_hyb':
					_num_hyb = int(_content[0]);
				elif str(_hyb) == 'num_reg':
					_num_reg = int(_content[0]);
				elif str(_hyb) == 'num_color':
					_num_color = int(_content[0]);
				else:
					_hyb_names.append(_hyb);
					_encodings.append(_content);
	# first, read all possible colors
	_header.pop(0);
	_colors = [];
	if len(_header) == _num_reg * _num_color:
		for _j in range(int(len(_header)/_num_reg)):
			if int(_header[_j*_num_reg]):
				_colors.append(_header[_j*_num_reg])
			else:
				raise EOFError('wrong color input'+str(_header[_j*_num_reg]))
	else:
		raise ValueError('Length of header doesnt match num reg and num color');
	# initialze encoding_scheme dictionary
	_encoding_scheme = {_color:{'names':[],'matrices':[]} for _color in _colors};
	# if number of region, number of hybe and number of color defined:
	if _num_hyb and _num_reg and _num_color:
		if len(_hyb_names)% _num_hyb == 0:
			for _i in range(int(len(_hyb_names)/ _num_hyb)):
				_hyb_group = _hyb_names[_i*_num_hyb: (_i+1)*_num_hyb]
				_hyb_matrix = np.array(_encodings[_i*_num_hyb: (_i+1)*_num_hyb], dtype=int)
				if _hyb_matrix.shape[1] == _num_reg * _num_color:
					for _j in range( int(_hyb_matrix.shape[1]/_num_reg)):
						_mat = _hyb_matrix[:,_j*_num_reg:(_j+1)*_num_reg]
						if not (_mat == -1).all():
							_encoding_scheme[_colors[_j]]['names'].append(_hyb_group);
							_encoding_scheme[_colors[_j]]['matrices'].append(_mat)
				else:
					raise ValueError('dimension of hyb matrix doesnt match color and region')
		else:
			raise ValueError('number of hybs doesnot match number of hybs per group.');
	# calculate number of groups per color
	_group_nums = []
	for _color in _colors:
		_group_nums.append(len(_encoding_scheme[_color]['matrices']));
	if verbose:
		print("-- hyb per group:",_num_hyb)
		print("-- region per group:", _num_reg)
		print("-- colors:", _colors)
		print('-- number of groups:', _group_nums)
	# return
	if return_info:
		return _encoding_scheme, _num_hyb, _num_reg, _colors, _group_nums
	return _encoding_scheme

def split_channels(ims, names, num_channel=2, buffer_frames=10, DAPI=False, verbose=True):
	'''Function to load images, and split images according to channels
	Inputs:
		ims: all images that acquired by get_images, list of images
		names: compatible names also from get_images, list of strings
		num_channel: number of channels, int
		buffer frames to be removed, non-negative int
		DAPI: whether DAPI exists in round 0, default is false.
	Outputs:
		image lists in all field_of_views
	'''
	_ims = ims;
	if verbose:
		print("Split multi-channel images (ia.get_img_info.split_channels)")
		print("--number of channels:", num_channel)
	# Initialize
	if DAPI:
		print("-- assuming H0R0 has extra DAPI channel")
		_splitted_ims = [[] for _channel in range(num_channel + 1)];
	else:
		_splitted_ims = [[] for _channel in range(num_channel)];
	# if H0 is for DAPI, there will be 1 more channel:
	if DAPI:
		_im = _ims[0];
		for _channel in range(num_channel+1):
			_splitted_ims[(buffer_frames-1+_channel)%(num_channel+1)].append(_im[int(buffer_frames): -int(buffer_frames)][_channel::num_channel+1])
	# loop through images
	for _im in _ims[int(DAPI):]:
		# split each single image
		for _channel in range(num_channel):
			_splitted_ims[(buffer_frames-1+_channel)%(num_channel)].append(_im[int(buffer_frames): -int(buffer_frames)][_channel::num_channel])
	return _splitted_ims;



def split_channels_by_image(ims, names, num_channel=4, buffer_frames=10, DAPI=False, verbose=True):
	'''Function to split loaded images into multi channels, save as dict'''
	# initialzie
	_ims = ims;
	_im_dic = {}
	if verbose:
		print("Split multi-channel images (ia.get_img_info.split_channels)")
		print("-- number of channels:", num_channel)
	# if dapi applied
	if DAPI:
		if verbose:
			print("-- scanning through images to find images with DAPI:")
		_im_sizes = [_im.shape[0] for _im in _ims];
		if np.max(_im_sizes) == np.min(_im_sizes):
			raise ValueError('image dimension does not match for DAPI');
		if (np.max(_im_sizes) - 2*buffer_frames) / (num_channel + 1) != (np.min(_im_sizes) - 2*buffer_frames) / num_channel:
			raise ValueError('image for DAPI does not have compatible frame numbers for 1 more channel');
		else:
			if verbose:
				print("--- DAPI images discovered, splitting images")
		for _im,_name in zip(_ims, names):
			if _im.shape[0] == np.max(_im_sizes):
				_im_list = [_im[int(buffer_frames): -int(buffer_frames)][(-buffer_frames+1+_channel)%(num_channel+1)::num_channel+1] for _channel in range(num_channel+1)]
			else:
				_im_list = [_im[int(buffer_frames): -int(buffer_frames)][(-buffer_frames+1+_channel)%num_channel::num_channel] for _channel in range(num_channel)]
			_im_dic[_name] = _im_list;
	else:
		if verbose:
			print("-- splitting through images without DAPI");
		for _im,_name in zip(_ims, names):
			_im_list = [_im[int(buffer_frames): -int(buffer_frames)][(-buffer_frames+1+_channel)%num_channel::num_channel] for _channel in range(num_channel)]
			_im_dic[_name] = _im_list;
	return _im_dic

# match harry's result with raw data_
def decode_match_raw(raw_data_folder, raw_feature, decode_data_folder, decode_feature, fovs, e_type):
    # initialize
    _match_dic = {};
    # get raw data file list
    _raw_list = glob.glob(raw_data_folder+os.sep+'*'+raw_feature)
    _raw_list = [_raw for _raw in _raw_list if str(e_type) in _raw]
    # get decode data file list
    _decode_list = glob.glob(decode_data_folder+os.sep+'*'+decode_feature)

    print(len(_raw_list), len(_decode_list))
    # loop to match!
    for _decode_fl in sorted(_decode_list):
        # fov id
        _fov_id = int(_decode_fl.split('fov-')[1].split('-')[0])
        # search to match
        _matched_raw = [_raw_fl for _raw_fl in _raw_list if fovs[_fov_id].split('.')[0] in _raw_fl]
        # keep unique match
        if len(_matched_raw) == 1:
            _match_dic[_fov_id] = [_matched_raw[0], _decode_fl]

    return _match_dic
