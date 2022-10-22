import numpy as np
import sys, os, glob, time
import pickle
import scipy.sparse as ss
from tqdm import tqdm
import multiprocessing as mp 

from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import MeltingTemp
from Bio.SeqUtils import GC
# variables from local
from . import _fasta_ext
sys.path.append(os.getcwd())
from seqint import seq2Int,seq2Int_rc
#import tools LibraryTools
from .LibraryTools import fastaread
from .LibraryTools import fastawrite
from .LibraryTools import constant_zero_dict
from .LibraryTools import seqrc
from .LibraryTools import OTTable


def tm(string):
    if isinstance(string, bytes):
        string = string.decode()
    return MeltingTemp.Tm_NN(string, nn_table=MeltingTemp.DNA_NN4, 
                             Na=390, dnac1=1, dnac2=0) #390mM Na from 2xSSC: 300mM from NaCl, 90mM from (tri)sodium citrate
def gc(string):
    if isinstance(string, bytes):
        string = string.decode()
    return GC(string) / 100.

def str_to_list(var):
    "Converst a string to a list"
    if type(var) is str:
        return [var]
    else:
        return var

def find_sub_sequence(string, sub_string):
    if isinstance(string, bytes):
        string = string.decode()
    if isinstance(sub_string, bytes):
        sub_string = sub_string.decode()    
    return sub_string.upper() in string.upper()


def check_extension(files,extensions=_fasta_ext):
    "Checks if file/(all files in list) have the same extension/extensions"
    _files = str_to_list(files)
    extensions_ = str_to_list(extensions)
    return np.prod([os.path.basename(fl).split(os.path.extsep)[-1] in extensions_ for fl in _files])==1
    
class countTable():
    def __init__(self,word=17,sparse=False,save_file=None,fromfile=False,verbose=False):
        """
        This constructs sparse array count table using scipy lil_matrix for efficient construction.
        """
        self.word = word
        self.verbose = verbose
        self.save_file = save_file
        
        if isinstance(self.save_file, str) and self.save_file.split(os.path.extsep)[-1] == 'npz':
            self.sparse = True
        elif isinstance(self.save_file, str) and self.save_file.split(os.path.extsep)[-1] == 'npy':
            self.sparse = False
        else:
            self.sparse = sparse

        self.max_size = 4**word
        self.fromfile=fromfile
        self.f=None
        self.matrix=[]
        self.ints=[]
        self.seqs,self.names = [],[]
    def create_matrix(self):
        if self.sparse:
            self.max_sparse_ind = 2**31 #scipy.sparse decided to encoded in indeces in int32. Correct!
            self.nrows = int(self.max_size/self.max_sparse_ind)
            if self.nrows>0:
                self.matrix = ss.csr_matrix((self.nrows,self.max_sparse_ind), dtype=np.uint16)
            else:
                self.matrix = ss.csr_matrix((1,self.max_size), dtype=np.uint16)
        else:
            self.matrix = np.zeros(self.max_size, dtype=np.uint16)
    def save(self):
        if self.verbose:
            print(f"- start saving to file: {self.save_file}")
        if self.save_file is not None:
            if self.sparse:
                ss.save_npz(self.save_file.split(os.path.extsep)[0], self.matrix)
            else:
                self.matrix.tofile(self.save_file)
    def load(self):
        if self.save_file is not None:
            if self.sparse:
                self.matrix = ss.load_npz(self.save_file)
                self.max_sparse_ind = 2**31 #scipy.sparse decided to encoded in indeces in int32. Correct!
            else:
                self.matrix = np.fromfile(self.save_file,dtype=np.uint16)
    def complete(self,verbose=False):
        """a np.unique is performed on self.ints and the number of occurences for each unique 17mer (clipped to 2^16-1) is recorded in a sparse array self.matrix"""
        if verbose:
            start = time.time()
        self.create_matrix()
        pos,cts = np.unique(self.ints,return_counts=True)
        countTable_values = np.array(np.clip(cts, 0, np.iinfo(np.uint16).max), dtype=np.uint16)#clip and recast as uint16
        countTable_indices = np.array(pos, dtype=np.uint64)
        if verbose:
            end=time.time()
            print('Time to compute unique and clip:',end-start)
        
        if verbose:
            start = time.time()
        if self.sparse:
            self.matrix = self.matrix.tolil()
            if self.max_sparse_ind<=self.max_size:
                pos_col = int(pos/self.max_sparse_ind)
                pos_row = int(pos-pos_col*self.max_sparse_ind)
                self.matrix[pos_col,pos_row] = countTable_values
            else:
                self.matrix[0,pos] = countTable_values
            self.matrix = self.matrix.tocoo()
        else:
            self.matrix[countTable_indices]=countTable_values
        if verbose:
            end=time.time()
            print('Time to update matrix:',end-start)

    def read(self,files):
        """Read fasta files and store the names and seqs in self.names, self.seqs"""
        if type(files) is str:
            files = [files]
        self.seqs,self.names = [],[]
        for fl in files:
            nms,sqs = fastaread(fl)
            sqs = [sq.upper() for sq in sqs]
            self.names.extend(nms)
            self.seqs.extend(sqs)

    def consume_loaded(self, parallel=True, num_threads=12):
        """Having self.seqs,self.names this updates the self.matrix with size 4**word and counts capped up to 2**16"""

        word = self.word

        # prepare args
        _consume_args = [(_seq, int(word), self.verbose) for _seq in self.seqs]
        if self.verbose:
            print(f"- Start multi-processing comsume {len(_consume_args)} sequences {num_threads} threads", end=', ')
            _start_time = time.time()
        with mp.Pool(int(num_threads)) as _comsume_pool:
            _result_ints = _comsume_pool.starmap(consume_sequence_to_int, _consume_args, chunksize=1)
            _comsume_pool.close()
            _comsume_pool.join()
            _comsume_pool.terminate()
        if self.verbose:
            print(f"finish in {time.time()-_start_time:.3f}s")
        
        for _ints in _result_ints:
            self.ints.extend(_ints)

        if self.verbose:
            print(f"- Total sequences loaded: {len(self.ints)}")

    def consume_batch_file(self,batch=1000000,reset=False):
        assert(self.save_file is not None)
        if reset:
            #create the file with 0s
            f = open(self.save_file,"wb")
            f.seek(4**self.word*2-1)
            f.write(b"\0")
            f.close()
        
        f = open(self.save_file, "r+b")
        word= self.word
        for nm,sq in zip(self.names,self.seqs):
            
            for isq in range(int(len(sq)/batch)+1):
                sq_ = sq[isq*batch:(isq+1)*batch]
                sq_word = [sq_[i:i+word] for i in range(len(sq_)-word)]
                ints = list(map(seq2Int,sq_word))
                pos,cts = np.unique(ints,return_counts=True)
                ints_vals = []
                for pos_,ct_ in zip(pos,cts):
                    if self.verbose:
                        print(nm,pos_)
                        
                    f.seek(pos_*2)
                    ct = np.fromfile(f,dtype=np.uint16,count=1)
                    ct_clip = np.clip(int(ct)+int(ct_), 0, 2**16-1).astype(np.uint16)
                    f.seek(-2,1)
                    f.write(ct_clip.tobytes())
                    ints_vals.append(ct_clip)
        f.close()
    def consume(self,sq,verbose=False):
        """Given a big sequence sq, this breaks in into all contigous subseqs of size <word> and records each occurence in self.ints"""
        word=self.word
        if len(sq)>=word:
            # encode to bytes
            sq_word = []
            for i in range(len(sq)-word+1):
                _sq = sq[i:i+word]
                if isinstance(_sq, str):
                    sq_word.append(_sq.encode())
                elif isinstance(_sq, bytes):
                    sq_word.append(_sq)
                else:
                    raise TypeError(f"Wrong input type for sequence:{_sq}")
            if verbose:
                start = time.time()
            self.ints.extend(list(map(seq2Int,sq_word)))
            if verbose:
                end=time.time()
                print('Time to compute seq2Int:',end-start)
    def get(self,seq,rc=False):
        """give an oligo, this breaks it into all contigous subseqs of size <word> and returns the sum of the counts"""
        word = self.word
        if len(seq)<word:
            return 0
        seqs = [seq[i:i+word] for i in range(len(seq)-word+1)]
        if not rc:
            ints = np.array(list(map(seq2Int,seqs)),dtype='uint64')
        else:
            ints = np.array(list(map(seq2Int_rc,seqs)),dtype='uint64')
            
        results = None
        if self.fromfile:
            #read from file
            if self.f is None:
                self.f = open(self.filename,'rb')
            results_ = []
            for int_ in ints:
                self.f.seek(int_*2)
                results_.append(np.fromfile(self.f,dtype=np.uint16,count=1))
            results = np.sum(results_)
            #self.f.close()
            #self.f=None
        else:
            #read from RAM
            if self.sparse:
                pos_col = int(ints/self.max_sparse_ind)
                pos_row = ints-pos_col*self.max_sparse_ind
                results = np.sum(self.matrix[pos_col,pos_row])
            else:
                results = np.sum(self.matrix[ints])
        return results

def OTmap(seqs,word_size=17,use_kmer=True,progress_report=False,save_file=None,sparse=False):
    """This creates an count table using either a sparse matrix form or a python dictionary.
    For large genome-sized tables we recommend the sparse matrix
    """
    if use_kmer:
        map_ = countTable(word=word_size,sparse=sparse)
        print("- Mapping no. of seqs: "+str(len(seqs)))
        for seq in seqs:
            map_.consume(seq.upper(),verbose=progress_report)
        if len(seqs):
            map_.complete()
        if save_file is not None:
            map_.save()
    else:
        specTable = OTTable()
        map_ = specTable.computeOTTable(seqs,word_size,progress_report=progress_report)

        if save_file is not None:
            pickle.dump(map_,open(save_file,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    return map_


class pb_reports_class:
    def __init__(self,
        sequence_dic={'file':None,'rev_com':False,'two_stranded':False},
        map_dic={'transcriptome':{'file':None,'rev_com':True,'two_stranded':False},
                 'genome':{'file':None,'rev_com':False,'two_stranded':True},
                 'rep_genome':{'file':None,'rev_com':False,'two_stranded':True},
                 'isoforms':{'file':None,'rev_com':True,'force_list':True,'two_stranded':False},
                 'self_sequences':{'file':None,'rev_com':False,'force_list':True,'two_stranded':False}
                 },
        save_file=None,
        params_dic={'word_size':17,
                    'pb_len':40, # 40 mer targeting sequences
                    'buffer_len':2,
                    'auto':False,
                    },
        check_dic={('transcriptome','isoforms'): 3, # extra hits outside of isoforms
                   ('genome','self_sequence'): (40-17+1)*2, #  
                   'rep_genome': 0,
                   'gc': (0.25,0.75),
                   'tm': 47+0.61*50+5, 
                   'masks': ['AAAA','TTTT','CCCC','GGGG', # Quartet-repeats
                             'GAATTC','CTTAAG', # EcoRI sites
                             'GGTACC','CCATGG',], # KpnI sites
                  },
        debugging=False, overwrite=False, verbose=True):
        """
        Create Probe designer derived from Bogdan Bintu:
        https://github.com/BogdanBintu/ChromatinImaging/blob/master/LibraryDesign/LibraryDesigner.py
        see more information in __str__
        Required keywords:
        sequence_dic: input information:
            file: input file(s)
            rev_com: whether take reverse compliment of input sequences
            two_stranded: whether design probes on both input sequence and rev_com
        map_dic: target references to map probes into.

        """
        #internalize paramaters
        self.sequence_dic=sequence_dic
        # initialize basic inputs
        self.input_seqs, self.input_names = [], []

        self.map_dic=map_dic
        # change default for self_sequences as input sequences
        if self.map_dic['self_sequences']['file'] is None:
            self.map_dic['self_sequences']['file'] = self.sequence_dic['file']

        self.params_dic = params_dic
        # internalize paramaters in params_dic
        for key in list(self.params_dic.keys()):
            setattr(self,key,self.params_dic[key])
        
        self.check_dic = check_dic
        self.save_file = save_file
        self.overwrite = overwrite
        self.verbose = verbose 
        self.debugging = debugging
        # initialize probes
        self.cand_probes = {}
        self.kept_probes = {}

        #load sequence file
        self.load_sequence_file()
        if self.params_dic['auto']:
            self.computeOTmaps()
            self.compute_pb_report()

    def __str__(self):
        _info_str = f"""
Probe designer derived from Bogdan Bintu:
https://github.com/BogdanBintu/ChromatinImaging/blob/master/LibraryDesign/LibraryDesigner.py
by Pu Zheng, 2020.11

Major changes:
    1. allow design of two strands
    2. separate reverse_complement (rev_com) and from two strands (two_stranded) as 
    two different inputs for map_dic and sequence_dic
    3. replace 'local_genome' with 'self_sequences' to be more explicit, and only 
    exclude the counts for the corresponding self_sequence within each input. 

Key information:
    - number of input_sequence(s): {len(self.input_seqs)}
    - save_file location: {self.save_file}
"""
        return _info_str

    def load_sequence_file(self):
        sequence_file = self.sequence_dic.get('file',None)
        self.input_names,self.input_seqs = [],[]
        self.input_files=[]

        if sequence_file is not None:
            sequence_files = str_to_list(sequence_file)
            for sequence_file_ in sequence_files:
                names,seqs = fastaread(sequence_file_,force_upper=True)
                self.input_files.extend([sequence_file_]*len(names))
                self.input_names.extend(names)
                self.input_seqs.extend(seqs)
        # append info into sequence_dic, for saving purposes
        self.sequence_dic['sequence_lens']=list(map(len,self.input_seqs))
        self.sequence_dic['input_files']=self.input_files
        self.sequence_dic['input_names']=self.input_names
        self.sequence_dic['input_seqs']=self.input_seqs

    def computeOTmaps(self):
        """This creates maps:
        Iterates over keys in map_dic and uses self.files_to_OTmap.
        """
        start = time.time()
        for key in list(self.map_dic.keys()):
            if key != 'self_sequences':
                self.files_to_OTmap("map_"+key,self.map_dic[key])
        end = time.time()
        print("Time(s): "+str(end-start))

    def files_to_OTmap(self, map_key, curr_dic):
        "This function transforms a file or list of files to an OT map and sets it as an attribute in self."
        if self.verbose:
            print(f"-- setting attribute: {map_key}",)
            _start_time = time.time()
        # internalize variables    
        files = curr_dic.get('file',None)
        force_list = curr_dic.get('force_list',False)
        save_file = curr_dic.get('save_file',None)
        use_kmer = curr_dic.get('use_kmer',True)
        merge = curr_dic.get('merge', False)

        if files is not None:
            _files = str_to_list(files)
            fasta_ext = ['fa','fasta']
            # create tables if sequences are given
            if check_extension(files,fasta_ext):
                names,seqs=[],[]
                for fl in _files:
                    names_,seqs_=fastaread(fl,force_upper=True)
                    if force_list:
                        names.append(names_)
                        seqs.append(seqs_)
                    else:
                        names.extend(names_)
                        seqs.extend(seqs_)
                if not force_list:
                    names =[names]
                    seqs = [seqs]
                if merge:
                    OTmaps = [OTmap(seqs,word_size=self.params_dic['word_size'],use_kmer=use_kmer,progress_report=False,save_file=save_file)]
                else:
                    OTmaps = [OTmap(seq_,word_size=self.params_dic['word_size'],use_kmer=use_kmer,progress_report=False,save_file=save_file)
                              for seq_ in seqs]
                setattr(self,map_key,OTmaps)
            # for pre-existing tables, load
            elif len(_files)==1 and check_extension(files, 'npy'):
                OTMap_ = countTable(word=self.params_dic['word_size'],sparse=False,save_file=_files[0])
                OTMap_.load()
                OTmaps = [OTMap_]
                setattr(self,map_key,OTmaps)
            # for pre-existing tables, load
            elif len(_files)==1 and check_extension(files, 'npz'):
                OTMap_ = countTable(word=self.params_dic['word_size'],sparse=True,save_file=_files[0])
                OTMap_.load()
                OTmaps = [OTMap_]
                setattr(self,map_key,OTmaps)
            elif len(_files)==1 and check_extension(files, 'pkl'):
                OTmaps = [pickle.load(open(_files[0],'rb'))]
                setattr(self,map_key,OTmaps)
            else:
                print(f"Extension error or more than 1 npy/pkl file provided for {map_key}")
        else:
            print(f"No files provided for {map_key}")
            setattr(self,map_key,[constant_zero_dict()])
        if self.verbose:
            print(f"--- finish {map_key} in {time.time()-_start_time:.3f}s.")

    def release_OTmaps(self,):
        start = time.time()
        for key in list(self.map_dic.keys()):
            if key != 'self_sequences':
                _map_key = "map_"+key
                if hasattr(self, _map_key):
                    delattr(self, _map_key)
        print(f"Time to release OTmaps: {time.time()-start:.3f}s. ")

    def compute_pb_report(self):
        block = self.params_dic['word_size']
        pb_len = self.params_dic['pb_len']
        #buffer_len = self.buffer_len
        #gen_seqs,gen_names = self.input_seqs,self.input_names
        input_rev_com = self.sequence_dic.get('rev_com',False)
        input_two_stranded = self.sequence_dic.get('two_stranded',False)
        #input_use_kmer = self.sequence_dic.get('use_kmer', True)
        if self.verbose:
            print(f"- Designing targeting sequence for {len(self.input_seqs)} regions")
        # initialize
        pb_reports = {}

        # iterate across multiple regions (input seqs)
        for _reg_id, (_name, _seq, _file) in enumerate(zip(self.input_names, self.input_seqs, self.input_files)):
            if self.verbose:
                print(f"-- designing region: {_name}", end=' ')
                _design_start = time.time()
            if len(_seq) <= pb_len:
                if self.verbose:
                    print(f"Too short the sequence, skip.")
                continue
            # compute this input file (fasta) into OTMap
            _input_map_dic = {_k:_v for _k,_v in self.map_dic['self_sequences'].items()}
            _input_map_dic['file'] = _file
            print(f"-- region: {_reg_id}, input file: {_input_map_dic['file']}")

            self.files_to_OTmap('map_self_sequences', _input_map_dic)
            _num_candidate_probes = 0
            # loop through all possible positions
            for _i in range(len(_seq)-pb_len+1):
                # extract sequence
                _cand_seq = _seq[_i:_i+pb_len]
                _rc_cand_seq = seqrc(_cand_seq)

                # case 0, here, skip any sequence contain N
                if 'N' in _cand_seq:
                    continue   

                # if design forward strand:
                if not input_rev_com or input_two_stranded:
                    # get forward strand sequence
                    if isinstance(_cand_seq, str):
                        _cand_seq = _cand_seq.encode()
                    # create basic info
                    pb_reports[_cand_seq] = constant_zero_dict()
                    pb_reports[_cand_seq].update(
                        {'name':f'{_name}_reg_{_reg_id}_pb_{_i}',
                        'reg_index':_reg_id,
                        'reg_name':_name,
                        'pb_index': _i,
                        'strand':'+',
                        'gc':gc(_cand_seq),
                        'tm':tm(_cand_seq),}
                    )
                    _num_candidate_probes += 1
                    # update map_keys
                    for _key, _curr_dic in self.map_dic.items():
                        _map_use_kmer = _curr_dic.get('use_kmer',True)
                        _map_rev_com = _curr_dic.get('rev_com',False)
                        _map_two_stranded = _curr_dic.get('two_stranded',False)               
                        _map_key = f"map_{_key}"
                        _maps = getattr(self, _map_key)
                        # process map counts, use kmer
                        if _map_use_kmer:
                            for _map in _maps:
                                # forward from maps
                                if not _map_rev_com or _map_two_stranded:
                                    pb_reports[_cand_seq][_map_key]+= _map.get(_cand_seq)
                                # reverse from maps
                                if _map_rev_com or _map_two_stranded:
                                    pb_reports[_cand_seq][_map_key]+= _map.get(_cand_seq, rc=True)
                        # not use kmer:
                        else:
                            #Iterate through block regions:
                            for j in range(pb_len-block+1):
                                _blk = _cand_seq[j:j+block]
                                for _map in _maps:
                                    # forward from maps
                                    if not _map_rev_com or _map_two_stranded:
                                        pb_reports[_cand_seq][_map_key] += _map.get(_blk)
                                    # reverse from maps
                                    if _map_rev_com or _map_two_stranded:
                                        pb_reports[_cand_seq][_map_key] += _map.get(seqrc(_blk))

                # if design reverse streand:
                if input_rev_com or input_two_stranded:
                    # get reverse strand sequence
                    if isinstance(_rc_cand_seq, str):
                        _rc_cand_seq = _rc_cand_seq.encode()
                    # create basic info
                    pb_reports[_rc_cand_seq] = constant_zero_dict()
                    pb_reports[_rc_cand_seq].update(
                        {'name':f'{_name}_reg_{_reg_id}_pb_{_i}',
                        'reg_index':_reg_id,
                        'reg_name':_name,
                        'pb_index': _i,
                        'strand':'-',
                        'gc':gc(_rc_cand_seq),
                        'tm':tm(_rc_cand_seq),}
                    )
                    _num_candidate_probes += 1
                    # update map_keys
                    for _key, _curr_dic in self.map_dic.items():
                        _map_use_kmer = _curr_dic.get('use_kmer',True)
                        _map_rev_com = _curr_dic.get('rev_com',False)
                        _map_two_stranded = _curr_dic.get('two_stranded',False)               
                        _map_key = f"map_{_key}"
                        _maps = getattr(self, _map_key)
                        # process map counts, use kmer
                        if _map_use_kmer:
                            for _map in _maps:
                                # forward from maps
                                if not _map_rev_com or _map_two_stranded:
                                    pb_reports[_rc_cand_seq][_map_key]+= _map.get(_rc_cand_seq)
                                # reverse from maps
                                if _map_rev_com or _map_two_stranded:
                                    pb_reports[_rc_cand_seq][_map_key]+= _map.get(_rc_cand_seq, rc=True)
                        # not use kmer:
                        else:
                            #Iterate through block regions:
                            for j in range(pb_len-block+1):
                                _blk = _rc_cand_seq[j:j+block]
                                for _map in _maps:
                                    # forward from maps
                                    if not _map_rev_com or _map_two_stranded:
                                        #blks.append(_rc_cand_seq)
                                        pb_reports[_rc_cand_seq][_map_key] += _map.get(_blk)
                                    # reverse from maps
                                    if _map_rev_com or _map_two_stranded:
                                        pb_reports[_rc_cand_seq][_map_key]+= _map.get(seqrc(_blk))
            
            if self.verbose:
                print(f"- Designed {_num_candidate_probes} candidate probes in {time.time()-_design_start:.3f}s.")
        # add to attribute
        self.cand_probes = pb_reports
        # save
        self.save_to_file()

    def check_probes(self, _cand_probes=None, _check_dic=None, pick_probe_by_hits=True):
        # load candidate probes
        if _cand_probes is None:
            _cand_probes = getattr(self, 'cand_probes', {})
        if _check_dic is None:
            _check_dic = getattr(self, 'check_dic', {})
        
        if not hasattr(self, 'kept_probes'):
            setattr(self, 'kept_probes', {})

        # iterate across multiple regions (input seqs)
        for _reg_id, (_name, _seq, _file) in enumerate(zip(self.input_names, self.input_seqs, self.input_files)):

            # select probes belongs to this region
            _reg_pb_dic = {_pb:_info for _pb, _info in _cand_probes.items()
                           if _info['reg_index'] == _reg_id}
            _sel_reg_pb_dic = {}                           
            _pb_score_dict = {}
            if self.verbose:
                print(f"-- check region:{_reg_id} {_name}, {len(_reg_pb_dic)} candidate probes")
                _check_start = time.time()
            # loop through probes
            for _pb, _info in _reg_pb_dic.items():
                # remove edge probes
                _edge_size = int(max(self.buffer_len, 0))
                if _info['pb_index'] < _edge_size or _info['pb_index'] > len(_seq)-self.pb_len+1-_edge_size:
                    continue
                # if GC or Tm doesn't pass filter, discard
                _check_gc, _check_tm = True, True
                if 'gc' in  _check_dic:
                    if isinstance(_check_dic['gc'], list) or isinstance(_check_dic['gc'], tuple):
                        if _info['gc'] > np.max(_check_dic['gc']) or _info['gc'] < np.min(_check_dic['gc']):
                            _check_gc = False
                    else:
                        if _info['gc'] < _check_dic['gc']:
                            _check_gc = False
                if 'tm' in _check_dic:
                    if isinstance(_check_dic['tm'], list) or isinstance(_check_dic['tm'], tuple):
                        if _info['tm'] > np.max(_check_dic['tm']) or _info['tm'] < np.min(_check_dic['tm']):
                            _check_tm = False
                    else:
                        if _info['tm'] < _check_dic['tm']:
                            _check_tm = False
                if not _check_gc or not _check_tm:
                    #print(_info['tm'], _check_tm, _info['gc'], _check_gc)
                    continue
                # calculate mask
                if 'masks' in _check_dic:
                    # get existence
                    _mask_exists = [find_sub_sequence(_pb, _str) for _str in _check_dic['masks']]
                    if np.sum(_mask_exists) > 0:
                        # skip this probe if masked
                        #print(f"mask {_mask_exists}")
                        continue
                # calculate map values
                _map_score_dict = {}
                _map_check = True
                for _check_key, _thres in _check_dic.items():
                    if _check_key not in ['gc', 'tm', 'masks']:
                        if isinstance(_check_key, str):
                            _check_map_key = f"map_{_check_key}"
                            _pb_map_value = _info[_check_map_key]

                            if _pb_map_value > _thres:
                                _map_check = False
                                break 
                            else:
                                # zero hit with non-zero threshold
                                if _pb_map_value <= 0 and _thres > 0:
                                    _map_score_dict[_check_key] = _thres / 0.5 #
                                # if both zero hits and zero threshold ,skip
                                elif _pb_map_value <= 0 and _thres <= 0: 
                                    _map_score_dict[_check_key] = np.nan 
                                # have hits with positive threshold
                                else:    
                                    _map_score_dict[_check_key] =  _thres / _pb_map_value

                        elif isinstance(_check_key, list) or isinstance(_check_key, tuple):
                            _tar_map_key, _ref_map_key = f"map_{_check_key[0]}", f"map_{_check_key[1]}"
                            _pb_map_value = _info[_tar_map_key] - _info[_ref_map_key]

                            if _pb_map_value > _thres:
                                _map_check = False
                                break 
                            else:
                                # zero hit with non-zero threshold
                                if _pb_map_value <= 0 and _thres > 0:
                                    _map_score_dict[_check_key] = _thres / 0.5 #
                                # if both zero hits and zero threshold ,skip
                                elif _pb_map_value <= 0 and _thres <= 0: 
                                    _map_score_dict[_check_key] = np.nan 
                                # have hits with positive threshold
                                else:    
                                    _map_score_dict[_check_key] =  _thres / _pb_map_value
                
                # if passed map check, append 
                if _map_check:
                    # append this probe into selected region probes
                    _sel_reg_pb_dic[_pb] = _info
                    # append the score dict into scores of this region
                    _scores = np.array(list(_map_score_dict.values()))
                    _pb_score = np.nanprod(_scores) ** (1/np.sum(np.isnan(_scores)==False))
                    _pb_score_dict[_pb] = _pb_score
                    #_pb_score_dict[_pb] = _map_score_dict # debug
                
            if self.verbose:
                print(f"--- {len(_sel_reg_pb_dic)} probes passed check_dic selection.")
                
            # initialize kept flag (for two-strands)
            _kept_flags = -1 * np.ones([2, len(_seq)], dtype=np.int)
            _kept_pbs = []
            if pick_probe_by_hits:
                ## after calculating all scores, selecte the best probes
                # extract probes and scores
                _pbs = np.array(list(_sel_reg_pb_dic.keys()))
                _scores = np.array(list(_pb_score_dict.values()))
                # calculate unique scores, pick amoung thess good probes
                _unique_scores = np.unique(_scores)

                # pick highest-scored probes first:
                for _s in _unique_scores[::-1]:
                    # find probes with certain high score:
                    _indices = np.where(_scores==_s)[0]
                    _sel_pbs = _pbs[_indices]
                    _sel_pb_indices = np.array([_sel_reg_pb_dic[_pb]['pb_index'] for _pb in _sel_pbs])
                    # sort based on pb_index
                    _sel_pbs = _sel_pbs[np.argsort(_sel_pb_indices)]
                    _sel_pb_indices = np.array([_sel_reg_pb_dic[_pb]['pb_index'] for _pb in _sel_pbs])

                    # add these probes into kept pbs if distances premit
                    for _pb, _ind in zip(_sel_pbs, _sel_pb_indices):
                        # determine start and end point in kept_flags
                        _start = _ind
                        _end = _ind+self.params_dic['pb_len']+self.params_dic['buffer_len']
                        _occupied_flags = _kept_flags[:, _start:_end]
                        # determine strand
                        _info = _sel_reg_pb_dic[_pb]
                        if _info['strand'] == '+':
                            _strand = 1
                        elif _info['strand'] == '-':
                            _strand = 0
                        else:
                            raise ValueError(f"strand information for probe: {_pb} should be either + or -, {_info['strand']} was given.")
                        # if no probes assigned in the neighborhood, pick this probe:
                        if (_occupied_flags < 0).all():
                            # append this probe:
                            _kept_pbs.append(_pb)
                            # update the kept_flags
                            _kept_flags[_strand, _start:_end] = _pb_score_dict[_pb]

                if self.verbose:
                    print(f"finish in {time.time()-_check_start:.3f}s, {len(_kept_pbs)} probes kept.")
                # update kept_probes
                _kept_pb_indices = np.array([_sel_reg_pb_dic[_pb]['pb_index'] for _pb in _kept_pbs])
            else:
                # directly loop through probes
                for _pb, _info in sorted(_sel_reg_pb_dic.items(), key=lambda v:int(v[1]['pb_index'])):
                    # determine start and end point in kept_flags
                    _start = _info['pb_index']
                    _end = _start+self.params_dic['pb_len']+self.params_dic['buffer_len']
                    #print(_start, _end)
                    _occupied_flags = _kept_flags[:, _start:_end]
                    # determine strand
                    if _info['strand'] == '+':
                        _strand = 1
                    elif _info['strand'] == '-':
                        _strand = 0
                    else:
                        raise ValueError(f"strand information for probe: {_pb} should be either + or -, {_info['strand']} was given.")
                    if (_occupied_flags < 0).all():
                        # append this probe:
                        _kept_pbs.append(_pb)
                        # update the kept_flags
                        _kept_flags[_strand, _start:_end] = _pb_score_dict[_pb]

                if self.verbose:
                    print(f"finish in {time.time()-_check_start:.3f}s, {len(_kept_pbs)} probes kept.")
                # update kept_probes
                _kept_pb_indices = np.array([_sel_reg_pb_dic[_pb]['pb_index'] for _pb in _kept_pbs])

            # append probes for this region/gene
            self.kept_probes.update(
                {_pb:_sel_reg_pb_dic[_pb] for _pb in np.array(_kept_pbs)[np.argsort(_kept_pb_indices)] }
                )

        return _sel_reg_pb_dic, _pb_score_dict


    def load_from_file(self, filename=None, load_probes_only=False):
        "load probes from the report file"
        # default is the savefile
        if filename is None:
            filename = self.save_file
        # load from file if exists
        if filename is not None and os.path.isfile(filename):
            if self.verbose:
                print(f"- Loading from savefile: {filename}.")
            dic_save = pickle.load(open(filename,'rb'))
            #internalize loaded values
            for key in list(dic_save.keys()):
                if not load_probes_only or 'probes' in key:
                    setattr(self,key,dic_save[key])
                    print(f"- loading {key} from file")
                    if key =='kept_probes':
                        print(f"- {len(dic_save[key])} filtered probes loaded")

            # set inputs
            if not load_probes_only:
                self.input_names = self.sequence_dic.get('input_names', [])
                self.input_seqs = self.sequence_dic.get('input_seqs', [])
                self.input_num_seqs = self.sequence_dic.get('input_num_seqs', [])
                #internalize paramaters in params_dic
                for key in list(self.params_dic.keys()):
                    setattr(self,key,self.params_dic[key])
            return True
        # otherwise
        else:
            if self.verbose:
                print(f"- Fail to load from savefile: {filename}, file doesn't exist.")
            return False

    def save_to_file(self, filename=None):
        "save probes into a report file"
        if filename is None:
            filename = getattr(self, 'save_file', None)

        if filename is not None:
            if self.verbose:
                print(f"- Save reports into file: {filename}")
            dic_save = {"cand_probes":self.cand_probes,
                        "kept_probes":self.kept_probes,
                        'sequence_dic':self.sequence_dic,
                        'map_dic':self.map_dic,
                        'params_dic':self.params_dic,
                        'check_dic':self.check_dic,
                        'save_file':self.save_file}
            # save
            pickle.dump(dic_save,open(self.save_file,'wb'))
        else:
            if self.verbose:
                print(f"- Fail to save into file: {filename}, invalid directory.")

    def plot_reports(self, bin):
        pass 

def pick_cand_probes(pb_dict, pb_score_dict, 
                     buffer_len=2, rev_com_ratio=0.8, kept_score_ratio=3.6):
    """NOT FINISHED
    Function to pick probes from a dictionary of candidate probe
    Inputs:
        pb_dict: probe dictionary, probe_sequence -> probe_information, dict
        pb_score_dict: probe dictionary, probe_sequence -> probe_score, dict
        buffer_len: buffer length between neighboring probes, int (default: 2)
        rev_com_ratio: scoring for reverse-complimented sequence, float (default: 0.8)
        kept_score_ratio: ratio of new probe score over existing probes to keep the new probe, float (default: 3.6)
    """
    # auto detect length of sequence
    _pb_indices = np.array([_info['pb_index'] for _pb, _info in pb_dict.items()])
    _pb_lens = np.array([len(_pb) for _pb, _info in pb_dict.items()])
    _len_seq = np.max(_pb_indices) + _pb_lens[np.argmax(_pb_indices)]
    
    # auto detect seq length and two stranded:
    _strand_usages = np.unique([_info['strand'] for _pb, _info in pb_dict.items()])
    if '+' in _strand_usages and '-' in _strand_usages:
        _two_stranded = True
    else:
        _two_stranded = False
        
    # extract seqs, infos and scores
    _seqs, _infos, _scores = [], [], []
    for _pb_seq, _info in pb_dict.items():
        _score = pb_score_dict[_pb_seq]
        _seqs.append(_pb_seq)
        _infos.append(_info)
        _scores.append(_score)
        
    # initalize kept_flags
    _kept_flags = np.zeros([1+int(_two_stranded), _len_seq])
    _kept_pbs = [[] for _i in range(_len_seq)]
    # assign highest score probes first
    for _i in np.argsort(_scores)[::-1]:
        # extract probe information
        _seq = _seqs[_i]
        _info = _infos[_i]
        _score = _scores[_i]
        _start = int(_info['pb_index'])
        _end = _start + len(_seq) + buffer_len
        if _two_stranded:
            if _info['strand'] == '+':
                _strand = 1
            else:
                _strand = 0
        else:
            _strand = 0
        # get target score
        _kept_score = np.max(_kept_flags[:, _start:_end])
        # if no probe exist, add a probe
        if _kept_score == 0:
            _kept_pbs[_start].append(_seq)
            _kept_flags[_strand, _start:_end] += _score
            if _two_stranded:
                _kept_flags[1-_strand, _start:_end] += rev_com_ratio * _score
        # if existing probes are bad:
        if _kept_score * kept_score_ratio < _score:
            pass
        
    return _kept_pbs, _kept_flags
    
    
def consume_sequence_to_int(seq, word_len=17, verbose=False):
    """Function to comsume a sequence, slice it into pieces of word_len, 
    and apply seq2int to convert into integer
    Input Args:
        seq: string of sequence (from genome assembly etc.), string of string of byte
        word_len: length of each unit, int (default: 17, 17**4 > genome size)
        verbose: report messages!, bool (default: False)
    Outputs:
        seq_ints: converted integers representing sequences, list of ints
    """
    if len(seq) > word_len:
        if verbose:
            print(f"-- converting seq of length={len(seq)} into ints.")
        # extract sequences
        seq_pieces = []
        for i in range(len(seq)-word_len+1):
            _seq = seq[i:i+word_len]
            if isinstance(_seq, str):
                seq_pieces.append(_seq.encode())
            elif isinstance(_seq, bytes):
                seq_pieces.append(_seq)
            else:
                raise TypeError(f"Wrong input type for sequence:{_seq}")
        if verbose:
            start = time.time()
        seq_ints = list(map(seq2Int,seq_pieces))
        if verbose:
            end=time.time()
            print('-- time to compute seq2Int:',end-start)
        return seq_ints
    else:
        return []


def select_probes_by_counts(kept_pb_dict, num_kept_pbs, count_type='genome', verbose=True):
    """Roughly select a subset of probes of lowest hits given certain count type
    Inputs:
        kept_pb_dict: kept_probes from library_tools.design.pb_designer_class, dict
        num_kept_pbs: number of kept probes, int
        count_type: type of intmap, str (default: genome)
        verbose: stdout switch, bool (default: True)
        
    """
    if verbose:
        print(f"- filtering {len(kept_pb_dict)} probes by {count_type} counts,", end=' ')
    
    for _seq, _info in kept_pb_dict:
        
        
        pass