import numpy as np
import sys, os, glob, time
# biopython imports
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class fasta_reader():
    """Basic class to read fasta files and parse into SeqRecord"""
    def __init__(self, in_files, verbose=False):
        # inherit
        super().__init__()
        
        self.verbose = verbose
        if isinstance(in_files, list):
            self.in_files = in_files
        elif isinstance(in_files, str):
            self.in_files = [in_files]
        else:
            if self.verbose:
                print(f"No valid in_files given.")
            self.in_files = []
        return

    def load(self):
        if self.verbose:
            print(f"loading {len(self.in_files)} fasta files")
        if not hasattr(self, 'records'):
            self.records = []

        for _in_file in self.in_files:
            with open(_in_file, 'r') as _fp:
                if self.verbose:
                    print(f"- loading from file: {_in_file}")
                _records = []
                for _r in SeqIO.parse(_fp, "fasta"):
                    _records.append(_r)
                self.records.extend(_records)
        return self.records




def read_region_file(filename, verbose=True):
    '''Sub-function to read region file'''
    if verbose:
        print (f'Input region file is: {filename}')
    
    # option 1: split txt region files
    if filename.split(os.extsep)[-1] == 'txt':
        with open(filename, 'r') as _reg_file:
            # start reading
            _lines = _reg_file.read().split('\n')
            _titles = _lines[0].split('\t')
            # save a list of dictionaries
            _reg_list = []
            for _line in _lines[1:]:
                _reg_dic = {} # dinctionary to save all informations
                _info = _line.split('\t') # split informations
                if len(_info) != len(_titles): # sanity check to make sure they are of the same size
                    continue
                for _i in range(len(_info)): # save info to dic
                    _reg_dic[_titles[_i]] = _info[_i]
                _reg_list.append(_reg_dic) # save dic to list
    # option 2: split rna txt file
    elif filename.split(os.extsep)[-1] == 'bed':
        _reg_list = []
        with open(filename, 'r') as _reg_file:
            # start reading
            _lines = _reg_file.read().split('\n')
            for _line in _lines:
                _info = _line.split('\t') # split informations
                if len(_info) < 4:
                    continue 
                
                # directly parse
                _dict = {'Chr': _info[0],
                         'Start': _info[1],
                         'End': _info[2],
                         'Name': _info[3].replace('_','-'),
                         } 
                if len(_info) >= 5:
                    _dict['Score'] = _info[4]      
                if len(_info) >= 6:
                    _dict['Strand'] = _info[5]      
                # try to match Txt version:
                if 'chr' in _info[0]:
                    _cname = _info[0].split('chr')[1]
                else:
                    _cname = _info[0]
                _dict.update(
                    {
                        'Gene': _dict['Name'],
                        'Region': f"{_cname}:{_dict['Start']}-{_dict['End']}",
                    }
                )
                _reg_list.append(_dict)

    else:
        raise IOError(f"input file type not supported.")

    if verbose:
        print(f"- {len(_reg_list)} regions loaded from file: {os.path.basename(filename)}")

    return _reg_list

def parse_region(reg_dic):
    '''given a dictionary of one region, 
    report:
        _chrom: str
        _start: int
        _stop: int'''
    region_str = reg_dic['Region']
    # grab chromosome
    _chrom = region_str.split(':')[0]
    _locus = region_str.split(':')[1]
    # grab start and stop positions
    _start, _stop = _locus.split('-')
    _start = int(_start.replace(',', ''))
    _stop = int(_stop.replace(',', ''))
    # return in this order:
    return _chrom, _start, _stop

def extract_sequence(reg_dicts, genome_reference, 
                     resolution=10000, flanking=0, 
                     split_gene_folder=False, merge=False, verbose=True,
                     save=True, save_folder=None,
                     ):
    '''sub-function to extract sequences of one locus
    Given:
    reg_dic: dic for region info, dictionary
    genome_reference: dir for genome files, str
    resolution: resolution of each region in bp, int
    flanking: upstream and downstream included in bp, int
    save: if save as fasta files, bool
    Return:
    dic of sequences of designed regions
    '''
    # check input
    if save and save_folder is None:
        raise ValueError(f"save_folder should be given if save is specified.")    
    if isinstance(reg_dicts, dict):
        reg_dicts = [reg_dicts]
    elif isinstance(reg_dicts, list):
        pass
    else:
        raise TypeError("Wrong input type for reg_dicts")
        
    # load all fasta files from given genomic folder
    if isinstance(genome_reference, list) \
        and len(genome_reference) > 0 \
        and isinstance(genome_reference[0], SeqRecord):
        _all_fasta_files = genome_reference
    elif isinstance(genome_reference, str) and os.path.isdir(genome_reference):
        _all_fasta_files = [os.path.join(genome_reference, _fl) 
                            for _fl in os.listdir(genome_reference) \
                            if _fl.split(os.extsep)[-1]== 'fa' or _fl.split(os.extsep)[-1]== 'fasta']
    else:
        raise ValueError(f"genome_reference should be either list of SeqReord or pwd of a folder.")
        
    if verbose:
        print(f"-- searching among {len(_all_fasta_files)} references")
    
    # initialize kept records
    kept_seqs_dict = {'all':[]}
    
    # loop thorugh regions
    for _reg_id, _reg_dict in enumerate(reg_dicts):
        _chrom, _start, _stop = parse_region(_reg_dict)
        # search all genomic files to match this region
        _wholechr = None 
        for _filename in _all_fasta_files:
            if isinstance(_filename, str):
                # load fasta file
                with open(_filename, 'r') as _handle:
                    for _record in SeqIO.parse(_handle, "fasta"):
                        if _record.id == _chrom:    
                            if verbose:
                                print(f"-- a match found in file: {_filename}.")
                            _wholechr = _record.seq
                            break
            elif isinstance(_filename, SeqRecord):
                _record = _filename
                if _record.id == _chrom:    
                    if verbose:
                        print(f"-- a match found in record: {_record.id}.")
                    _wholechr = _record.seq
                    break
            else:
                continue 
            # stop searching if chromosome already exists    
            if _wholechr is not None:
                break     
        # report error if this region not found
        if _wholechr is None:
            raise ValueError(f"Chromosome: {_chrom} doesn't exist in genome reference.")
        # number of regions
        _gene_start = np.max([0, np.int(_start-flanking)])
        _gene_stop = np.min([len(_wholechr), np.int(_stop+flanking)])

        # case 1: resolution specified
        if resolution > 0:
            _n_reg = int(np.ceil( float(_gene_stop - _gene_start) / resolution))
            # extract all required seq
            #_whole_seq = _wholechr[_gene_start: min(_gene_start+_n_reg*resolution, len(_wholechr))]
            # extract each region
            _kept_seqs = []
            for _i in range(_n_reg):
                # end location
                _reg_start = int(_gene_start-1 + _i * resolution)
                _reg_end = np.min([_reg_start + resolution, len(_wholechr)])
                #_end_loc = min((_i+1)*resolution, len(_wholechr))
                # extract sequence for this region
                _seq = _wholechr[_reg_start:_reg_end]
                _name = f"{_chrom}:{_reg_start}-{_reg_end}_"
                if 'Strand' in _reg_dict:
                    _name += f"strand_{_reg_dict['Strand']}_"
                # gene
                if "Gene" in _reg_dict:
                    _name += f"gene_{_reg_dict['Gene']}-seg-{_i+1}"
                else:
                    _name += 'reg_'+str(_i+1)
                # append
                if "Strand" in _reg_dict and _reg_dict['Strand'] == '-':
                    _record = SeqRecord(_seq.reverse_complement(), id=_name, name='', description='')
                else:
                    _record = SeqRecord(_seq, id=_name, name='', description='')
                _kept_seqs.append(_record)
            if "Gene" in _reg_dict:
                kept_seqs_dict[_reg_dict['Gene']] = _kept_seqs
            else:
                kept_seqs_dict['all'].extend(_kept_seqs)
        # case 2: no valid resolution give, get the whole sequence
        ## NOTICE: gene_start and gene_end should minus 1 because the coordinate start with 1 but python index start with 0
        ## NOTICE: gene_end for the whole sequence should +1 because the ending base-pair is included in 
        ## this has been checked with publically available ensembl fasta reference
        elif resolution <= 0:
            _seq = _wholechr[_gene_start-1:_gene_stop-1+1] # 0 is not allowed as a gene start
            _name = f"{_chrom}:{_gene_start}-{_gene_stop}_"
            if 'Strand' in _reg_dict:
                _name += f"strand_{_reg_dict['Strand']}_"
            if "Gene" in _reg_dict:
                _name += f"gene_{_reg_dict['Gene']}_"
            #_name += 'reg_'
            if _name[-1] == '_':
                _name = _name[:-1]
            # append
            if "Strand" in _reg_dict and _reg_dict['Strand'] == '-':
                _record = SeqRecord(_seq.reverse_complement(), id=_name, name='', description='')
            else:
                _record = SeqRecord(_seq, id=_name, name='', description='')

            if "Gene" in _reg_dict:
                kept_seqs_dict[_reg_dict['Gene']] = [_record]
            else:
                kept_seqs_dict['all'].extend([_record])
    
    # save
    if save:
        if verbose:
            print(f"-- saving sequences into folder: {save_folder}")
    
        for _gene, _records in kept_seqs_dict.items():
            # decide savefolder for this gene
            if split_gene_folder:
                _gene_save_folder = os.path.join(save_folder, _gene)
            else:
                _gene_save_folder = save_folder
            # save
            if merge:
                _save_filename = os.path.join(_gene_save_folder, f"{_gene}-seg-1-{len(_records)}.fasta")
                if verbose:
                    print(f"-- save to file: {_save_filename}")
                with open(_save_filename, 'w') as _output_handle:
                    SeqIO.write(_records, _output_handle, "fasta")
            else:
                for _i, _record in enumerate(_records):
                    _save_filename = os.path.join(_gene_save_folder, f"{_gene}-seg-{_i+1}.fasta")
                    if verbose:
                        print(f"-- save to file: {_save_filename}")
                    with open(_save_filename, 'w') as _output_handle:
                        SeqIO.write([_record], _output_handle, "fasta")
    
    return kept_seqs_dict






def generate_flags_for_isoforms(gene_dict, plot_flags=True):
    """Generate a flag vector, which is used for further parsing of exons/introns
    exons are x2, 5'UTRs are x5, 3'UTRs are x3, introns are x1.
    """
    exon_flags = []
    mrna_limits = []
    mrna_names = []
    for _i, _mrna_dict in enumerate(gene_dict['Children']):
        _mstart = int(_mrna_dict['start'])
        _mend = int(_mrna_dict['end'])
        ## NOTICE: the base-pair at region_end should be included in this region
        _exon_flag = np.ones(_mend - _mstart+1, dtype=np.int)
        # find name
        if 'Name' in _mrna_dict['infos']:
            _mname = _mrna_dict['infos']['Name']
        else:
            _mname = f"{gene_dict['infos']['Name']}-{_i+1}"
        # load exons
        for _dict in _mrna_dict['Children']:
            if _dict['type'] == 'exon':
                _estart = int(_dict['start'])
                _eend = int(_dict['end'])
                _exon_flag[_estart-_mstart:_eend+1-_mstart] *= 2
            if _dict['type'] == 'five_prime_UTR':
                _estart = int(_dict['start'])
                _eend = int(_dict['end'])
                _exon_flag[_estart-_mstart:_eend+1-_mstart] *= 5
            if _dict['type'] == 'three_prime_UTR':
                _estart = int(_dict['start'])
                _eend = int(_dict['end'])
                _exon_flag[_estart-_mstart:_eend+1-_mstart] *= 3
        # append
        exon_flags.append(_exon_flag)
        mrna_limits.append([_mstart, _mend])
        mrna_names.append(_mname)
        
    if plot_flags:
        import matplotlib.pyplot as plt
        plt.figure()
        for _i, ((_mstart,_mend), _exon_flag, _mname) in enumerate(zip(mrna_limits, exon_flags, mrna_names)):
            plt.plot(np.arange(_mstart, _mend+1), _exon_flag+0.5/(len(exon_flags))*_i, label=_mname)
        plt.xlabel(f"Coordinate on {gene_dict['seqid']}")
        plt.ylabel(f"Flag value")

        plt.legend()
        plt.show()
    
    return exon_flags, mrna_limits, mrna_names

class sequence_reader:
    """Class to prepare sequences to read"""
    
    def __init__(
        self, genome_folder, resolution=0, flanking=0, 
        auto_load_ref=False, overwrite=False, verbose=True):
        # save the input attributes
        self.genome_folder = genome_folder
        self.resolution = resolution # size of each region in bp, 0 means the whole region
        self.flanking = flanking # size of flanking region in bp
        self.overwrite = overwrite # whether overwrite existing sequences and savefiles
        self.verbose = verbose
        # automatically search fasta files to extract sequence
        if isinstance(self.genome_folder, str) and os.path.isdir(self.genome_folder):
            self.input_files = [os.path.join(self.genome_folder, _fl) 
                                for _fl in os.listdir(self.genome_folder) \
                                if _fl.split(os.extsep)[-1]== 'fa' or _fl.split(os.extsep)[-1]== 'fasta']
        else:
            raise ValueError(f"Failed to find sequences from genome_folder")
        # initialize references and final sequences
        self.ref_seq_dict = {}
        self.seq_dict = {}
        # inherit from superclass
        super().__init__()
        # auto load if necessary
        if auto_load_ref:
            self.load_ref_sequences()
        return

    def __str__(self, infotype='all'):
        _str = ''
        if infotype == 'input' or infotype == 'all':
            _str += f"reader for region: {self.region_dict}\n"
            _str += f"load sequence from folder: {self.genome_folder}\n"
        return _str
    
    def load_ref_sequences(self):
        # search through files
        for _fl in self.input_files:
            # load fasta file
            with open(_fl, 'r') as _handle:
                for _record in SeqIO.parse(_handle, "fasta"):
                    if _record.id not in self.ref_seq_dict or self.overwrite:    
                        if self.verbose:
                            print(f"-- load sequence: {_record.id}, size={len(_record)}")
                        self.ref_seq_dict[_record.id] = _record
        # summarize
        if self.verbose:
            print(f"- {len(self.ref_seq_dict)} sequences loaded.")
    
    def find_sequence_for_region(self, reg_dict, save=True):
        """find sequence from single reg_dict"""
        # find chromosome
        if isinstance(reg_dict, dict):
            _ref = [self.ref_seq_dict[str(reg_dict['Chr'])]]
        elif isinstance(reg_dict, list):
            _ref = []
            for _chr in np.unique([str(_r['Chr']) for _r in reg_dict]):
                _ref.append(self.ref_seq_dict[_chr])

        seqs = extract_sequence(reg_dict, _ref, 
                                resolution=self.resolution,
                                flanking=self.flanking,
                                save=False, verbose=self.verbose,
                                )
        # update
        if save:
            self.seq_dict.update({_k:_v for _k,_v in seqs.items() if len(_v)>0})
        return seqs
    
    def save_sequences(self, save_folder, merge=False, split_gene_folder=False):
        if self.verbose:
            print(f"-- saving sequences into folder: {save_folder}")
        
        for _gene, _records in self.seq_dict.items():
            # decide savefolder for this gene
            if split_gene_folder:
                _gene_save_folder = os.path.join(save_folder, _gene)
            else:
                _gene_save_folder = save_folder
            # save
            if merge:
                _save_filename = os.path.join(_gene_save_folder, f"{_gene}-seg-1-{len(_records)}.fasta")
                if self.verbose:
                    print(f"-- save to file: {_save_filename}")
                with open(_save_filename, 'w') as _output_handle:
                    SeqIO.write(_records, _output_handle, "fasta")
            else:
                for _i, _record in enumerate(_records):
                    _save_filename = os.path.join(_gene_save_folder, f"{_gene}-seg-{_i}.fasta")
                    if self.verbose:
                        print(f"-- save to file: {_save_filename}")
                    with open(_save_filename, 'w') as _output_handle:
                        SeqIO.write([_record], _output_handle, "fasta")

class RNA_sequence_reader(sequence_reader):
    def __init__(self, genome_folder, resolution=0, flanking=0, overwrite=False, verbose=True):
        # inherit from superclass: reference_reader
        super(RNA_sequence_reader, self).__init__(genome_folder, 
                                                  resolution, flanking, 
                                                  overwrite, verbose)
        # add unique attrs
        self.isoform_mRNA_dict = {}
        self.isoform_intron_dict = {}
        self.small_exon_dict = {}
        self.small_intron_dict = {}
        
    def find_transcript_isoforms(self, gene_dicts, plot_flags=True, save=True):
        # check inputs
        if isinstance(gene_dicts, dict):
            gene_dicts = [gene_dicts]
        elif isinstance(gene_dicts, list):
            pass
        else:
            raise TypeError
        # initialize output sequences
        transcript_dict = {}
        # loop through gene dicts
        for gene_dict in gene_dicts:
            # initialize
            gene_isoform_dict = {}
            # get gene name
            gene_name = gene_dict['infos']['Name']
            if self.verbose:
                print(f"- find mRNA for gene: {gene_name}")
            ## step 1: find flags for each isoform
            exon_flags, mrna_limits, mrna_names = generate_flags_for_isoforms(gene_dict, plot_flags=plot_flags)
            ## step 2: for each isoform, load exon sequence
            for _i, (_mrna_dict, _exon_flag, (_mstart,_mend), _mname) in enumerate(zip(gene_dict['Children'], exon_flags, mrna_limits, mrna_names)):
                # convert full-length mrna_dict into reg_dict
                _mrna_reg_dict = gene_dict_2_reg_dict(_mrna_dict)
                _mrna_reg_dict['Strand'] = '+'
                # extract positive strand
                _mrna_seq_dict = self.find_sequence_for_region(_mrna_reg_dict, save=False)
                _mrna_seq = [_v for _k,_v in _mrna_seq_dict.items() if len(_v)>0][0][0]
                # load required sequencenp.where(_exon_flag>1)[0]
                _exon_inds = np.where(_exon_flag>1)[0]
                _sel_seq = Seq(''.join(np.array(_mrna_seq.seq)[_exon_inds])) 
                # invert the sequence if minus_strand
                if _mrna_dict['strand'] == '-':
                    _sel_seq = _sel_seq.reverse_complement()
                # assemble records
                _name = '_'.join(['gene', f"{gene_name}",
                                'id', f"{_mrna_dict['infos']['ID']}",
                                'name', f"{_mname}", 
                                'type', f"{_mrna_dict['type']}",
                                'strand', f"{_mrna_dict['strand']}",
                                'loc', f"{_mrna_dict['seqid']}:{_mstart}-{_mend}",
                                ])
                _record = SeqRecord(_sel_seq, id=_name, name='', description='')
                # append
                if self.verbose:
                    print(f"-- extracting transcript for {_name}, size={len(_sel_seq)}")
                gene_isoform_dict[_mname] = [_record]
            # append
            transcript_dict[gene_name] = gene_isoform_dict

        if save:
            self.isoform_mRNA_dict.update(transcript_dict)

        return transcript_dict
    
    def find_introns_for_isoforms(self, gene_dicts, plot_flags=True, save=True):
        # check inputs
        if isinstance(gene_dicts, dict):
            gene_dicts = [gene_dicts]
        elif isinstance(gene_dicts, list):
            pass
        else:
            raise TypeError
        ## initialize output sequences
        intron_record_dict = {}
        # loop through gene dicts
        for gene_dict in gene_dicts:
            # get gene name
            gene_name = gene_dict['infos']['Name']
            # initialize
            gene_intron_dict = {}
            # get gene name
            gene_name = gene_dict['infos']['Name']
            if self.verbose:
                print(f"- find mRNA for gene: {gene_name}")
            ## step 1: find flags for each isoform
            exon_flags, mrna_limits, mrna_names = generate_flags_for_isoforms(gene_dict, plot_flags=plot_flags)
            ## step 2: for each isoform, load exon sequence
            for _i, (_mrna_dict, _exon_flag, (_mstart,_mend), _mname) in enumerate(zip(gene_dict['Children'], exon_flags, mrna_limits, mrna_names)):
                # convert full-length mrna_dict into reg_dict
                _mrna_reg_dict = gene_dict_2_reg_dict(_mrna_dict)
                _mrna_reg_dict['Strand'] = '+'
                # extract positive strand
                _mrna_seq_dict = self.find_sequence_for_region(_mrna_reg_dict, save=False)
                _mrna_reg = [_v for _k,_v in _mrna_seq_dict.items() if len(_v)>0][0][0]
                # find indices for introns
                _intron_index_list = []
                _inds = []
                _prev_ind = 0
                for _i in np.where(_exon_flag==1)[0]: 
                    if len(_inds) == 0 or _prev_ind == _i-1:
                        _inds.append(_i)
                        _prev_ind = _i
                    else:
                        _intron_index_list.append(np.array(_inds))
                        _inds = []
                if len(_inds) > 0:
                    _intron_index_list.append(_inds)
                # flip the order if negative strand
                if _mrna_dict['strand'] == '-':
                    _intron_index_list = [_inds for _inds in _intron_index_list[::-1]]
                # loop through intron indices to extract intronic sequences
                _intron_records = []
                for _intron_id, _inds in enumerate(_intron_index_list):
                    # load required sequence
                    _sel_seq = Seq(''.join(np.array(_mrna_reg.seq)[_inds]))
                    # invert the sequence if minus_strand
                    if _mrna_dict['strand'] == '-':
                        _sel_seq = _sel_seq.reverse_complement()
                    # assemble records
                    _name = '_'.join(['gene', f"{gene_name}",
                                    'id', f"{_mrna_dict['infos']['ID']}",
                                    'name', f"{_mname}-intron-{_intron_id+1}", 
                                    'type', f"intron",
                                    'strand', f"{_mrna_dict['strand']}",
                                    'loc', f"{_mrna_dict['seqid']}:{_mstart}-{_mend}",
                                    ])
                    _record = SeqRecord(_sel_seq, id=_name, name='', description='')
                    _intron_records.append(_record)
                # store    
                gene_intron_dict[_mname] = _intron_records
            # append
            intron_record_dict[gene_name] = gene_intron_dict

        if save:
            self.isoform_mRNA_dict.update(intron_record_dict)

        return intron_record_dict

    def find_smallest_exons(self, gene_dicts, plot_flags=True, save=True):
        """Merging all the isoforms and generate the finest splitted exons (including 5' and 3' UTRs)"""
        # check inputs
        if isinstance(gene_dicts, dict):
            gene_dicts = [gene_dicts]
        elif isinstance(gene_dicts, list):
            pass
        else:
            raise TypeError
        ## initialize output sequences
        sm_exon_record_dict = {}
        # loop through gene_dicts
        for gene_dict in gene_dicts:
            # get gene name
            gene_name = gene_dict['infos']['Name']
            if self.verbose:
                print(f"- find smallest exons for gene: {gene_name}")
            ## step 1: find flags for each isoform
            exon_flags, mrna_limits, mrna_names = generate_flags_for_isoforms(gene_dict, plot_flags=plot_flags)
            # find a merged mrna_limit
            merged_start, merged_end = np.min([_l[0] for _l in mrna_limits]), np.max([_l[1] for _l in mrna_limits])
            # generate a merged flag for smallest exon
            merged_flags = np.ones([merged_end + 1 - merged_start, len(exon_flags)], dtype=np.int) *np.nan
            
            for _i, (_exon_flag, (_mstart,_mend), _mname) in enumerate(zip(exon_flags, mrna_limits, mrna_names)):
                merged_flags[_mstart-merged_start: _mend+1-merged_start, _i] = _exon_flag
            
            # extract smallest exons
            sm_exon_flag = np.nanmin(merged_flags, axis=1)
            # find indices for introns
            sm_exon_index_list = []
            _inds = []
            _prev_ind = 0
            for _i in np.where(sm_exon_flag > 1)[0]: 
                if len(_inds) == 0 or _prev_ind == _i-1:
                    _inds.append(_i)
                    _prev_ind = _i
                else:
                    sm_exon_index_list.append(np.array(_inds))
                    _inds = []
            if len(_inds) > 0:
                sm_exon_index_list.append(_inds)
            # flip the order if negative strand
            if gene_dict['strand'] == '-':
                sm_exon_index_list = [_inds for _inds in sm_exon_index_list[::-1]]

            ## load sequence for the entire gene
            # convert full-length gene_dict into reg_dict
            _gene_reg_dict = gene_dict_2_reg_dict(gene_dict)
            _gene_reg_dict['Strand'] = '+'
            # extract positive strand
            _gene_record_dict = self.find_sequence_for_region(_gene_reg_dict, save=False)
            _gene_record = [_v for _k,_v in _gene_record_dict.items() if len(_v)>0][0][0]
            ## loop through intron indices to extract intronic sequences
            sm_exon_records = []
            for sm_exon_id, _inds in enumerate(sm_exon_index_list):
                # load required sequence
                _sel_seq = Seq(''.join(np.array(_gene_record.seq)[_inds]))
                # invert the sequence if minus_strand
                if gene_dict['strand'] == '-':
                    _sel_seq = _sel_seq.reverse_complement()
                # assemble records
                _name = '_'.join(['gene', f"{gene_name}",
                                'id', f"{gene_dict['infos']['ID']}",
                                'name', f"{_mname}-smexon-{sm_exon_id+1}", 
                                'type', f"smexon",
                                'strand', f"{gene_dict['strand']}",
                                'loc', f"{gene_dict['seqid']}:{merged_start+min(_inds)}-{merged_start+max(_inds)}",
                                ])
                _record = SeqRecord(_sel_seq, id=_name, name='', description='')
                sm_exon_records.append(_record)
            # store    
            sm_exon_record_dict[gene_name] = sm_exon_records

        if save:
            self.small_exon_dict.update(sm_exon_record_dict)

        return sm_exon_record_dict
        
            
    def find_smallest_introns(self, gene_dicts, plot_flags=True, save=True):
        """Merging all the isoforms and generate the finest splitted introns"""
        # check inputs
        if isinstance(gene_dicts, dict):
            gene_dicts = [gene_dicts]
        elif isinstance(gene_dicts, list):
            pass
        else:
            raise TypeError
        ## initialize output sequences
        sm_intron_record_dict = {}
        # loop through gene_dicts
        for gene_dict in gene_dicts:
            # get gene name
            gene_name = gene_dict['infos']['Name']
            ## step 1: find flags for each isoform
            intron_flags, mrna_limits, mrna_names = generate_flags_for_isoforms(gene_dict, plot_flags=plot_flags)
            ## step 2: generate a merged flag indicating introns
            # find a merged mrna_limit
            merged_start, merged_end = np.min([_l[0] for _l in mrna_limits]), np.max([_l[1] for _l in mrna_limits])
            # generate a merged flag for smallest intron
            merged_flags = np.ones([merged_end + 1 - merged_start, len(intron_flags)], dtype=np.int) *np.nan
            
            for _i, (_intron_flag, (_mstart,_mend), _mname) in enumerate(zip(intron_flags, mrna_limits, mrna_names)):
                merged_flags[_mstart-merged_start: _mend+1-merged_start, _i] = _intron_flag
            # extract smallest introns
            sm_intron_flag = np.nanmax(merged_flags, axis=1)

            ## step 3: find indices for introns
            sm_intron_index_list = []
            _inds = []
            _prev_ind = 0
            for _i in np.where(sm_intron_flag == 1)[0]: 
                if len(_inds) == 0 or _prev_ind == _i-1:
                    _inds.append(_i)
                    _prev_ind = _i
                else:
                    sm_intron_index_list.append(np.array(_inds))
                    _inds = []
            if len(_inds) > 0:
                sm_intron_index_list.append(_inds)
            # flip the order if negative strand
            if gene_dict['strand'] == '-':
                sm_intron_index_list = [_inds for _inds in sm_intron_index_list[::-1]]

            ## step 4. load sequence for the entire gene
            # convert full-length gene_dict into reg_dict
            _gene_reg_dict = gene_dict_2_reg_dict(gene_dict)
            _gene_reg_dict['Strand'] = '+'
            # extract positive strand
            _gene_record_dict = self.find_sequence_for_region(_gene_reg_dict, save=False)
            _gene_record = [_v for _k,_v in _gene_record_dict.items() if len(_v)>0][0][0]
            ## loop through intron indices to extract intronic sequences
            sm_intron_records = []
            for sm_intron_id, _inds in enumerate(sm_intron_index_list):
                # load required sequence
                _sel_seq = Seq(''.join(np.array(_gene_record.seq)[_inds]))
                # invert the sequence if minus_strand
                if gene_dict['strand'] == '-':
                    _sel_seq = _sel_seq.reverse_complement()
                # assemble records
                _name = '_'.join(['gene', f"{gene_name}",
                                'id', f"{gene_dict['infos']['ID']}",
                                'name', f"{_mname}-smintron-{sm_intron_id+1}", 
                                'type', f"smintron",
                                'strand', f"{gene_dict['strand']}",
                                'loc', f"{gene_dict['seqid']}:{merged_start+min(_inds)}-{merged_start+max(_inds)}",
                                ])
                _record = SeqRecord(_sel_seq, id=_name, name='', description='')
                sm_intron_records.append(_record)
            # store    
            sm_intron_record_dict[gene_name] = sm_intron_records

        if save:
            self.small_intron_dict.update(sm_intron_record_dict)

        return sm_intron_record_dict

    def _save_sequences_from_dict(self, dict_name, save_folder, merge=False, split_gene_folder=False):
        """Save seuquences from a specific dict into save folder."""
        if not hasattr(self, dict_name):
            raise KeyError(f"{dict_name} doesn't exist in class, exit.")
        # initialize saved filename
        saved_filenames = []
        # extract dict
        _seq_dict = getattr(self, dict_name)
        _dict_savename = dict_name
        if '_dict' in _dict_savename:
            _dict_savename = _dict_savename.split('_dict')[0]
        if self.verbose:
            print(f"-- saving {dict_name} sequences into folder: {save_folder}")
        # loop through all possible genes
        for _gene, _infos in _seq_dict.items():
            # decide savefolder for this gene
            if split_gene_folder:
                _gene_save_folder = os.path.join(save_folder, _gene)
            else:
                _gene_save_folder = save_folder
            # if isoforms involved:
            if isinstance(_infos, list):
                _records = _infos
                # save without isoforms
                if merge:
                    _save_filename = os.path.join(_gene_save_folder, f"{_gene}_{_dict_savename}_1-{len(_records)}.fasta")
                    if self.verbose:
                        print(f"-- save to file: {_save_filename}")
                    with open(_save_filename, 'w') as _output_handle:
                        saved_filenames.append(_save_filename)
                        SeqIO.write(_records, _output_handle, "fasta")
                else:
                    for _i, _record in enumerate(_records):
                        _save_filename = os.path.join(_gene_save_folder, f"{_gene}_{_dict_savename}_{_i+1}.fasta")
                        if self.verbose:
                            print(f"-- save to file: {_save_filename}")
                        with open(_save_filename, 'w') as _output_handle:
                            saved_filenames.append(_save_filename)
                            SeqIO.write([_record], _output_handle, "fasta")
            elif isinstance(_infos, dict):
                # save with isoforms
                if merge:
                    _isoform_records = []
                    for _mname, _records in _infos.items():
                        _isoform_records.extend(_records)
                    _save_filename = os.path.join(_gene_save_folder, f"{_gene}_{_dict_savename}_merged_{len(_infos)}_isoforms.fasta")
                    if self.verbose:
                        print(f"-- save to file: {_save_filename}")
                    with open(_save_filename, 'w') as _output_handle:
                        saved_filenames.append(_save_filename)
                        SeqIO.write(_isoform_records, _output_handle, "fasta")
                else:
                    for _mname, _records in _infos.items():
                        _save_filename = os.path.join(_gene_save_folder, f"{_gene}_{_dict_savename}_isoform_{_mname}.fasta")
                        if self.verbose:
                            print(f"-- save to file: {_save_filename}")
                        with open(_save_filename, 'w') as _output_handle:
                            saved_filenames.append(_save_filename)
                            SeqIO.write(_records, _output_handle, "fasta")
            else:
                raise TypeError
        if self.verbose:
            print(f"- {len(saved_filenames)} fasta files saved.")
        return saved_filenames

def gene_dict_2_reg_dict(_gene_dict):
    """Convert standard output TSS into standard region dict"""
    _reg_dict = {
        'Chr':_gene_dict['seqid'],
        'Start':_gene_dict['start'],
        'End':_gene_dict['end'],
        'Name':f"{_gene_dict['infos']['ID']}-{_gene_dict['infos']['Name']}",
        'Gene':_gene_dict['infos']['Name'],
        'Region':f"{_gene_dict['seqid']}:{_gene_dict['start']}-{_gene_dict['end']}",
        'Strand':_gene_dict['strand'],
    }
    return _reg_dict

def reg_dict_2_tss_dict(reg_dict, reg_size):
    """Convert a standard region_dict into TSS and its flanking dict"""
    # initialize
    _tss_dict = {_k:_v for _k,_v in reg_dict.items()}
    # modify start, end
    if _tss_dict['Strand'] == '+': 
        _reg_center = int(reg_dict['Start'])
    elif _tss_dict['Strand'] == '-': 
        _reg_center = int(reg_dict['End'])
    else:
        raise ValueError(f"Wrong input strand, should be + or -")
    # modify tss 
    _tss_dict["Start"] = _reg_center - int(reg_size/2)
    _tss_dict["End"] = _reg_center + int(reg_size/2)
    _tss_dict["Region"] = f"{reg_dict['Chr']}:{_tss_dict['Start']}-{_tss_dict['End']}"
    # modify names
    _tss_dict["Name"] = _tss_dict["Name"] + f"-TSS-{reg_size}" 
    return _tss_dict