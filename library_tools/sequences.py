import numpy as np
import sys, os, glob, time
from . import LibraryDesigner as ld
from . import LibraryTools as lt
# biopython imports
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

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
                         'Name': _info[3],
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
    Dependencies:
    ld.fastaread, ld.fastawrite, ./parse_region'''
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
    for _reg_dict in reg_dicts:
        _chrom, _start, _stop = parse_region(_reg_dict)
        # search all genomic files to match this region
        _wholechr = None 
        for _filename in _all_fasta_files:
            # load fasta file
            with open(_filename, 'r') as _handle:
                for _record in SeqIO.parse(_handle, "fasta"):
                    if _record.id == _chrom:    
                        if verbose:
                            print(f"-- a match found in file: {_filename}.")
                        _wholechr = _record.seq
                        break
            # stop searching if chromosome already exists    
            if _wholechr is not None:
                break     
        # report error if this region not found
        if _wholechr is None:
            raise ValueError(f"Chromosome: {_chrom} doesn't exist in genome reference.")
        # number of regions
        _gene_start = np.max([0, np.int(_start-flanking)])
        _gene_stop = np.min([len(_wholechr), np.int(_stop+flanking)])
        _n_reg = int(np.ceil( float(_gene_stop - _gene_start) / resolution))
        # extract all required seq
        #_whole_seq = _wholechr[_gene_start: min(_gene_start+_n_reg*resolution, len(_wholechr))]
        # extract each region
        _kept_seqs = []
        for _i in range(_n_reg):
            # end location
            _reg_start = int(_gene_start + _i * resolution)
            _reg_end = np.min([_reg_start+resolution, len(_wholechr)])
            #_end_loc = min((_i+1)*resolution, len(_wholechr))
            # extract sequence for this region
            _seq = _wholechr[_reg_start:_reg_end]
            _name = f"{_chrom}:{_reg_start}-{_reg_end}_reg_"
            if "Gene" in _reg_dict:
                _name += _reg_dict['Gene']+'-'
            _name += str(_i+1)
            # append
            _record = SeqRecord(_seq, id=_name, name='', description='')
            _kept_seqs.append(_record)
        if "Gene" in _reg_dict:
            kept_seqs_dict[_reg_dict['Gene']] = _kept_seqs
        else:
            kept_seqs_dict['all'].extend(_kept_seqs)
    
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
            _save_filename = os.path.join(_gene_save_folder, f"{_gene}_reg_1-{len(_records)}.fasta")
            if verbose:
                print(f"-- save to file: {_save_filename}")
            with open(_save_filename, 'w') as _output_handle:
                SeqIO.write(_records, _output_handle, "fasta")
        else:
            for _i, _record in enumerate(_records):
                _save_filename = os.path.join(_gene_save_folder, f"{_gene}_reg_{_i+1}.fasta")
                if verbose:
                    print(f"-- save to file: {_save_filename}")
                with open(_save_filename, 'w') as _output_handle:
                    SeqIO.write([_record], _output_handle, "fasta")
    
    return kept_seqs_dict