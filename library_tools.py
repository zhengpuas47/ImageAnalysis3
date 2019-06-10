## Tools for library design
import sys,os,re,time,glob
import numpy as np
import pickle as pickle
import matplotlib.pylab as plt

# from ImageAnalysis3
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy,_image_size, _allowed_colors

# biopython
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML 
_readout_folder = r'\\SMIRNOV\Chromatin_NAS_3\Pu\Readouts'
_genome_folder = r'\\SMIRNOV\Chromatin_NAS_3\Pu\Genomes\hg38'

def __init__():
    pass
#-------------------------------------------------------------------------------
## Readout releated
def Extend_Readout(input_seq, target_len=30, add_5p=True):
    '''Function to extend existing short readout to longer one by generation of random sequence
    Inputs:
        input_seq: input short readout, Bio.Seq.Seq object
        target_len: length of target readouts, int
        add_5p: the direction to patch new random seq, bool, default is patch at 5p
    Output:
        out_seq: extended readout, Bio.Seq.Seq object'''
    # imports
    from random import choice
    # Sanity check
    if len(input_seq) >= target_len:
        raise ValueError('input seq length doesnot match target length!')
        out_seq = input_seq[:target_len]
    # Start processing readout
    dna_alphabet = ['A','C','G','T']
    input_seq_string = str(input_seq)
    # first added base should be A/T
    if add_5p:
        input_seq_string = choice(['A','T'])+input_seq_string
    else:
        input_seq_string += choice(['A','T'])
    # random for the rest
    for i in range(target_len - len(input_seq) - 1):
        if add_5p:
            input_seq_string = choice(dna_alphabet)+input_seq_string
        else:
            input_seq_string += choice(dna_alphabet)
    out_seq = Seq(input_seq_string, IUPAC.unambiguous_dna)
    return out_seq


def Filter_Readout(input_seq, 
                   GC_percent=[0.4, 0.6], 
                   max_consecutive=4, 
                   max_rep=6, 
                   C_percent=[0.22, 0.28], 
                   blast_hsp_thres=10.0,
                   readout_folder=_readout_folder,
                   blast_ref='cand_readouts.fasta',
                   verbose=False):
    '''Filter a readout by defined criteria
    Inputs:
    input_seq: the readout sequence, Bio.Seq.Seq object 
    check_GC: whether check gc content, list of two values or False ([0.4, 0.6])
    max_consecutive: maximum allowed consecutive bases, int (4)
    max_rep: maximum replicated sequence in a readout allowed, int (6)
    C_percent: percentage of base C, list of two values or False([0.22, 0.28])
    blast_hsp_thres: threshold for blast hsp, no hsp larger than this allowed, int (10)
    readout_folder: folder to store readout information, string
    blast_ref: file basename for fasta file of existing readouts in readout_folder, used for blast
    Output:
    keep: whether this readout should be kept.
    '''
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.Alphabet import IUPAC
    input_seq = Seq(str(input_seq).upper(), IUPAC.unambiguous_dna)
    
    def _checking_GC(input_seq=input_seq, check_GC=GC_percent):
        if check_GC:
            if max(check_GC) == min(check_GC):
                raise ValueError('Invalid check_GC input vector!')
            from Bio.SeqUtils import GC
            _gc = GC(input_seq) / 100.0
            if verbose:
                print ("GC:", _gc)
            if _gc >= max(check_GC) or _gc <= min(check_GC): # if gc content doesn't pass filter
                return False
        return True
            
    def _checking_consecutive(input_seq=input_seq, max_consecutive=max_consecutive):
        if max_consecutive:
            _seq_str = str(input_seq).upper()
            # Check Consecutive 
            _mask = ['A'*max_consecutive, 'C'*max_consecutive, 'G'*max_consecutive, 'T'*max_consecutive] # mask for consecutive bases
            for i in range(len(_seq_str) - max_consecutive + 1):
                if _seq_str[i:i+max_consecutive] in _mask:
                    return False
        return True
    
    def _checking_repetitive(input_seq=input_seq, max_rep=max_rep):
        if max_rep:
            _seq_str = str(input_seq).upper()
            # check repetitive
            _rep_dic = {}
            for i in range(len(_seq_str) - max_rep + 1):
                if _seq_str[i:i+max_rep] in _rep_dic.keys():
                    return False
                else:
                    _rep_dic[_seq_str[i:i+max_rep]] = 1
        return True
    
    def _checking_C_percent(input_seq=input_seq, C_percent=C_percent):
        if C_percent:
            # global c percent
            if max(C_percent) == min(C_percent):
                raise ValueError('Invalid C_percent input vector!')
            _c_per = input_seq.count('C') / float(len(input_seq))
            if verbose:
                print ("C:", _c_per)
            if _c_per >= max(C_percent) or _c_per <= min(C_percent):
                return False
            else:
                # constraining C in first 12 bases
                _seq_str = str(input_seq).upper()
                for i in range(12-6):
                    if Seq(_seq_str[i:i+6]).count('C') >= 4:
                        return False
        return True
            
    def _checking_blast(input_seq=input_seq, blast_hsp_thres=blast_hsp_thres, 
                        readout_folder=readout_folder, blast_ref=blast_ref):
        import glob, os
        from Bio.Blast.Applications import NcbiblastnCommandline
        from Bio.Blast import NCBIXML 
        # write input_seq into a temp file
        SeqIO.write(SeqRecord(input_seq),  "temp.fasta", "fasta")
        # Run BLAST and parse the output as XML
        output = NcbiblastnCommandline(query="temp.fasta", 
                                       subject=os.path.join(readout_folder, blast_ref), 
                                       evalue=10,
                                       word_size=7,
                                       out='temp_out.xml',
                                       outfmt=5)()[0]
        _blast_record = NCBIXML.read(open(r'temp_out.xml', 'r'))
        _hsp_scores = []
        # parse xml results
        for _alignment in _blast_record.alignments:
            for _hsp in _alignment.hsps:
                _hsp_scores.append(_hsp.score)
        # clean temp files        
        os.remove("temp.fasta")
        os.remove("temp_out.xml")
        if verbose:
            print ("blast:", sum([_hsp_score > blast_hsp_thres for _hsp_score in _hsp_scores]))
        #False for having hsp_score larger than threshold
        _blast_result = sum([_hsp_score > blast_hsp_thres for _hsp_score in _hsp_scores])
        return _blast_result==0
            
    _keep_GC = _checking_GC()
    if not _keep_GC:
        return False
    #print _keep_GC
    _keep_cons = _checking_consecutive()
    if not _keep_cons:
        return False
    #print _keep_cons
    _keep_rep = _checking_repetitive()
    if not _keep_rep:
        return False
    #print _keep_rep
    _keep_c = _checking_C_percent()
    if not _keep_c:
        return False
    #print _keep_c
    _keep_blast = _checking_blast()
    if not _keep_blast:
        return False
    #print _keep_blast
    
    # Merge all keeps
    _keeps = [_keep_GC, _keep_cons, _keep_rep, _keep_c, _keep_blast]
    if verbose:
        print (_keeps)
        
    return True # All true, return true, otherwise False


def _generate_existing_readouts(filenames, readout_folder=_readout_folder, 
                                save_name=r'cand_readouts.fasta', verbose=True):
    """Function to generate one single fasta file containing all existing readouts
    Inputs:
        filenames: list of filenames that contain existing readouts, list of strings
        readout_folder: folder to store readout information, string
        save_name: filename to save merged readout information, string (default: cand_readouts.fasta)
        verbose: say something!, bool (default: True)"""
    _readout_fls = []
    for _fl in filenames:
        if not os.path.isfile(_fl):
            _full_fl = os.path.join(readout_folder, _fl)
            if os.path.isfile(_full_fl):
                _readout_fls.append(_full_fl)
            else:
                raise IOError(f"Wrong input filename, both {_fl} and {_full_fl} doesn't exist.")
    if verbose:
        print(f"- Start merging existing readouts from files: {_readout_fls}")
    # Loading
    _readout_records = []
    for _fl in _readout_fls:
        with open(_fl, "rU") as _handle:
            for _record in SeqIO.parse(_handle, "fasta"):
                _readout_records.append(_record)
    if verbose:
        print(f"-- {len(_readout_records)} readouts are loaded.")
    
    # save
    _save_filename = os.path.join(readout_folder, save_name)
    if verbose:
        print(f"-- saving to file:{_save_filename}")
    with open(_save_filename, "w") as _output_handle:
        SeqIO.write(_readout_records, _output_handle, "fasta")
    
    return _readout_records    

def Search_Candidates(source_readout_file, total_cand=200, existing_readout_file='cand_readouts.fasta',
                      readout_folder=_readout_folder, GC_percent=[0.4,0.6], max_consecutive=4, 
                      max_rep=6, C_percent=[0.2, 0.28], blast_hsp_thres=10.0, 
                      save_name='selected_candidates.fasta', verbose=True):
    """Function to search readout sequences in a given pool compared with existing readouts
    Inputs:
        source_readout_file: filename for readout sequence pool, string (should be .fasta)
        total_cand: number of candidates we hope to generate, int (default: 1000)
        existing_readout_file: filename for existing readouts, string (should be ,fasta)
        readout_folder: folder to store readout information, string (default: globally given)
        GC_percent: whether check gc content, list of two values or False ([0.4, 0.6])
        max_consecutive: maximum allowed consecutive bases, int (4)
        max_rep: maximum replicated sequence in a readout allowed, int (6)
        C_percent: percentage of base C, list of two values or False([0.22, 0.28])
        blast_hsp_thres: threshold for blast hsp, no hsp larger than this allowed, int (10)
        blast_ref: file basename for fasta file of existing readouts in readout_folder, used for blast
        verbose: say something!, bool (default: True)
    Outputs:
        _cand_readouts: list of Bio.SeqRecord.SeqRecord objects
    """
    ## check input files
    if not os.path.isfile(source_readout_file):
        source_readout_file = os.path.join(readout_folder, source_readout_file)
        if not os.path.isfile(source_readout_file):
            raise IOError(f"Wrong input source readout file:{source_readout_file}, not exist.")
    elif '.fasta' not in source_readout_file:
        raise IOError(f"Wrong input file type for {source_readout_file}")
    if not os.path.isfile(existing_readout_file):
        existing_readout_file = os.path.join(readout_folder, existing_readout_file)
        if not os.path.isfile(existing_readout_file):
            raise IOError(f"Wrong input source readout file:{existing_readout_file}, not exist.")
    elif '.fasta' not in existing_readout_file:
        raise IOError(f"Wrong input file type for {existing_readout_file}")
    # load candidate sequences and filter
    
    # start looping
    if verbose:
        print(f"- Start selecting readout candidates from {source_readout_file},\n\tfiltering with {existing_readout_file} ")
    _cand_records = []
    _ct = 0
    with open(source_readout_file, "rU") as _handle:
        for _record in SeqIO.parse(_handle, "fasta"):
            if len(_cand_records) >= total_cand:
                if verbose:
                    print(f"-- {total_cand} new candidates acquired, stop iteration.")
                break
            
            if verbose:
                print (f"--- processing: {_record.seq}")
                for i in range(32):
                    _new_seq = Extend_Readout(_record.seq)
                    _keep = Filter_Readout(_new_seq,GC_percent=GC_percent, 
                                            max_consecutive=max_consecutive,
                                            max_rep=max_rep, C_percent=C_percent,
                                            blast_hsp_thres=blast_hsp_thres,
                                            readout_folder=readout_folder,
                                            blast_ref=os.path.basename(existing_readout_file),
                                            verbose=False)
                    if _keep:
                        _kept_record = SeqRecord(_new_seq, id='cand_'+str(_ct+1), description='30mer_candidate')
                        _cand_records.append(_kept_record)
                        if verbose:
                            print (f"--- candidate:{_ct} {_new_seq} saved")
                        # Save to candidate records
                        with open(existing_readout_file, "a") as _output_handle:
                            SeqIO.write(_kept_record, _output_handle, "fasta")
                        _ct += 1
                        break
            else:
                break
    # after selection, save selected_candidates
    _save_filename = os.path.join(readout_folder, save_name)
    with open(_save_filename, 'w') as _output_handle:
        if verbose:
            print(f"-- saving candidate readouts into file: {_save_filename}")
        SeqIO.write(_cand_records, _output_handle, "fasta")
    
    return _cand_records


def filter_readouts_by_blast(blast_record, hard_thres=17, soft_thres=14, soft_count=100, verbose=False):
    '''filter for genome blast record
    Input:
        blast_record: xml format of blast result, generated by NCBIWWW.qblast or NCBIblastnCommandLine
        hard_threshold: no hsp_score match larger than this is allowed, int
        soft_threshold: hsp_score larger than this should be counted, int
        soft_count: number of hits larger than soft_threshold shouldnot pass this threshold, int
        verbose: it says a lot of things!
    Output:
        keep the record or not, bool'''
    if verbose:
        print (blast_record.query_id, len(blast_record.alignments))
    # extract information
    hsp_scores = []
    hsp_aligns = []
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            hsp_scores.append(hsp.score) 
            hsp_aligns.append(hsp.align_length)
    # hard threshold
    keep_hard = [hsp_score>hard_thres for hsp_score in hsp_scores]
    if verbose: 
        print( "hard count:", sum(keep_hard))
    if sum(keep_hard) > 0:
        if verbose:
            print ("Filtered out by hard threshold.")
        return False
    # soft threshold 
    keep_soft = [hsp_align>soft_thres for hsp_align in hsp_aligns]
    if verbose: 
        print ("soft count:", sum(keep_soft))
    if sum(keep_soft) >= soft_count:
        if verbose:
            print ("Filtered out by soft threshold count!")
        return False
    return True


def Filter_Readouts_by_Genome(cand_readout_file='selected_candidates.fasta', 
                              genome_db='hg38',
                              readout_folder=_readout_folder, genome_folder=_genome_folder,
                              word_size=10, evalue=1000, save_postfix='genome', 
                              verbose=True):
    """Filter readout candiates by blasting against genome
    Inputs:
    Outputs:
    """
    if not os.path.isfile(cand_readout_file):
        cand_readout_file = os.path.join(readout_folder, cand_readout_file)
        if not os.path.isfile(cand_readout_file):
            raise IOError(f"Wrong input candidate readout file:{cand_readout_file}, not exist.")
    elif '.fasta' not in cand_readout_file:
        raise IOError(f"Wrong input file type for {cand_readout_file}")
    # blast!
    blast_outfile = cand_readout_file.replace('.fasta', f'_{genome_db}.xml')
    output = NcbiblastnCommandline(query=cand_readout_file,
                                    num_threads=12,
                                    db=os.path.join(genome_folder, genome_db),
                                    evalue=500,
                                    word_size=10,
                                    out=blast_outfile,
                                    outfmt=5)()[0]
    # decide which to keep
    genomeblast_keeps = []                
    blast_records = NCBIXML.parse(open(os.path.join(readout_folder, 'selected_candidates_hg38.xml'), 'r'))
    for blast_record in blast_records:
        if verbose:
            print(blast_record.query_id, len(blast_record.alignments))
        keep = filter_readouts_by_blast(blast_record, verbose=verbose)
        genomeblast_keeps.append(keep)

    # save all 
    with open(cand_readout_file, "r") as handle:
        record_keeps = []
        for _i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if genomeblast_keeps[_i]:
                record_keeps.append(record)
    save_filename = cand_readout_file.replace('.fasta', f'_{save_postfix}.fasta')
    with open(save_filename, "w") as output_handle:
        SeqIO.write(record_keeps, output_handle, "fasta")
    if verbose:
        print(f"-- number of readout candidates kept: {len(record_keeps)}")
    
    return record_keeps

def Filter_Readouts_by_RNAfold(cand_readout_file='selected_candidates_genome.fasta',
                               rnafold_exe=r'E:\Shared_Apps\ViennaRNA\RNAfold',
                               energy_th =-6.0, readout_folder=_readout_folder, 
                               make_plot=False, verbose=True):
    """Filter Readouts by energy of secondary structure generated by RNAfold
    Inputs:
    Outputs:
    """
    if not os.path.isfile(cand_readout_file):
        cand_readout_file = os.path.join(readout_folder, cand_readout_file)
        if not os.path.isfile(cand_readout_file):
            raise IOError(f"Wrong input candidate readout file:{cand_readout_file}, not exist.")
    elif '.fasta' not in cand_readout_file:
        raise IOError(f"Wrong input file type for {cand_readout_file}")
    # run RNAfold
    _rnafold_output = cand_readout_file.replace('.fasta', '_structreport.txt')
    os.system(f"{rnafold_exe} < {cand_readout_file} > {_rnafold_output}")
    # load RNAfold result and read it
    import re
    structure_dics = []
    energy_list = []
    with open(_rnafold_output, 'r') as handle:
        structure_reports = handle.read()
        structure_reports = structure_reports.split('>')[1:]
        for structure_report in structure_reports:
            lines = structure_report.split('\n')[:-1]
            barcode_id = lines[0].split(' ')[0]
            barcode_description = lines[0].split(' ')[1]
            re_result = re.match('(\S+)\s\(\s*?([0-9\-\+\.]+)\)', lines[2])
            barcode_energy = float(re_result.group(2))
            structure_dic = {'id':barcode_id, 'description':barcode_description, 'energy':barcode_energy}
            structure_dics.append(structure_dic)
            energy_list.append(barcode_energy)
    # whether keep each record:
    structure_keeps = np.array(energy_list) > energy_th
    # extract kept records
    kept_records = []
    with open(os.path.join(readout_folder, 'selected_candidates_genome.fasta'), "r") as handle:
        for _i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if structure_keeps[_i]:
                kept_records.append(record)
    # save selected records
    _save_filename =   cand_readout_file.replace('.fasta', '_structure.fasta')
    with open(_save_filename, "w") as output_handle:
        SeqIO.write(kept_records, output_handle, "fasta")
    
    if make_plot:
        f1 = plt.figure()
        plt.hist(energy_list)
        plt.show()

    return kept_records


def Save_Readouts(cand_readout_file='selected_candidates_genome_structure.fasta',
                  existing_readout_file='NDBs.fasta', readout_folder=_readout_folder,
                  verbose=True):
    """Rename candidate readouts along with existing readouts and save
    Inputs:
    Output:"""
    # check inputs
    if not os.path.isfile(cand_readout_file):
        cand_readout_file = os.path.join(readout_folder, cand_readout_file)
        if not os.path.isfile(cand_readout_file):
            raise IOError(f"Wrong input candidate readout file:{cand_readout_file}, not exist.")
    elif '.fasta' not in cand_readout_file:
        raise IOError(f"Wrong input file type for {cand_readout_file}")
    if not os.path.isfile(existing_readout_file):
        existing_readout_file = os.path.join(readout_folder, existing_readout_file)
        if not os.path.isfile(existing_readout_file):
            raise IOError(f"Wrong input existing readout file:{existing_readout_file}, not exist.")
    elif '.fasta' not in existing_readout_file:
        raise IOError(f"Wrong input file type for {existing_readout_file}")
    
    # load existing readouts
    with open(existing_readout_file, 'r') as handle:
        existing_readouts = []
        existing_id_num = 0
        for _i, record in enumerate(SeqIO.parse(handle, "fasta")):
            existing_readouts.append(record)    
            _rec_id = int(record.id.split('_')[1])
            if _rec_id > existing_id_num:
                existing_id_num = _rec_id
    # load candidate readouts
    with open(cand_readout_file, 'r') as handle:
        cand_readouts = []
        for _i, record in enumerate(SeqIO.parse(handle, "fasta")):
            cand_readouts.append(record)    
    # modify records
    new_records = [SeqRecord(_record.seq, id='NDB_'+str(existing_id_num+_i+1), description="", name="")
                   for _i, _record in enumerate(cand_readouts)]
    if verbose:
        print(f"-- saving {len(new_records)} new readouts ")
    
    _save_filename = existing_readout_file.replace('.fasta', '_new.fasta')
    
    with open(_save_filename, "w") as output_handle:
        SeqIO.write(new_records, output_handle, "fasta")
        
    return new_records


#-------------------------------------------------------------------------------
## FISH library related