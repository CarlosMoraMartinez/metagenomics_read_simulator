from typing import List, Set, Dict, Tuple
import argparse
import os
import re 
import json

import numpy as np
import pandas as pd
import ftputil

# Example command:
# python simulate_metagenomes.py -o test1 -p ../data/k2standard_20240112_library_report.tsv -d test_genomes1 -i ../data/remove_tanda2_raw_counts.tsv -s dirichlet_prop -u 5 -m mean_abundance -a 10 -b 20 -x 0.1
# python simulate_metagenomes.py -o testSingle -p ../data/k2standard_20240112_library_report.tsv -d test_genomes1 -i ../data/remove_tanda2_raw_counts.tsv -s single_species -u 5 -m mean_abundance -a 10 -b 2 -x 0 -k 1 -t 10000
RES_SUBDIRS: List[str] = {'tables':'tables', 'sampled_genomes':'sampled_genomes', 'final_samples':'final_samples'}
MERGED_GENOMES_PATH: str = 'merged_genomes'
BASE_URL: str = "ftp.ncbi.nlm.nih.gov"
DEFAULT_N_READS: int = 1000000
DEFAULT_READ_LENGTH: int = 150
DEFAULT_FRAGMENT_LENGTH: int = 500
DEFAULT_FRAGMENT_SD: int = 50
DEFAULT_BASE_ERROR_RATE: float = 0
DEFAULT_MUTATION_RATE: float = 0.001
DEFAULT_INDEL_FRACTION: float = 0.15
DEFAULT_PROB_INDEL_EXT: float = 0.3
DEFAULT_SEED: int = 123

NON_CHROMOSOMAL_MERGE: int = 2
CHROMOSOMAL_NAMES = ['bacteria', 'archaea']
OTHER_ALLOWED = ['plasmid', 'viral', 'UniVec_Core']

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Logger:
    def __init__(self, logname: str = None) -> None:
        self.logname: str = logname
    def set_file(self, logname: str) -> None:
        self.logname = logname
    def log2file(self, msg: str):
        if self.logname is not None:
            with open(self.logname, "a") as fo:
                fo.write(str(msg) + '\n')
    def log(self, msg: str, col: bcolors = bcolors.BOLD) -> None:
        print(col + str(msg) + bcolors.ENDC)
        self.log2file(msg)

logger = Logger()

def remove_special_characters(name: str):
    return re.sub(r'[^a-zA-Z0-9\_\. ]', '', name)

def create_directories(dirlist: List[str]) -> None:
    logger.log("Creating directories", bcolors.BOLD)
    for d in dirlist:
        if not os.path.isdir(d):
            try:
                os.mkdir(d)
                logger.log(f"----{d} created", bcolors.OKGREEN)
            except Exception as e:
                logger.log(e, bcolors.FAIL)
                logger.log(f"----Unable to create directory: {d}", bcolors.FAIL)
        else:
            logger.log(f"----{d} already exists", bcolors.WARNING)

def run_command(cmd: str):
    logger.log(f"----Running command:\n{cmd}", bcolors.HEADER)
    try:
        exitcode: int = os.system(cmd)
        if exitcode == 0:
            logger.log(f"---Success: {cmd}", bcolors.OKGREEN)
        else:
            logger.log(f"---FAIL {exitcode}: {cmd}", bcolors.FAIL)
    except Exception as e:
        logger.log(e, bcolors.FAIL)
        logger.log(f"---FAIL: {cmd}", bcolors.FAIL)

def getFTPRoutesFromKraken2Report(ftp_paths: str) -> List[Dict[str, str]]:
    """Parses the Kraken2 database report and returns a list of dictionaries.

    Args:
        ftp_paths (str): Local path of the Kraken2 database report.

    Returns:
        List[Dict[str, str]]: List of dictionaries. Each entry corresponds to an entry in the original Kraken2 database report. 
                                Each dictionary contains three fields: 'kingdom', 'species' and 'route'. 
                                'Kingdom' is used to discriminate between plasmid and chromosomal fasta files.
                                'species' is used to match against input OTU table, to select interest taxa.
                                'route' is the path to the genome in the NCBI FTP. 
    """
    logger.log(f"Reading FTP routes from Kraken2 database report", bcolors.BOLD)
    res : List[Dict[str, str]] = []
    with open(ftp_paths, 'r') as ftpdirs:
        for l in ftpdirs:
            l = l.strip().split('\t')
            if l[0].startswith('#'):
                continue
            assert l[1].startswith(">"), f"----Error: Species does not start with '>': {l[1]}"
            assert l[2].startswith("ftp"), f"----Error: Route does not start with 'ftp': {l[2]}"
            l[1] = remove_special_characters(l[1])
            res.append(dict([(i, j) for i, j in zip(["kingdom", "species", "route"], l)])) 
    logger.log(f"----Success", bcolors.OKGREEN)
    return res

def downloadGenomes(ftproutes: List[Dict[str, str]], 
                    genomes_path: str, 
                    non_chromosomal: float,
                    rewrite_genomes: bool = True, 
                    url: str = BASE_URL) -> List[Dict[str, str]]:
    """Downloads a list of genomes from the NCBI FTP to local directory.

    Args:
        ftproutes (List[Dict[str, str]]): List of dictionaries containing FTP routes of genomes.
        genomes_path (str): Local path to save genomes.
        non_chromosomal (float): If set to 0, plasmid/viral DNA will not be downloaded.
        rewrite_genomes (bool, optional): If True, genomes will be downloaded regardless of whether the file already exists. Defaults to True.
        url (str, optional): Url of the NCBI FTP. Defaults to BASE_URL.

    Returns:
        List[Dict[str, str]]: _description_
    """
    logger.log(f"Downloading genomes from NCBI FTP to {genomes_path}", bcolors.BOLD)
    if not os.path.isdir(genomes_path):
        os.mkdir(genomes_path)
    ftp_host = ftputil.FTPHost(url, user='anonymous', passwd='@anonymous')

    for genome in ftproutes:
        fname: str = os.path.basename(genome['route'])
        remote_path: str = genome['route'].replace(f"ftp://{url}", "").replace("//", "/")
        new_path: str = os.path.join(os.path.abspath(genomes_path), fname)
        genome["local_path"] = new_path
        logger.log(f"Downloading: {remote_path}\n--to: {new_path}", bcolors.HEADER)
        if genome['kingdom'] in OTHER_ALLOWED and non_chromosomal == 0:
            logger.log("----Non-chromosomal, skipping.", bcolors.WARNING)
            genome["downloaded"] = "discard"
            continue            
        elif os.path.isfile(new_path) and not rewrite_genomes:
            logger.log("----already downloaded, skipping.", bcolors.OKBLUE)
            genome["downloaded"] = "old"
            continue
        elif os.path.isfile(new_path):
            logger.log("----WARNING: already downloaded, overwritting.", bcolors.WARNING)
        try:
            ftp_host.download(remote_path, new_path)
            genome["downloaded"] = "new"
            logger.log("----download finished.", bcolors.OKGREEN)
        except Exception as e:
            logger.log(e, bcolors.FAIL)
            genome["downloaded"] = "failed"
            logger.log("----download FAILED.", bcolors.FAIL)
    return ftproutes

def downloadGenomes_all(speciestab: pd.DataFrame, genomes_path: str, 
                        non_chromosomal: float, rewrite_genomes: bool = True, 
                        url: str = BASE_URL) -> List[Dict[str, str]]:
    """Downloads all genomes for all species in the input tab.

    Args:
        speciestab (pd.DataFrame): Input table with taxa in rows and columns with FTP routes.
        genomes_path (str): Path to store the genomes permanently. Will create a separate subdirectory for each species.
        non_chromosomal (float): If it is set to 0, plasmidid/Viral genomes will not be downloaded.
        rewrite_genomes (bool, optional): If True, genomes will be downloaded regardless of whether the file already exists. Defaults to True.
        url (str, optional): Url of the NCBI FTP. Defaults to BASE_URL.

    Returns:
        List[Dict[str, str]]: _description_
    """
    logger.log(f"Downloading all genomes to: {genomes_path}", bcolors.BOLD)
    for index, row in speciestab.iterrows():
        spname: str = re.sub(r'[^a-zA-Z0-9\_]', '', row.taxon)
        row.local_path = downloadGenomes(ftproutes=row.local_path, 
                                         genomes_path=os.path.join(genomes_path, spname), 
                                         non_chromosomal=non_chromosomal, 
                                         rewrite_genomes=rewrite_genomes, 
                                         url=url)
    logger.log("----All downloads finished.", bcolors.OKGREEN)
    return speciestab

def find_genomes(taxon: str, ftproutes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Matches species name from input OTU table with species names from the Kraken2 report.
    May return duplicated genomes if in the input database the same fasta is stored as 'bacteria' and 'plasmid', for instance. Deduplicate later.

    Args:
        taxon (str): taxon name from input OTU table to query against Kraken2 database report.
        ftproutes (List[Dict[str, str]]): Parsed kraken2 database report as returned by 'getFTPRoutesFromKraken2Report'.

    Returns:
        List[Dict[str, str]]: Elements in ftproutes matching input taxon.
    """
    return [genome for genome in ftproutes if taxon in genome['species']]

def readSpecies2Sim(input_table: str, ftproutes: List[Dict[str, str]], outdir: str = './', non_chromosomal: float = 0) -> pd.DataFrame:
    """Reads OTU table/table with target species proportion and merges it with information from the Kraken2 database report, 
    including NCBI FTP routes for each species reference genomes. 

    Args:
        input_table (str): input table. The first column must contain a list of taxa, and the rest of columns can be abundances or proportions of different samples.
        ftproutes (List[Dict[str, str]]): Parsed kraken2 database report as returned by 'getFTPRoutesFromKraken2Report'.
        outdir (str, optional): Output directory. Defaults to './'.

    Returns:
        pd.DataFrame: Dataframe in the same format as input_table, with columns for the ftp routes for each taxon.
    """
    logger.log(f"Reading species table to simulate", bcolors.BOLD)

    species2sim: pd.DataFrame = pd.read_csv(input_table, sep='\t')    
    species2sim.columns = species2sim.columns.str.replace(species2sim.columns[0], "taxon")
    species2sim.taxon = [remove_special_characters(i) for i in species2sim.taxon]

    logger.log(f"----Matching with genomes", bcolors.WARNING)
    species2sim['local_path'] = species2sim.apply(lambda x: find_genomes(x.taxon.replace('_', ' '), ftproutes), axis=1)
    species2sim['num_genomes'] = species2sim.apply(lambda x: len(x.local_path), axis=1)
    species2sim['num_genomes_chrom'] = species2sim.apply(lambda x: len([i for i in x.local_path if i['kingdom'] in CHROMOSOMAL_NAMES]), axis=1)

    filter_by: str = 'num_genomes' if non_chromosomal in [NON_CHROMOSOMAL_MERGE, 1] else 'num_genomes_chrom'
    logger.log(f"----Filtering present genomes by {filter_by}", bcolors.WARNING)
 
    absent_species = species2sim.loc[species2sim[filter_by] == 0]
    present_species = species2sim.loc[species2sim[filter_by] > 0]
    absent_species.to_csv(os.path.join(outdir, 'absent_species.tsv'), sep='\t')
    present_species.to_csv(os.path.join(outdir, 'present_species.tsv'), sep='\t')

    tab2write = species2sim.copy()
    tab2write['local_path'] = tab2write.apply(lambda x: '|'.join([str(i) for i in x.local_path]), axis=1)
    tab2write.to_csv(os.path.join(outdir, 'all_species_with_ftp_path.tsv'), sep='\t')

    logger.log(f"----Matched {present_species.shape[0]} and didn't find {absent_species.shape[0]} out of {species2sim.shape[0]}", bcolors.WARNING)
    logger.log(f"----Success", bcolors.OKGREEN)
    return present_species

def calculateAbundances(species2sim: pd.DataFrame) -> pd.DataFrame:
    """Starting from an OTU table, it calculates 

    Args:
        species2sim (pd.DataFrame): input data. OTU table with abundances in different samples or single column with proportions. 
        It calculates proportions for each sample and also mean and median proportions, and prevalence. 

    Returns:
        pd.DataFrame: Input table with additional columns: mean_abundance, median_abundance and prevalence
    """
    if species2sim.shape[1] > 4:
        sampledata = species2sim.drop(['taxon', 'local_path', 'num_genomes'], axis=1).apply(lambda x: x/sum(x))
        samplenames = sampledata.columns.tolist()
        sampledata['mean_abundance'] = sampledata.apply(np.mean, axis=1)
        sampledata['median_abundance'] = sampledata[samplenames].apply(np.median, axis=1)
        sampledata['prevalence'] = sampledata[samplenames].apply(lambda x: sum(x>0)/x.shape[0], axis=1)
        sampledata = pd.concat([species2sim[['taxon', 'local_path', 'num_genomes']], sampledata], axis=1)
    else:
        sampledata = species2sim.copy()
        samplename = [i for i in sampledata.columns if i not in ['taxon', 'local_path', 'num_genomes']]
        sampledata['mean_abundance'] = sampledata[samplename]/np.sum(sampledata[samplename])
        sampledata['median_abundance'] = sampledata['mean_abundance']
        sampledata['prevalence'] = sampledata.apply(lambda x: 1 if x.mean_abundance > 0  else 0, axis=1)
    return sampledata

def simulate_dirichlet_samples(props: np.array, size: int, n_samples: int, params: str) -> pd.DataFrame:
    """Generates a series of multinomial distributions from a Dirichlet distribution parameterized with an input distribution and an 'alpha' value.

    Args:
        props (np.array): Input multinomial distribution, i.e., species proportions. If size 0, all input values for the Dirichlet are set to alpha.
        size (int): Used only if input proportions array is of length 0. Generates samples of this size, starting from a multinomial with equal probabilities.
        n_samples (int): Number of distributions to sample (i.e., number of random samples to take from the Dirichlet distribution).
        params (str): Alpha to parameterize the dirichlet distribution

    Returns:
        pd.DataFrame: _description_
    """
    if props.shape[0] == 0:
        alpha: List[float] = [float(params)]*size
    else:
        alpha = props*float(params)
    distribs: np.array = np.random.dirichlet(alpha, n_samples).transpose()
    distribs_df: pd.DataFrame = pd.DataFrame(distribs, columns=[f"D{i}_a{1}" for i in range(1, n_samples+1)])
    return distribs_df

def simulate_single_species(top_n_species: int, taxanames: List[str]) -> pd.DataFrame:
    """Generate one sample per input species, each with only one species.

    Args:
        top_n_species (int): Number of species.
        taxanames (List[str]): List of species names.

    Returns:
        pd.DataFrame: Data frame with species proportion in each sample. Should be all zeroes and ones in the diagonal. 
    """
    distribs: np.array = np.diag(np.ones(top_n_species)) 
    taxanames = [re.sub(r'[^a-zA-Z0-9\_]', '', s) for s in taxanames]
    distribs_df: pd.DataFrame = pd.DataFrame(distribs, columns=[f"D{i}_{t}" for i, t in enumerate(taxanames)])
    return distribs_df

def calculateSpeciesProportion(species2sim: pd.DataFrame, sym_mode: str, top_n_species: int, mode_select_species: str, 
                               dstr_args: str, dstr_reps: int, outdir: str = "./") -> pd.DataFrame:   
    """Calculates the species proportion in each target distribution. 
    Depending on 'sym_mode', returns one uniform/equal to input proportions, or many generated distributions.

    Args:
        species2sim (pd.DataFrame): OTU table or table specifying proportions for each taxa
        sym_mode (str): One of 'unif', 'prop', 'dirichlet_unif', 'dirichlet_prop' or 'single_species'. Different modes to generate sample distributions.
        top_n_species (int): Select the top 'n' taxa from the input table 'species2sim'.
        mode_select_species (str): One of 'mean_abundance', 'median_abundance', 'prevalence'. Sort samples basing on this variable in order to select the top 'n'.
        dstr_args (str): Additional arguments to sampling distribution. Alpha in dirichlet distribution. It is converted to float. 
        dstr_reps (int): Number of distributions to sample. Only used if 'dirichlet_unif' or 'dirichlet_prop' are used.
        outdir (str, optional): Output directory. Defaults to "./".

    Returns:
        pd.DataFrame: _description_
    """
    logger.log(f"Calculating simulated proportions", bcolors.BOLD)
    logger.log(f"----Mode: {sym_mode}", bcolors.OKCYAN)
    logger.log(f"----Top n species: {top_n_species}", bcolors.OKCYAN)
    logger.log(f"----Species Selection Mode: {mode_select_species}", bcolors.OKCYAN)
    logger.log(f"----Distribution arguments: {dstr_args}", bcolors.OKCYAN)
    logger.log(f"----Distribution samples: {dstr_args}", bcolors.OKCYAN)
    sampledata = calculateAbundances(species2sim)
    if top_n_species > sampledata.shape[0] or top_n_species < 1:
        top_n_species = sampledata.shape[0]
    if species2sim.shape[1] <= 4:
        mode_select_species = 'mean_abundance'
        print("Only one sample detected. Sorting taxa basing on proportion of reads.")
    sampledata = sampledata.sort_values(mode_select_species, ascending=False)
    sampledata.to_csv(os.path.join(outdir, 'sample_relative_abundances.tsv'), sep='\t')
    selected_taxa: pd.DataFrame = sampledata[['taxon', 'local_path', 'num_genomes', 'mean_abundance', 'median_abundance', 'prevalence']].head(top_n_species)
    selected_taxa['base_proportion'] = selected_taxa['mean_abundance']/np.sum(selected_taxa['mean_abundance'])    

    sim_distribs: pd.DataFrame = pd.DataFrame()
    if top_n_species == 1:
        selected_taxa['D0_u'] = 1
    elif sym_mode == 'unif':
        selected_taxa['D0_u'] = 1/top_n_species
    elif sym_mode == 'prop':
        selected_taxa['D0_p'] = selected_taxa['base_proportion']
    elif sym_mode == 'dirichlet_unif':
        selected_taxa['D0_p'] = selected_taxa['base_proportion']
        sim_distribs = simulate_dirichlet_samples(np.array([]), top_n_species, dstr_reps, dstr_args)
        selected_taxa = pd.concat([selected_taxa.reset_index(), sim_distribs], axis=1)
    elif sym_mode == 'dirichlet_prop':
        selected_taxa['D0_p'] = selected_taxa['base_proportion']
        sim_distribs = simulate_dirichlet_samples(selected_taxa['base_proportion'].to_numpy(), top_n_species, dstr_reps, dstr_args)
        selected_taxa = pd.concat([selected_taxa.reset_index(), sim_distribs], axis=1)
    elif sym_mode == 'single_species':
        sim_distribs = simulate_single_species(top_n_species, selected_taxa.taxon.tolist())
        selected_taxa = pd.concat([selected_taxa.reset_index(), sim_distribs], axis=1)
    else:
        print(f"Unknown distribution. Using proportions.")
        selected_taxa['D1'] = selected_taxa['base_proportion']
    selected_taxa.to_csv(os.path.join(outdir, f"simulated_proportions_top{top_n_species}_{sym_mode}_dstr{dstr_args}"))

    logger.log(f"----Success", bcolors.OKGREEN)
    return selected_taxa

def getFiles2MergeSingleSpecies(row: pd.Series, non_chromosomal: float) ->  List[Tuple[str, Set[str]]]:
    """Groups all the fasta files from a single species in order to concatenate them. 
    If non_chromosomal == 2, it groups all the files together. 
    If non_chromosomal == 0, only groups the cromosomal fasta files. 
    Otherwise, it groups chromosomal and plasmid/viral files separately

    Args:
        row (pd.Series): A row in the species_proportion dataframe. Needs to have the 'local_path' field with a list of dictionaries with keys ['local_path', 'kingdom'].
        non_chromosomal (float): Controls the grouping.

    Returns:
        List[Tuple[str, Set[str]]]: A list with three tuples of size 2. Each tuple has 
            1) the name of the group (merged_all, merged_chrom or merged_plasmid) and
            2) a set of unique fasta files to concatenate
    """
    logger.log(f"----Getting files of taxon {row.taxon} to merge with non chromosomal set to: {non_chromosomal}", bcolors.OKCYAN)
    files_all: Set[str] = set()
    files_chr: Set[str] = set()
    files_plas: Set[str] = set()
    if non_chromosomal == NON_CHROMOSOMAL_MERGE:
        files_all = set(map(lambda d: d['local_path'], row.local_path))
        files_chr = set()
        files_plas = set()
    elif non_chromosomal == 0:
        files_all = set()
        files_chr = set([i['local_path'] for i in row.local_path if i['kingdom'] in CHROMOSOMAL_NAMES])
        files_plas = set()
    else:
        files_all = set()
        files_chr = set([i['local_path'] for i in row.local_path if i['kingdom'] in CHROMOSOMAL_NAMES])
        files_plas = set([i['local_path'] for i in row.local_path if i['kingdom'] in OTHER_ALLOWED])
    files2generate:  List[Tuple[str, Set[str]]] = [i for i in zip(('merged_all', 'merged_chrom', 'merged_plasmid'), (files_all, files_chr, files_plas))]
    #print(files2generate)
    return files2generate

def mergeGenomes(species_proportion: pd.DataFrame, rewrite_genomes: bool, 
                 outdir: str, non_chromosomal: float) -> pd.DataFrame:
    """Generates concatenated genome files for species with more than one reference genome. 
    Depending on the 'non_chromosomal' argument generates a single file with all the sequences from a given species (non_chromosomal=2),
    a file with only the chromosomal fasta (non_chromosomal=0) or a file with the chromosomal and a file with the plasmid/viral sequences 
    (non_chromosomal between 0 and 1). Modifies the species_proportion dataframe with three columns that include the paths for the three 
    possible files generated. 


    Args:
        species_proportion (pd.DataFrame): Dataframe from previous step with species name, local paths to genomes, etc. 
        rewrite_genomes (bool): If False, generates merged genome files regardless of whether they exist or not. 
        outdir (str): Directory to write merged genome files.
        non_chromosomal (float): Controls the grouping.of the output files. -1 to merge all fasta, between 0 and 1 to separate chromosomal and plasmid/viral. 

    Returns:
        pd.DataFrame: species_proportion dataframe with the paths of the files with merged genomes.
    """

    logger.log(f"Merging genomes for species with several fasta. Non-chromosomal DNA is set to {non_chromosomal}", bcolors.BOLD)
    species_proportion['merged_all'] = ""
    species_proportion['merged_chrom'] = ""
    species_proportion['merged_plasmid'] = ""
    for i, row in species_proportion.iterrows():
        files_2_generate: List[Tuple[str, Set[str]]]  = getFiles2MergeSingleSpecies(row, non_chromosomal) 
        fname: str
        localfiles: Set[str]
        for fname, localfiles in files_2_generate:
            if not localfiles:
                logger.log(f"----{fname} is not to be created or does not exist. Skipping.", bcolors.WARNING)
                continue
            if len(localfiles) == 1:
                logger.log(f"----{fname} has only 1 filename: {list(localfiles)[0]}. Skipping merge.", bcolors.OKBLUE)
                species_proportion.loc[i, fname] = list(localfiles)[0]
                continue
            merged_fname: str = os.path.join(outdir, f"{row.taxon}_{fname}.fna.gz")
            species_proportion.loc[i, fname] = merged_fname
            if os.path.isfile(merged_fname) and not rewrite_genomes:
                logger.log(f"----{merged_fname} already exists, skipping merge.", bcolors.OKBLUE)
                continue
            elif os.path.isfile(merged_fname):
                logger.log(f"----{merged_fname} already exists, overwritting.", bcolors.WARNING)
            cmd: str = f"zcat {' '.join(localfiles)} | pigz -c > {merged_fname}"
            run_command(cmd)
    logger.log(f"----Success", bcolors.OKGREEN)
    return species_proportion

def updateSeed(seed:int = DEFAULT_SEED):
    """Generates a random number which will be used as seed for wgsim

    Args:
        seed (int, optional): _description_. Defaults to DEFAULT_SEED.

    Returns:
        _type_: _description_
    """
    return int(np.random.uniform(low=1, high=10000))

def wgsim_get_sample(input_fasta: str, output_fastq: str,
                     num_reads: int, read_length_r1: int, read_length_r2: int, 
                     fragment_length: int, fragment_sdesv: int,
                     base_error_rate: float, mutation_rate: float, 
                     indel_fraction: float, prob_indel_ext: float,
                     rewrite_sim_fastq: bool,
                     seed: int) -> Tuple[str, str]:
    """Creates and runs wgsim command to simulate a given number of reads from an input fasta.

    Args:
        input_fasta (str): Optionally compressed fasta file to simulate reads from. 
        output_fastq (str): Name of the output fastq files.
        num_reads (int): Number of reads to generate. Read wgsim docs.
        read_length_r1 (int): Length of read 1 in each read pair. Read wgsim docs.
        read_length_r2 (int): Length of read 2 in each read pair. Read wgsim docs.
        fragment_length (int): Read wgsim docs.
        fragment_sdesv (int): Read wgsim docs.
        base_error_rate (float): Read wgsim docs.
        mutation_rate (float): Read wgsim docs.
        indel_fraction (float): Read wgsim docs
        prob_indel_ext (float): Read wgsim docs.
        rewrite_sim_fastq (bool): Rewrite output file if it exists.
        seed (int): Random seed for wgsim.

    Returns:
        Tuple[str, str]: Names of the two compressed fastq files generated. 
    """
    r1: str = f"{output_fastq}_R1.fastq"
    r2: str = f"{output_fastq}_R2.fastq"
    if os.path.isfile(r1 + '.gz') or os.path.isfile(r2 + '.gz'):
        if rewrite_sim_fastq:
            cmd_rm: str = f"rm {r1}.gz {r2}.gz"
            logger.log(f"----Species files already exist. Removing", bcolors.WARNING)
            run_command(cmd_rm)
        else:
            logger.log(f"----Species files already exist. Skipping WGSIM", bcolors.WARNING)
            return(r1 + '.gz', r2 + '.gz')
    
    logfile: str = f"{output_fastq}_WGSIM.log"
    errfile: str = f"{output_fastq}_WGSIM.err"
    cmd: str = (f"wgsim -e {base_error_rate}"
                f" -d {fragment_length}"
                f" -s {fragment_sdesv}"
                f" -N {num_reads}"
                f" -1 {read_length_r1}"
                f" -2 {read_length_r2}"
                f" -r {mutation_rate}"
                f" -R {indel_fraction}"
                f" -X {prob_indel_ext}"
                f" -S {seed}"
                f" {input_fasta} {r1} {r2} >{logfile} 2>{errfile};" 
                f" pigz {r1}; pigz {r2}")
    run_command(cmd)
    return(r1 + '.gz', r2 + '.gz')

def simulate_reads_by_species(species_proportion: pd.DataFrame, outdir: str, 
                                      total_reads: int, read_length_r1: int, read_length_r2: int, 
                                      fragment_length: int, fragment_sdesv: int,
                                      base_error_rate: float, mutation_rate: float, 
                                      indel_fraction: float, prob_indel_ext: float,
                                      seed: int, n_samples: int,
                                      non_chromosomal: float, rewrite_sim_fastq: bool,
                                      n_threads: int) -> pd.DataFrame:
    """ Calls wgsim for each sample distribution, for each repetition, for each species. 
    Depending on 'non_chromosomal' value, calls wgsim separately for chromosomal and plasmid fasta.

    Args:
        species_proportion (pd.DataFrame): Data Frame with taxa in rows and sample distributions, as well as paths to each species merged fasta, in columns.
        outdir (str): Output directory. 
        total_reads (int): Total number of reads per sample.
        read_length_r1 (int): Length of read 1 in each read pair. Read wgsim docs.
        read_length_r2 (int): Length of read 2 in each read pair. Read wgsim docs.
        fragment_length (int): Read wgsim docs.
        fragment_sdesv (int): Read wgsim docs.
        base_error_rate (float): Read wgsim docs.
        mutation_rate (float): Read wgsim docs.
        indel_fraction (float): Read wgsim docs.
        prob_indel_ext (float): Read wgsim docs.
        seed (int): Starting random seed for wgsim. It will change automatically according to some function of more than one replicate is generated. 
        n_samples (int): Number of replicates per sample distribution.
        non_chromosomal (float): If it is between 0 and 1, indicates the proportion of plasmidic reads in each species. If it is 2, reads are generated from a fasta with merged chromosomal and plasmidic sequences.
        rewrite_sim_fastq (bool): Rewrite output files if they exist.
        n_threads (int): Not used.

    Returns:
        pd.DataFrame: Data frame with data for simulated samples, including paths to the separate fastq for each species.
    """
    logger.log(f"Simulating reads for each sample, separately by species.", bcolors.BOLD)
    sample_regex: re.Pattern = re.compile('^D[0-9]+_')
    sample_names: List[str] = [i for i in species_proportion.columns if sample_regex.match(i)]
    sample: str
    samples_generated: pd.DataFrame = pd.DataFrame(columns = ['sample_distr', 'sample_rep', 'full_sample_name', 'seed','sample_dir', 'species_r1', 'species_r2'])
    sim_files: Tuple[str, str]
    np.random.seed(seed)
    sample_seed = updateSeed(seed)
    for sample in sample_names:
        #sample_seed = seed
        rep: int
        for rep in range(1, n_samples + 1):
            full_sample_name: str = f"{sample}_rep{rep}_seed{sample_seed}"
            sample_dir: str = os.path.join(outdir, full_sample_name)
            create_directories([sample_dir])
            samples_generated = samples_generated._append({'sample_distr': sample, 
                                      'sample_rep': rep,
                                      'full_sample_name': full_sample_name,
                                      'seed': sample_seed,
                                      'sample_dir': sample_dir,
                                      'species_r1': [],
                                      'species_r2': []}, ignore_index=True)
            logger.log(f"Generating sample {sample}, rep {rep}", bcolors.BOLD)
            for i, row in species_proportion.iterrows():
                if row[sample] == 0:
                    logger.log(f"----Skipping sample {sample}, rep {rep}, species {row.taxon} because proportion is {row[sample]}", bcolors.WARNING)
                    continue
                if non_chromosomal == 2:
                    num_reads: int = int(round(total_reads*row[sample]))
                    fastq_name: str = os.path.join(sample_dir, f"{full_sample_name}_{row.taxon}_N{num_reads}_merged_all")
                    logger.log(f"----Generating {num_reads} ({round(row[sample],3)} of {total_reads}) chr+plasmid reads for sample {sample}, rep {rep}, species {row.taxon}", bcolors.OKCYAN)
                    sim_files = wgsim_get_sample(input_fasta=row.merged_all, 
                                     output_fastq=fastq_name,
                                     num_reads = num_reads, 
                                     read_length_r1 = read_length_r1,
                                     read_length_r2=read_length_r2, 
                                     fragment_length=fragment_length, 
                                     fragment_sdesv=fragment_sdesv,
                                     base_error_rate=base_error_rate, mutation_rate=mutation_rate, 
                                     indel_fraction=indel_fraction, 
                                     prob_indel_ext=prob_indel_ext,
                                     rewrite_sim_fastq=rewrite_sim_fastq,
                                     seed=sample_seed)
                    samples_generated.iloc[-1].species_r1.append(sim_files[0])
                    samples_generated.iloc[-1].species_r2.append(sim_files[1])
                elif non_chromosomal == 0 or not row.merged_plasmid:
                    num_reads: int = int(round(total_reads*row[sample]))
                    fastq_name: str = os.path.join(sample_dir, f"{full_sample_name}_{row.taxon}_N{num_reads}_merged_chrom")
                    logger.log(f"----Generating {num_reads} ({round(row[sample],3)} of {total_reads}) chromosomal only reads for sample {sample}, rep {rep}, species {row.taxon}", bcolors.OKCYAN)
                    sim_files = wgsim_get_sample(input_fasta=row.merged_chrom, 
                                     output_fastq=fastq_name,
                                     num_reads = num_reads, 
                                     read_length_r1 = read_length_r1,
                                     read_length_r2=read_length_r2, 
                                     fragment_length=fragment_length, 
                                     fragment_sdesv=fragment_sdesv,
                                     base_error_rate=base_error_rate, mutation_rate=mutation_rate, 
                                     indel_fraction=indel_fraction, 
                                     prob_indel_ext=prob_indel_ext,
                                     rewrite_sim_fastq=rewrite_sim_fastq,
                                     seed=sample_seed)
                    samples_generated.iloc[-1].species_r1.append(sim_files[0])
                    samples_generated.iloc[-1].species_r2.append(sim_files[1])
                else:
                    num_reads_plas: int = int(round(total_reads*row[sample]*non_chromosomal))
                    num_reads_chrom: int = int(round(total_reads*row[sample]*(1 - non_chromosomal)))
                    fastq_name_chrom: str = os.path.join(sample_dir, f"{full_sample_name}_{row.taxon}_N{num_reads_chrom}_merged_chrom")
                    logger.log(f"----Generating {num_reads_chrom} ({round(row[sample],3)}*{1-non_chromosomal} of {total_reads}) chromosomal only reads for sample {sample}, rep {rep}, species {row.taxon}", bcolors.OKCYAN)
                    sim_files = wgsim_get_sample(input_fasta=row.merged_chrom, 
                                     output_fastq=fastq_name_chrom,
                                     num_reads = num_reads_chrom, 
                                     read_length_r1 = read_length_r1,
                                     read_length_r2=read_length_r2, 
                                     fragment_length=fragment_length, 
                                     fragment_sdesv=fragment_sdesv,
                                     base_error_rate=base_error_rate, mutation_rate=mutation_rate, 
                                     indel_fraction=indel_fraction, 
                                     prob_indel_ext=prob_indel_ext,
                                     rewrite_sim_fastq=rewrite_sim_fastq,
                                     seed=sample_seed)
                    samples_generated.iloc[-1].species_r1.append(sim_files[0])
                    samples_generated.iloc[-1].species_r2.append(sim_files[1])

                    fastq_name_plas: str = os.path.join(sample_dir, f"{full_sample_name}_{row.taxon}_N{num_reads_plas}_merged_plas")
                    logger.log(f"----Generating {num_reads_plas} ({round(row[sample],3)}*{non_chromosomal} of {total_reads}) plasmid only reads for sample {sample}, rep {rep}, species {row.taxon}", bcolors.OKCYAN)
                    sim_files = wgsim_get_sample(input_fasta=row.merged_plasmid, 
                                     output_fastq=fastq_name_plas,
                                     num_reads = num_reads_plas, 
                                     read_length_r1 = read_length_r1,
                                     read_length_r2=read_length_r2, 
                                     fragment_length=fragment_length, 
                                     fragment_sdesv=fragment_sdesv,
                                     base_error_rate=base_error_rate, mutation_rate=mutation_rate, 
                                     indel_fraction=indel_fraction, 
                                     prob_indel_ext=prob_indel_ext,
                                     rewrite_sim_fastq=rewrite_sim_fastq,
                                     seed=sample_seed)
                    samples_generated.iloc[-1].species_r1.append(sim_files[0])
                    samples_generated.iloc[-1].species_r2.append(sim_files[1])

            sample_seed = updateSeed(sample_seed)
        
    logger.log(f"----Success", bcolors.OKGREEN)
    return samples_generated

def merge_simulated_samples(sim_samples: pd.DataFrame, outdir: str):
    """_summary_

    Args:
        sim_samples (pd.DataFrame): _description_
        outdir (str): _description_

    Returns:
        _type_: _description_
    """
    logger.log(f"Merging simulated reads into final fastq files.", bcolors.BOLD)
    sim_samples['final_r1'] = ''
    sim_samples['final_r2'] = ''
    for i, row in sim_samples.iterrows():
        logger.log(f"----Merging sample {row.full_sample_name} files.", bcolors.BOLD)
        out_r1: str = os.path.join(outdir, f"{row.full_sample_name}_R1.fastq.gz")
        out_r2: str = os.path.join(outdir, f"{row.full_sample_name}_R2.fastq.gz")
        cmd_r1: str = f"zcat {' '.join(row.species_r1)} | pigz -c > {out_r1}"
        cmd_r2: str = f"zcat {' '.join(row.species_r2)} | pigz -c > {out_r2}"
        run_command(cmd_r1)
        run_command(cmd_r2)
        sim_samples.iloc[i].final_r1 = out_r1
        sim_samples.iloc[i].final_r2 = out_r2
    logger.log(f"----Success", bcolors.OKGREEN)
    return sim_samples


# ----- command line parsing -----
parser: argparse.ArgumentParser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, 
                                                          description='A tool to simulate metagenome experiments with the same distribution as input experiments')
parser.add_argument('-o', '--outdir', type = str, help='Output directory', default="")
#Download genomes params
group1 = parser.add_argument_group('NCBI genome download options')
group1.add_argument('-p', '--ftp_paths', type = str, help='Kraken2 library report with three columns: kingdom, sequence name (including species name) and FTP route. If absent, will rely on genomes_path.', default="")
group1.add_argument('-d', '--genomes_path', type = str, help='Path to save downloaded genomes from NCBI. If rewrite is false, it will download the files regardless of whether they are present. If Kraken2 report is not supplied, it will assume that all the files are there already.', default="./")
group1.add_argument('-r', '--rewrite_genomes', help = 'Rewrite or not genome files if already downloaded', action=argparse.BooleanOptionalAction, default=False)

group2 = parser.add_argument_group('Genome selection and distribution options')
group2.add_argument('-i', '--input_table', type = str, help='Input table with taxa names and counts per sample or proportions', default="")
group2.add_argument('-s', '--sym_mode', choices=['unif','prop', 'dirichlet_unif', 'dirichlet_prop', 'single_species'], help='How to generate distribution of selected taxa', default="prop")
group2.add_argument('-u', '--top_n_species', type = int, help='Number of species to choose from input table.', default=DEFAULT_N_READS)
group2.add_argument('-m', '--mode_select_species', choices = ['median_abundance', 'mean_abundance', 'prevalence'], type = str, help='Select top_n_species basing on abundance or prevalence', default='mean_abundance')
group2.add_argument('-a', '--dstr_args', type = str, help='Extra arguments for sampling distribution. Behaviour changes depending on sym_mode. Currently, only \'alpha\' for the Dirichlet distribution sampling.', default='1')
group2.add_argument('-b', '--dstr_reps', type = int, help='If sampling distributions (normal, exp or dirichlet), number of distributions to sample.', default=1)

group3 = parser.add_argument_group('Read simulation options')
group3.add_argument('-t', '--total_reads', type = int, help='Total number of paired end reads to generate per sample.', default=DEFAULT_N_READS)
group3.add_argument('-1', '--read_length_r1', type = int, help='Read length for R1', default=DEFAULT_READ_LENGTH)
group3.add_argument('-2', '--read_length_r2', type = int, help='Read length for R2', default=DEFAULT_READ_LENGTH)
group3.add_argument('-F', '--fragment_length', type = int, help='Fragment length', default=DEFAULT_FRAGMENT_LENGTH)
group3.add_argument('-G', '--fragment_sdesv', type = int, help='Variation in sample length.', default=DEFAULT_FRAGMENT_SD)
group3.add_argument('-E', '--base_error_rate', type = int, help='Variation in sample length.', default=DEFAULT_BASE_ERROR_RATE)
group3.add_argument('-R', '--mutation_rate', type = int, help='Variation in sample length.', default=DEFAULT_MUTATION_RATE)
group3.add_argument('-T', '--indel_fraction', type = int, help='Variation in sample length.', default=DEFAULT_INDEL_FRACTION)
group3.add_argument('-X', '--prob_indel_ext', type = int, help='Variation in sample length.', default=DEFAULT_PROB_INDEL_EXT)
group3.add_argument('-S', '--seed', type = int, help='Variation in sample length.', default=DEFAULT_SEED)
group3.add_argument('-k', '--n_samples', type = int, help='Number of samples per taxa distribution.', default=1)
group3.add_argument('-x', '--non_chromosomal', type = float, help='Proportion of non-chromosomal (plasmid, viral, UniVec_Core) reads. If set to 2, proportion will be according to length of sequences in database.', default=0)
group1.add_argument('-w', '--rewrite_merge', help = 'Rewrite merged genome files for each species if already generated', action=argparse.BooleanOptionalAction, default=False)
group3.add_argument('-n', '--n_threads', type = int, help='Number of threads', default=1)
group1.add_argument('-y', '--rewrite_sim_fastq', help = 'Do not rewrite simulated fastq if already exist', action=argparse.BooleanOptionalAction, default=False)

def main():
    args = parser.parse_args()
    outdir: str = args.outdir
    
    ftp_paths: str = args.ftp_paths
    genomes_path: str = args.genomes_path

    rewrite_genomes: bool = args.rewrite_genomes

    input_table: str = args.input_table
    sym_mode: str = args.sym_mode
    top_n_species: int = args.top_n_species
    mode_select_species: str = args.mode_select_species
    dstr_args: str = args.dstr_args
    dstr_reps: int = args.dstr_reps

    total_reads: int = args.total_reads
    read_length_r1: int = args.read_length_r1
    read_length_r2: int = args.read_length_r2
    fragment_length: int = args.fragment_length
    fragment_sdesv: int = args.fragment_sdesv
    base_error_rate: float = args.base_error_rate
    mutation_rate: float = args.mutation_rate
    indel_fraction: float = args.indel_fraction
    prob_indel_ext: float = args.prob_indel_ext
    seed: int = args.seed
    n_samples: int = args.n_samples
    non_chromosomal: float = args.non_chromosomal
    rewrite_merge: bool = args.rewrite_merge
    n_threads: int = args.n_threads
    rewrite_sim_fastq: bool = args.rewrite_sim_fastq
    
    create_directories([genomes_path, outdir, os.path.join(genomes_path, MERGED_GENOMES_PATH)] + [os.path.join(outdir, s) for s in RES_SUBDIRS.values()])
    with open(os.path.join(outdir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    logger.set_file(os.path.join(outdir, "metagenomics_read_simulator.log"))

    #Read Kraken2 database report to get the NCBI FTP paths
    ftproutes: List[Dict[str, str]] = getFTPRoutesFromKraken2Report(ftp_paths)
    #print(ftproutes)

    #Read species to simulate and match with genomes in NCBI FTP
    species2sim: pd.DataFrame = readSpecies2Sim(input_table, 
                                                ftproutes, 
                                                os.path.join(outdir, RES_SUBDIRS['tables']),
                                                non_chromosomal=non_chromosomal)
    if species2sim.shape[0] == 0:
        logger.log("Error: no species found in genomes.", bcolors.FAIL)
        logger.log(str(species2sim.head(3)), bcolors.FAIL)
        exit(1)

    #Calculate simulated proportions
    species_proportion: pd.DataFrame = calculateSpeciesProportion(species2sim=species2sim, 
                                                                  sym_mode=sym_mode, 
                                                                  top_n_species=top_n_species, 
                                                                  mode_select_species=mode_select_species, 
                                                                  dstr_args=dstr_args, 
                                                                  dstr_reps=dstr_reps, 
                                                                  outdir=os.path.join(outdir, RES_SUBDIRS['tables']))

    # Download Genomes
    species_proportion = downloadGenomes_all(speciestab=species_proportion, 
                                             genomes_path=genomes_path, 
                                             non_chromosomal=non_chromosomal,
                                             rewrite_genomes=rewrite_genomes, 
                                             url=BASE_URL)
    species_proportion.to_csv(os.path.join(outdir, RES_SUBDIRS['tables'], "species_tab_with_local_genomes.tsv"), sep='\t')

    # Merge genomes if needed
    species_proportion = mergeGenomes(species_proportion=species_proportion, 
                                      rewrite_genomes=rewrite_merge, 
                                      outdir=os.path.join(genomes_path, MERGED_GENOMES_PATH), 
                                      non_chromosomal=non_chromosomal)
    species_proportion.to_csv(os.path.join(outdir, RES_SUBDIRS['tables'], "species_tab_with_merged_genomes.tsv"), sep='\t')

    # Calculate reads and simulate
    sim_sample_df: pd.DataFrame = simulate_reads_by_species(species_proportion=species_proportion, 
                                      outdir=os.path.join(outdir, RES_SUBDIRS['sampled_genomes']), 
                                      total_reads=total_reads, 
                                      read_length_r1=read_length_r1, 
                                      read_length_r2=read_length_r2, 
                                      fragment_length=fragment_length, 
                                      fragment_sdesv=fragment_sdesv,
                                      base_error_rate=base_error_rate, 
                                      mutation_rate=mutation_rate, 
                                      indel_fraction=indel_fraction, 
                                      prob_indel_ext=prob_indel_ext,
                                      seed=seed, n_samples=n_samples,
                                      non_chromosomal=non_chromosomal, 
                                      rewrite_sim_fastq = rewrite_sim_fastq,
                                      n_threads=n_threads)
    sim_sample_df.to_csv(os.path.join(outdir, RES_SUBDIRS['tables'], "species_tab_sim_byspecies.tsv"), sep='\t')
    print(sim_sample_df.head(5))
    
    # Merge samples
    sim_sample_df = merge_simulated_samples(sim_sample_df, outdir = os.path.join(outdir, RES_SUBDIRS['final_samples']))

    sim_sample_df.to_csv(os.path.join(outdir, RES_SUBDIRS['tables'], "species_tab_sim_final.tsv"), sep='\t')
  
if __name__ == "__main__":
    main()