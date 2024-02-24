from typing import List, Callable, Dict
import argparse
import glob
import os
from multiprocessing import Pool
import re 

import numpy as np
import pandas as pd
import ftputil

# Example command:
# python simulate_metagenomes.py -o test1 -p ../data/k2standard_20220607_inspect.txt -d test_genomes1 -i ../data/remove_tanda2_raw_counts.tsv -s dirichlet_prop -u 5 -m mean_abundance -a 10 -b 20

RES_SUBDIRS: List[str] = {'tables':'tables', 'merged_genomes':'merged_genomes', 'sampled_genomes':'sampled_genomes', 'final_samples':'final_samples'}
BASE_URL: str = "ftp.ncbi.nlm.nih.gov"
DEFAULT_N_READS: int = 1000000
DEFAULT_READ_LENGTH: int = 150
DEFAULT_FRAGMENT_LENGTH: int = 300
DEFAULT_FRAGMENT_SD: int = 50

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

def log(msg: str, col: bcolors = bcolors.BOLD):
    print(col + msg + bcolors.ENDC)

def create_directories(dirlist: List[str]) -> None:
    log(f"Creating directories", bcolors.BOLD)
    for d in dirlist:
        if not os.path.isdir(d):
            try:
                os.mkdir(d)
                log(f"----{d} created", bcolors.OKGREEN)
            except:
                log(f"----Unable to create directory: {d}", bcolors.FAIL)
        else:
            log(f"----{d} already exists", bcolors.WARNING)

def getFTPRoutesFromKraken2Report(ftp_paths: str) -> List[Dict[str, str]]:
    log(f"Reading FTP routes from Kraken2 database report", bcolors.BOLD)
    res : List[Dict[str, str]] = []
    with open(ftp_paths, 'r') as ftpdirs:
        for l in ftpdirs:
            l = l.strip().split('\t')
            if l[0].startswith('#'):
                continue
            assert l[1].startswith(">"), f"----Error: Species does not start with '>': {l[1]}"
            assert l[2].startswith("ftp"), f"----Error: Route does not start with 'ftp': {l[2]}"
            res.append(dict([(i, j) for i, j in zip(["kingdom", "species", "route"], l)])) 
    log(f"----Success", bcolors.OKGREEN)
    return res

def downloadGenomes(ftproutes: List[Dict[str, str]], genomes_path: str, rewrite_genomes: bool = True, url: str = BASE_URL) -> List[Dict[str, str]]:
    log(f"Downloading genomes from NCBI FTP to {genomes_path}", bcolors.BOLD)
    if not os.path.isdir(genomes_path):
        os.mkdir(genomes_path)
    ftp_host = ftputil.FTPHost(url, user='anonymous', passwd='@anonymous')

    for genome in ftproutes:
        fname: str = os.path.basename(genome['route'])
        remote_path: str = genome['route'].replace(f"ftp://{url}", "").replace("//", "/")
        new_path: str = os.path.join(os.path.abspath(genomes_path), fname)
        genome["local_path"] = new_path
        log(f"Downloading: {remote_path}\n--to: {new_path}", bcolors.HEADER)

        if os.path.isfile(new_path) and not rewrite_genomes:
            log("----already downloaded, skipping.", bcolors.OKBLUE)
            genome["downloaded"] = "old"
            continue
        elif os.path.isfile(new_path):
            log("----WARNING: already downloaded, overwritting.", bcolors.WARNING)
        try:
            ftp_host.download(remote_path, new_path)
            genome["downloaded"] = "new"
            log("----download finished.", bcolors.OKGREEN)
        except:
            genome["downloaded"] = "failed"
            log("----download FAILED.", bcolors.FAIL)
    return ftproutes

def downloadGenomes_all(speciestab: pd.DataFrame, genomes_path: str, rewrite_genomes: bool = True, url: str = BASE_URL, outdir: str = "./") -> List[Dict[str, str]]:
    log(f"Downloading all genomes to: {genomes_path}", bcolors.BOLD)
    for index, row in speciestab.iterrows():
        spname: str = re.sub(r'[^a-zA-Z0-9\_]', '', row.taxon)
        row.local_path = downloadGenomes(row.local_path, os.path.join(genomes_path, spname), rewrite_genomes, url)
    speciestab.to_csv(os.path.join(outdir, "species_tab_with_local_genomes.tsv"), sep='\t')
    log("----All downloads finished.", bcolors.OKGREEN)
    return speciestab

def find_genomes(taxon: str, ftproutes: List[Dict[str, str]]) -> List[str]:
    """May return duplicated genomes if in the input database the same fasta is stored as 'bacteria' and 'plasmid', for instance. Deduplicate later."""
    return [genome for genome in ftproutes if taxon in genome['species']]

def readSpecies2Sim(input_table: str, ftproutes: List[Dict[str, str]], outdir: str = './') -> pd.DataFrame:
    log(f"Reading species table to simulate", bcolors.BOLD)

    species2sim: pd.DataFrame = pd.read_csv(input_table, sep='\t')    
    species2sim.columns = species2sim.columns.str.replace(species2sim.columns[0], "taxon")

    log(f"----Matching with genomes", bcolors.WARNING)
    species2sim['local_path'] = species2sim.apply(lambda x: find_genomes(x.taxon.replace('_', ' '), ftproutes), axis=1)
    species2sim['num_genomes'] = species2sim.apply(lambda x: len(x.local_path), axis=1)

    absent_species = species2sim.loc[species2sim['num_genomes'] == 0]
    absent_species.to_csv(os.path.join(outdir, 'absent_species.tsv'), sep='\t')
    present_species = species2sim.loc[species2sim['num_genomes'] > 0]
    tab2write = species2sim.copy()
    tab2write['local_path'] = tab2write.apply(lambda x: '|'.join([str(i) for i in x.local_path]), axis=1)
    tab2write.to_csv(os.path.join(outdir, 'all_species_with_ftp_path.tsv'), sep='\t')

    log(f"----Success", bcolors.OKGREEN)
    return present_species

def calculateAbundances(species2sim: pd.DataFrame) -> pd.DataFrame:
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
    if props.shape[0] == 0:
        alpha: List[float] = [float(params)]*size
    else:
        alpha = props*float(params)
    distribs: np.array = np.random.dirichlet(alpha, n_samples).transpose()
    distribs_df: pd.DataFrame = pd.DataFrame(distribs, columns=[f"D{i}_a{1}" for i in range(1, n_samples+1)])
    return distribs_df

def simulate_single_species(top_n_species: int, taxanames: List[str]) -> pd.DataFrame:
    distribs: np.array = np.diag(np.ones(top_n_species)) 
    taxanames = [re.sub(r'[^a-zA-Z0-9\_]', '', s) for s in taxanames]
    distribs_df: pd.DataFrame = pd.DataFrame(distribs, columns=[f"D{i}_{t}" for i, t in enumerate(taxanames)])
    return distribs_df

def calculateSpeciesProportion(species2sim: pd.DataFrame, sym_mode: str, top_n_species: int, mode_select_species: str, 
                               dstr_args: str, dstr_reps: int, outdir: str = "./") -> pd.DataFrame:    
    log(f"Calculating simulated proportions", bcolors.BOLD)
    log(f"----Mode: {sym_mode}", bcolors.OKCYAN)
    log(f"----Top n species: {top_n_species}", bcolors.OKCYAN)
    log(f"----Species Selection Mode: {mode_select_species}", bcolors.OKCYAN)
    log(f"----Distribution arguments: {dstr_args}", bcolors.OKCYAN)
    log(f"----Distribution samples: {dstr_args}", bcolors.OKCYAN)
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
        selected_taxa['D0'] = 1
    elif sym_mode == 'unif':
        selected_taxa['D0'] = 1/top_n_species
    elif sym_mode == 'prop':
        selected_taxa['D0'] = selected_taxa['base_proportion']
    elif sym_mode == 'dirichlet_unif':
        selected_taxa['D0'] = selected_taxa['base_proportion']
        sim_distribs = simulate_dirichlet_samples(np.array([]), top_n_species, dstr_reps, dstr_args)
        selected_taxa = pd.concat([selected_taxa.reset_index(), sim_distribs], axis=1)
    elif sym_mode == 'dirichlet_prop':
        selected_taxa['D0'] = selected_taxa['base_proportion']
        sim_distribs = simulate_dirichlet_samples(selected_taxa['base_proportion'].to_numpy(), top_n_species, dstr_reps, dstr_args)
        selected_taxa = pd.concat([selected_taxa.reset_index(), sim_distribs], axis=1)
    elif sym_mode == 'single_species':
        sim_distribs = simulate_single_species(top_n_species, selected_taxa.taxon.tolist())
        selected_taxa = pd.concat([selected_taxa.reset_index(), sim_distribs], axis=1)
    else:
        print(f"Unknown distribution. Using proportions.")
        selected_taxa['D1'] = selected_taxa['base_proportion']
    selected_taxa.to_csv(os.path.join(outdir, f"simulated_proportions_top{top_n_species}_{sym_mode}_dstr{dstr_args}"))

    log(f"----Success", bcolors.OKGREEN)
    return selected_taxa

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
group3.add_argument('-l', '--read_length', type = int, help='Read length', default=DEFAULT_READ_LENGTH)
group3.add_argument('-f', '--fragment_length', type = int, help='Fragment length', default=DEFAULT_FRAGMENT_LENGTH)
group3.add_argument('-g', '--fragment_sdesv', type = int, help='Variation in sample length.', default=DEFAULT_FRAGMENT_SD)
group3.add_argument('-k', '--n_samples', type = int, help='Number of samples per taxa distribution.', default=1)
group3.add_argument('-n', '--n_threads', type = int, help='Number of threads', default=1)

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
    read_length: int = args.read_length
    fragment_length: int = args.fragment_length
    fragment_sdesv: int = args.fragment_sdesv
    n_threads: int = args.n_threads
    
    create_directories([genomes_path, outdir] + [os.path.join(outdir, s) for s in RES_SUBDIRS.values()])
    #Read Kraken2 database report to get the NCBI FTP paths
    ftproutes: List[Dict[str, str]] = getFTPRoutesFromKraken2Report(ftp_paths)
    #print(ftproutes)

    #Read species to simulate and match with genomes in NCBI FTP
    species2sim: pd.DataFrame = readSpecies2Sim(input_table, ftproutes, os.path.join(outdir, RES_SUBDIRS['tables']))
    if species2sim.shape[0] == 0:
        log("Error: no species found in genomes.", bcolors.FAIL)
        log(str(species2sim.head(3)), bcolors.FAIL)
        exit(1)

    #Calculate simulated proportions
    species_proportion: pd.DataFrame = calculateSpeciesProportion(species2sim, sym_mode, top_n_species, 
                                                                  mode_select_species, dstr_args, dstr_reps, 
                                                                  os.path.join(outdir, RES_SUBDIRS['tables']))

    # Download Genomes
    species_proportion = downloadGenomes_all(species_proportion, genomes_path, rewrite_genomes, BASE_URL, 
                                             os.path.join(outdir, RES_SUBDIRS['tables']))

    # Merge genomes if needed

    # Calculate reads and simulate

    # Merge samples
    
   
  
if __name__ == "__main__":
    main()