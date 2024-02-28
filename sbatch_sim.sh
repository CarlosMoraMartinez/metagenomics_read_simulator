#!/bin/bash

#################################################################################################################
## script para el quality control de los ficheros                                                              ##
##                                                                                                             ##
## SBATCH --job-name=name: Nombre del trabajo                                                                  ##
## SBATCH --output=dir/name_%A_%a.out #archivo_%j.out: Directorio y nombre del fichero de salida               ##
## SBATCH --error=dir/name_%A_%a.err #archivo_%j.err: Directorio y nombre del ficherro de errores              ##
## SBATCH --array=1-n: paralelizacion del trabajo, lanza simultaneamente 16 trabajos                           ##
## SBATCH --partition=long: la particion que queremos usar                                                     ##
##          "slimits" para ver las opciones                                                                    ##
## SBATCH --cpus-per-task 4                                                                                    ##
## SBATCH --mem 10G                                                                                            ##
#################################################################################################################

#SBATCH --job-name=k2sim
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --qos=long
#SBATCH --cpus-per-task 4
#SBATCH --mem=12G
#SBATCH --time=8-00:00:00 # 8 días 

#Do this before executing sbatch
module load anaconda #3_2022.10
conda activate wgsim-env

K2LIB=/home/mysanz/projects/cmora/benchmark/data/library_report.tsv
INTAB=/home/mysanz/projects/cmora/benchmark/metagenomics_read_simulator/example_data/remove_tanda2_raw_counts.tsv
GENOMEDIR=ref_genomes1

NREADS=20000000

#python metagenomics_read_simulator/simulate_metagenomes.py -o unif10 -p $K2LIB -d $GENOMEDIR -i $INTAB -s dirichlet_unif -u 10 -m mean_abundance -a 10 -b 20 -x 0 -k 1 -t $NREADS

python metagenomics_read_simulator/simulate_metagenomes.py -o unif100 -p $K2LIB -d $GENOMEDIR -i $INTAB -s dirichlet_unif -u 100 -m mean_abundance -a 10 -b 20 -x 0 -k 1 -t $NREADS

#python metagenomics_read_simulator/simulate_metagenomes.py -o prop10 -p $K2LIB -d $GENOMEDIR -i $INTAB -s dirichlet_prop -u 10 -m mean_abundance -a 10 -b 20 -x 0 -k 1 -t $NREADS

python metagenomics_read_simulator/simulate_metagenomes.py -o prop100 -p $K2LIB -d $GENOMEDIR -i $INTAB -s dirichlet_prop -u 100 -m mean_abundance -a 10 -b 20 -x 0 -k 1 -t $NREADS


#python metagenomics_read_simulator/simulate_metagenomes.py -o single50 -p $K2LIB -d $GENOMEDIR -i $INTAB -s single_species -u 50 -m mean_abundance -a 10 -b 1 -x 0 -k 1 -t 5000000