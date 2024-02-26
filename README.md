This is a script to simulate metagenomics samples. Startng from a distribution of microbes, automatically downloads the genomes from NCBI and uses **wgsim** to generate reads from each species in the specified proportions. 

  1) Reads Kraken2 database report file with FTP routes for each genome, such as this one: https://genome-idx.s3.amazonaws.com/kraken/standard_20240112/library_report.tsv
  2) Reads input distribution table. This can be an OTU table with taxa names in the first column and sample abundances in the next columns, or a simpler table with taxa in the first column and proportions in the second column.
  3) Sorts taxa according to median/mean abundance or prevalence and selects the top *n* organisms.
  4) Calculates microorganism proportions in the simulated samples. Depending on options:
     - One sample with uniform probabilities
     - One sample with the same proportions as in the input
     - *N* samples from a dirichlet distribution with uniform background probabilities
     - *N* samples from a dirichlet distribution with the same background probabilities as in the input sample.
     - One sample per species, each with only one microorganism
  6) Searches the selected microorganisms in the Kraken database report and downloads the genomes.
  7) Merges the different references for each microbial species. Depending on options, chromosomal and plasmidic genomes can be merged together or not. 
  8) Generates reads separately for each species, and, depending on options, separetely for plasmidic and chromosomal genomes. Can generate several samples with different random seeds for each of the *N* input distributions.
  9) Merges simluated reads for each sample.

      
