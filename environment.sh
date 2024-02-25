conda create -y --name wgsim-env python=3.10
conda activate wgsim-env
conda install -y -c bioconda wgsim
conda install -y numpy pandas ftputil matplotlib
