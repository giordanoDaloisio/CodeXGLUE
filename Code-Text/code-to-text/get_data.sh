source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate codex

unzip dataset.zip
cd dataset
wget https://zenodo.org/record/7857872/files/go.zip
wget https://zenodo.org/record/7857872/files/java.zip
wget https://zenodo.org/record/7857872/files/javascript.zip
wget https://zenodo.org/record/7857872/files/php.zip
wget https://zenodo.org/record/7857872/files/python.zip
wget https://zenodo.org/record/7857872/files/ruby.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
