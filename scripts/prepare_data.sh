set -o xtrace

BASE_ADDR=https://github.com/yuhuan-wu/EDN/releases/download/v1.0

DATA_ADDR=${BASE_ADDR}/SOD_datasets.zip


# download data from github address
wget -c --no-check-certificate --content-disposition $DATA_ADDR -O SOD_datasets.zip
unzip -n SOD_datasets.zip -d data/
