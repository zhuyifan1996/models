#! /bin/bash

sudo apt-get update
#git
sudo apt-get install git-all

#spark
#git clone https://github.com/apache/spark.git
#cd spark
#build/mvn -DskipTests clean package

#conda
mkdir -p tools && cd tools
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo apt-get install bzip2
bash Miniconda3-latest-Linux-x86_64.sh

#gpu
# sudo apt-get install pciutils # device list
# sudo apt-get install build-essential #gcc
# sudo apt-get install linux-headers-$(uname -r) #install kernel-version header

echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda -y
  apt-get install linux-headers-$(uname -r) -y
fi
