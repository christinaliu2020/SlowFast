### setup file
#conda create -n SlowFast python=3.8 -y
#conda activate SlowFast
conda update -n base -c defaults conda -y
conda install -c conda-forge libiconv -y
conda install ffmpeg=4.2 -c conda-forge -y
#
pip install numpy simplejson psutil opencv-python pillow requests urllib3 scipy pandas tqdm scikit-learn
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-9
sudo apt install g++-9
conda install -y -c conda-forge gxx=9
#sudo apt install gcc-9
conda install -y av -c conda-forge
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install pytorchvideo
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo

#git clone --recursive https://github.com/pytorch/pytorch
#pip install cmake --upgrade
#cd pytorch
#git submodule update --init --recursive
#python setup.py install


#conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# conda remove torchvision torchaudio -y
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install -U iopath
conda install -y -c conda-forge moviepy
pip install 'git+https://github.com/facebookresearch/fairscale'
#pip install -U torch torchvision cython

#git clone https://github.com/pytorch/vision.git
#cd vision
#git checkout v0.9.0
#python setup.py install
#cd ..

#pip install -U git+

python setup.py build develop

pip uninstall pytorchvideo
pip install "git+https://github.com/facebookresearch/pytorchvideo.git"