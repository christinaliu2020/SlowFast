conda install -y -c conda-forge gxx=9
conda install -y av -c conda-forge
pip install pytorchvideo
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo

conda update -n base -c defaults conda -y
conda install -c conda-forge libiconv
conda install ffmpeg=5.1 -c conda-forge -y
pip install numpy simplejson psutil opencv-python pillow requests urllib3 scipy pandas tqdm scikit-learn

#conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install -U iopath
conda install -y -c conda-forge moviepy
pip install 'git+https://github.com/facebookresearch/fairscale'
#pip install -U torch torchvision cython

git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.9.0
python setup.py install

pip install -U git+

python setup.py build develop

pip install "git+https://github.com/facebookresearch/pytorchvideo.git"