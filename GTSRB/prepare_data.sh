mkdir datasets
cd datasets
mkdir gtsrb
cd gtsrb
unzip GTSRB-Training_fixed.zip
cd ../..
python crop_img.py
