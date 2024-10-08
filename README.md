# IUGC MICCAI challenge

This is the official repository for the paper "A Multi-Task Approach with Weak Supervision for Improved Intrapartum Ultrasound Measurements"


We trained the baseline model for the first 2 tasks

**Authors**: Salem AlNasi, Arsen Abzhanov, Numan Saeed, Mohammad Yaqub


### Setup:
1. Clone this repository
   ```
   git clone https://github.com/innerArsenio/MICCAI-IUGC-2024.git
   ```
2. Install requirements:
   
   - use Python 3.9
   - install requirements:
   ```
   conda create -n iugc python=3.9
   conda activate iugc
   pip install -r requirements.txt
   ```
   
3. Download data
   
   PSFH:
   - download data from the source
  
  
   JNU-IFM:
   - download data from the source

   Merge the two datasets into one folder


### Reproduce our results:
run
```
python CombinedData_Final_train_LateCLS.py
```
