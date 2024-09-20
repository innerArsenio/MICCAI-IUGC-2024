# IUGC MICCAI challenge

This is the official repository for the paper "insert_name"


We trained the baseline model for the first 2 tasks
**Authors**: [Salem AlNasi], [Numan Saeed], [Mohammad Yaqub], [Arsen Abzhanov]


### Setup:
1. Clone this repository
   ```
   git clone https://github.com/[insert_here]
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
   - download data from 'insert_here'
  
  
   JNU-IFM:
   - download data from 'insert_here'

   Merge the two datasets into one folder


### Reproduce our results:
run
```
python [-m do we need that?] CombinedData_Final_train_LateCLS.py
```