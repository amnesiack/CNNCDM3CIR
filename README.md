# The code and trained model for paper "Color Image Restoration Exploiting Inter-channel Correlation with a 3-stage CNN"

### K. Cui, A. Boev, E. Alshina and E. Steinbach, Color Image Restoration Exploiting Inter-channel Correlation with a 3-stage CNN,IEEE-JSTSP, 2020.
#### Kai Cui <Kai.cui@tum.de>
#### Lehrstuhl fuer Medientechnik
#### Technische Universitaet Muenchen
#### Last modified 05.02.2021

1. There are three subtasks in our paper, color demosaicking (CDM), compression artifacts reduction (CAR), and real-world color image denoising (RIDN). The code and the trained models are in the corresponding folder.

2. You need to download the testing datasets to run the demo test. We summarize the datasets [here](https://tumde-my.sharepoint.com/:f:/g/personal/kai_cui_tum_de/Elln4Vp3-AdBqCHtvuHe4VMB0tQdIV238QuTHMJjum0vYg?e=B2y4nR). Unzip the datasets and put it into the data folder. If you have your own dataset, please follow the readme.md in the data folder to organize the dataset.

3. For CDM, run "python main_py3_tfrecord.py" to test the Kodak dataset. When testing other datasets, simply add " --test_set NAME". e.g., "python main_py3_tfrecord.py --test_set McM". It supports also the emsemble testing mode, run "python main_py3_tfrecord.py --phase ensemble"

4. For CAR, run "python main_py3_tfrecord.py" to test the LIVE1 dataset with qp = 10, use "--qp XX" to test different QPs, e.g., "python main_py3_tfrecord.py --qp 100". 

5. For RIDN, run "python main_py3_tfrecord.py" to test the SIDD_validation dataset.

6. The code support both CPU and GPU, default is GPU 0, use "--gpu -1" to run on CPU or choose other GPUs.

7. Please read our paper for more details!

8. Have fun!

@ARTICLE{CNNCIR-JSTSP-2020,
  author={K. {Cui} and A. {Boev} and E. {Alshina} and E. {Steinbach}},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Color Image Restoration Exploiting Inter-channel Correlation with a 3-stage CNN}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JSTSP.2020.3043148}}