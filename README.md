# LeCoT: Revisiting Network Architecture for Two-View Correspondence Pruning

**Authors:** Luanyuan Dai, Xiaoyu Du, Jinhui Tang  
**Conference/Journal:** **SCIENCE CHINA Information Sciences (SCIS)**  



## Abstract


Two-view correspondence pruning aims to accurately remove incorrect correspondences (outliers) from initial ones and is widely applied to various computer vision tasks. Current popular strategies adopt multilayer perceptron (MLP) as the backbone, supplemented by additional modules to enhance the network ability to handle context information, which is a known limitation of MLPs. In contrast, we introduce a novel perspective for capturing correspondence context information without  extra design modules. To this end, we design a two-view correspondence pruning network called LeCoT, which can naturally leverage global context information at different stages. Specifically, the core design of LeCoT is the Spatial-Channel Fusion Transformer block, a newly proposed component that efficiently utilizes both spatial and channel global context information among sparse correspondences. In addition, we integrate the proposed prediction block that utilizes correspondence features from intermediate stages to generate a probability set, which acts as guiding information for subsequent learning phases, allowing the network to more effectively capture robust global context information. Notably, this prediction block progressively refines the probability set, thereby mitigating the issue of information loss that is common in the traditional one. Extensive experiments prove that the proposed LeCoT outperforms state-of-the-art methods in correspondence pruning, relative pose estimation, homography estimation, visual localization, and $3$D~reconstruction  tasks. The code is provided in https://github.com/Dailuanyuan2024/LeCoT-Revisiting-Network-Architecture-for-Two-View-Correspondence-Pruning.


## Requirements

Please use **Python 3.6**, **OpenCV-contrib-python (3.4.0.12)**, and **PyTorch (>= 1.10)**.  
Other dependencies should be easily installed through `pip` or `conda`.

## Preparing Data

Please follow their instructions to download the training and testing data:

```bash
# Download and extract YFCC100M data
bash download_data.sh raw_data_yfcc.tar.gz 0 88 # YFCC100M
tar -xvf raw_data_yfcc.tar.gz

# Download and extract SUN3D data
bash download_data.sh raw_sun3d_test.tar.gz 0 22 # SUN3D
tar -xvf raw_sun3d_test.tar.gz
bash download_data.sh raw_sun3d_train.tar.gz 0 63
tar -xvf raw_sun3d_train.tar.gz

```

After downloading the datasets, the initial matches for YFCC100M and SUN3D can be generated as follows.  
Here we provide descriptors for **SIFT (default)**, **ORB**, and **SuperPoint**:

```bash 
cd dump_match
python extract_feature.py
python yfcc.py
python extract_feature.py --input_path=.../raw_data/sun3d_test
python sun3d.py

```

----------

## Testing and Training Model

We provide a pretrained model on YFCC100M.  
The results in our paper can be reproduced by running the test script:

```bash
cd LeCoT-Revisiting-Network-Architecture-for-Two-View-Correspondence-Pruning
python main.py --run_mode=test --model_path=../model/yfcc --res_path=../model/yfcc

```

Set `--use_ransac=True` to get results after RANSAC post-processing.

If you want to retrain the model on YFCC100M, run the training script:

```bash
cd LeCoT-Revisiting-Network-Architecture-for-Two-View-Correspondence-Pruning
python main.py

```

You can also retrain the model on SUN3D by modifying related settings in `code/config.py`.
