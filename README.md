# TAFSSR


**we propose a novel approach that incorporates the Transformer into the model to learn global contextual information
and better capture long-range relationships within LF structures. During the feature fusion process, each feature extractor contains different information from the 4D LF dataset. Since directly concatenating all branch outputs may not be a preferable solution, we propose an attention fusion operation to weightedly combine the features from each branch.**


<br>

## Contributions
* **We proposed a network based on Transformer and attention fusion for LF image SR.**
* **Compared with the baseline, the super-resolution performance was significantly improved with our's method..**
* **We share the codes, models and results of existing methods to help researchers better get access to this area.**


## Datasets
**We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for both training and test. 
Please first download our datasets via [Baidu Drive](https://pan.baidu.com/s/1AyMJUSwwDf9T7Tr8xhs1nw) (key:lfsr) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place the 5 datasets to the folder `./datasets/`.**

* **Our project has the following structure:**
  ```
  ├──./datasets/
  │    ├── EPFL
  │    │    ├── training
  │    │    │    ├── Bench_in_Paris.mat
  │    │    │    ├── Billboards.mat
  │    │    │    ├── ...
  │    │    ├── test
  │    │    │    ├── Bikes.mat
  │    │    │    ├── Books__Decoded.mat
  │    │    │    ├── ...
  │    ├── HCI_new
  │    ├── ...
  ```
* **Run `Generate_Data_for_Training.m` or `Generate_Data_for_Training.py` to generate training data. The generated data will be saved in `./data_for_train/` (SR_5x5_2x, SR_5x5_4x).**
* **Run `Generate_Data_for_Test.m` or `Generate_Data_for_Test.py` to generate test data. The generated data will be saved in `./data_for_test/` (SR_5x5_2x, SR_5x5_4x).**
* **Run `Generate_Data_for_inference.py` to generate test data. The generated data will be saved in `./data_for_inference/` (SR_5x5_2x, SR_5x5_4x).**
<br>

## Commands for Training
* **Run **`train.py`** to perform network training. Example for training [model_name] on 5x5 angular resolution for 2x/4x SR:**
  ```
  $ python train.py --model_name [model_name] --angRes 5 --scale_factor 2 --batch_size 8
  $ python train.py --model_name [model_name] --angRes 5 --scale_factor 4 --batch_size 4
  ```
* **Checkpoints and Logs will be saved to **`./log/`**, and the **`./log/`** has the following structure:**
  ```
  ├──./log/
  │    ├── SR_5x5_2x
  │    │    ├── [dataset_name]
  │    │         ├── [model_name]
  │    │         │    ├── [model_name]_log.txt
  │    │         │    ├── checkpoints
  │    │         │    │    ├── [model_name]_5x5_2x_epoch_01_model.pth
  │    │         │    │    ├── [model_name]_5x5_2x_epoch_02_model.pth
  │    │         │    │    ├── ...
  │    │         │    ├── results
  │    │         │    │    ├── VAL_epoch_01
  │    │         │    │    ├── VAL_epoch_02
  │    │         │    │    ├── ...
  │    │         ├── [other_model_name]
  │    │         ├── ...
  │    ├── SR_5x5_4x
  ```

<br>

## Commands for Test
* **Run **`test.py`** to perform network inference. Example for test [model_name] on 5x5 angular resolution for 2x/4xSR:**
  ```
  $ python test.py --model_name [model_name] --angRes 5 --scale_factor 2  
  $ python test.py --model_name [model_name] --angRes 5 --scale_factor 4 
  ```
  
* **The PSNR and SSIM values of each dataset will be saved to **`./log/`**, and the **`./log/`** has the following structure:**
  ```
  ├──./log/
  │    ├── SR_5x5_2x
  │    │    ├── [dataset_name]
  │    │        ├── [model_name]
  │    │        │    ├── [model_name]_log.txt
  │    │        │    ├── checkpoints
  │    │        │    │   ├── ...
  │    │        │    ├── results
  │    │        │    │    ├── Test
  │    │        │    │    │    ├── evaluation.xls
  │    │        │    │    │    ├── [dataset_1_name]
  │    │        │    │    │    │    ├── [scene_1_name]
  │    │        │    │    │    │    │    ├── [scene_1_name]_CenterView.bmp
  │    │        │    │    │    │    │    ├── [scene_1_name]_SAI.bmp
  │    │        │    │    │    │    │    ├── views
  │    │        │    │    │    │    │    │    ├── [scene_1_name]_0_0.bmp
  │    │        │    │    │    │    │    │    ├── [scene_1_name]_0_1.bmp
  │    │        │    │    │    │    │    │    ├── ...
  │    │        │    │    │    │    │    │    ├── [scene_1_name]_4_4.bmp
  │    │        │    │    │    │    ├── [scene_2_name]
  │    │        │    │    │    │    ├── ...
  │    │        │    │    │    ├── [dataset_2_name]
  │    │        │    │    │    ├── ...
  │    │        │    │    ├── VAL_epoch_01
  │    │        │    │    ├── ...
  │    │        ├── [other_model_name]
  │    │        ├── ...
  │    ├── SR_5x5_4x
  ```

<br>

## Contact
**Welcome to raise issues or email to [hrzh.cq@outlook.com] for any question regarding our TAFSSR.**


