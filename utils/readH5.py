import h5py
import numpy as np


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data_hz = np.array(hf.get('Hr_SAI_y'))
        data_vt = np.array(hf.get('Lr_SAI_y'))
        data_rf = np.array(hf.get('Sr_SAI_cbcr'))
        #train_data = np.transpose(data_hz, (0, 3, 2, 1))
        #train_label = np.transpose(label, (0, 3, 2, 1))
        return data_hz, data_vt, data_rf

if __name__=="__main__":
    data_hz, data_vt, data_rf=read_training_data("/storage/BK/pycharm_project/BasicLFSR-main/data_for_test/SR_5x5_4x/Real/0725.h5")
    
    print(data_hz)