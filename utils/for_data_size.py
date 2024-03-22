import argparse
import os
import h5py
from  utils.imresize import *
from pathlib import Path
import scipy.io as scio
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--scale_factor", type=int, default=2, help="4, 2")
    parser.add_argument('--data_for', type=str, default='test', help='')
    parser.add_argument('--src_data_path', type=str, default='./raw_tdata/', help='')

    return parser.parse_args()


def get_datasz():
    args = parse_args()
    angRes, scale_factor = args.angRes, args.scale_factor

    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()
    datasz_dict={}
    for index_dataset in range(len(src_datasets)):
        # if src_datasets[index_dataset] not in ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']:
        #     continue
        name_dataset = src_datasets[index_dataset]
        
        datasz_set={}
        src_sub_dataset = args.src_data_path + name_dataset + '/' + args.data_for + '/'
        for root, dirs, files in os.walk(src_sub_dataset):
            for file in files:
                # print('Generating test data of Scene_%s in Dataset %s......\t' %(file, name_dataset))
                try:
                    data = h5py.File(root + file, 'r')
                    LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                except:
                    data = scio.loadmat(root + file)
                    LF = np.array(data['LF'])

                (U, V, H, W, _) = LF.shape
                datasz=[]
                datasz.append(H)
                datasz.append(W)
                file=file[:-4]
                datasz_set[file]=datasz
            pass
        pass
        datasz_dict[name_dataset]=datasz_set
    pass
    
    return datasz_dict


if __name__ == '__main__':
    datasz_dict=get_datasz()
    print(datasz_dict)
    print(datasz_dict.get('Real').get('IMG_5160'))
    h,w=datasz_dict.get('Real').get('IMG_5160')
    print(h)

