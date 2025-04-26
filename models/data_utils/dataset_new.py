# [Source] https://github.com/urbanaudiosensing/Models/blob/main/data_utils/dataset.py

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import numpy as np
import json
import os
import pandas as pd

from .transforms import VGGish, VGGish_PreProc, ASTFeatureExtractor, AST_PreProc

from typing import Literal
from tqdm import tqdm


METADATA_PATH = 'Labels/cam2rec.json'
AUDIO_EXT = '.npy' #Audio Files are pre-processed into 1-channel 16Khz numpy arrays
AUDIO_PREFIX = 'DR-05X-{0}'
LABELS_SUFFIX = '{}.csv'
LABEL_HEADER = 'recorder{0}_{1}m'
VIEW_PREFIX = 'view_'
SR = 16000
IGNORE_INDEX = -1 #label for ground truth obstructed by view, ignored in gradient computation

class ASPEDDataset(Dataset):
    
    _transform = torch.nn.Identity() #Pre-process the raw waveform as input to n_classesl
    n_classes = 'cls'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    min_val = 1
    max_val = 1
    do_transform = True

    @classmethod
    def transform(cls, *args) -> torch.Tensor:
        if type(ASPEDDataset._transform) == torch.nn.Identity:
            return ASPEDDataset._transform(args[0])
        else:
            return ASPEDDataset._transform(*args)
    
    @classmethod
    def threshold(cls, X: torch.Tensor) -> torch.Tensor:
        if ASPEDDataset.n_classes == 1:
            return torch.clamp(X / 6, max=1.0) #normalize values for regression
        else: #Constrain class labels to [0, n_classes - 1], where all values < min_val are set to 0
            return torch.clamp(X // ASPEDDataset.min_val, max=ASPEDDataset.n_classes - 1) 
        
    def __init__(self, rec_path, labels, segment_length = 10, n_classes = 2):
        #Instance vars
        self.rec_path = rec_path
        self.labels = labels

        #Class Vars
        ASPEDDataset.segment_length = segment_length
        ASPEDDataset.n_classes = n_classes #1 sets model to regression mode
         
        # Load data into a NumPy memory-mapped array
        self.data, self.indices = self._load_data()

        #for locations with uneven audio-labels
        self.labels = self.labels[:self.indices[-1] // SR]

    def __len__(self):
        #print(f"len(self.labels): {len(self.labels)}") # 158994
        #print(f"self.segment_length + 1: {self.segment_length + 1}") # 2
        return len(self.labels) - (self.segment_length + 1)
    
    # YONGHYUN
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.long()

        # Load the data sample at the specified index
        c = np.searchsorted(self.indices, idx * SR, side='right') - 1
        file_offset = idx * SR - self.indices[c]

        try:
            if (c < len(self.data) - 1) and (file_offset + (SR * self.segment_length) >= len(self.data[c])):
                item_1 = torch.Tensor(self.data[c][file_offset:].copy())
                item_2 = torch.Tensor(self.data[c+1][:(SR * self.segment_length) - item_1.shape[0]].copy())
                item = torch.cat((item_1, item_2), axis=0)
            else:
                item = torch.Tensor(self.data[c][file_offset:file_offset + (SR * self.segment_length)].copy())
        except Exception as e:
            print(e)
            raise Exception(idx, self.indices, len(self), file_offset, len(self.indices), c, idx * SR, self.indices[c], self.indices[-1], idx * SR + (SR * self.segment_length), len(self.data), (c < len(self.data) - 1))

        labels = self.labels[idx:idx+self.segment_length]

        # ✅ 🚨 bus frame 검사
        '''
        훈련하면서 실행된다.
        busFrame detected in /media/backup_SSD/ASPED_v2_npy/Session_11072023/FifthSt_C/Audio/recorder1_DR-05X-03 at 77113
        new_idx: 77123
        busFrame detected in /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_B/Audio/recorder1_DR-05X-10 at 91582
        new_idx: 91592
        '''
        if torch.any(labels == IGNORE_INDEX):  # IGNORE_INDEX가 있는 경우
            # print(f"busFrame detected in {self.rec_path} at {idx}")
            new_idx = (idx + self.segment_length) % len(self.labels)  # 다음 샘플로 이동
            # print(f"new_idx: {new_idx}")
            # input()
            return self.__getitem__(new_idx)  # 재귀적으로 새로운 샘플을 불러옴

        if not ASPEDDataset.do_transform:
            return item.view(self.segment_length, SR), ASPEDDataset.threshold(labels)

        return ASPEDDataset.transform(item.view(self.segment_length, SR), SR), ASPEDDataset.threshold(labels)

    
    def _zeropad(self, X, num):
        if len(X) >= num:
            return X
        return torch.cat([X, torch.zeros((num - X.shape[0],))])


    def _load_data(self):
        # Assuming each file contains data that can be loaded into a NumPy array
        # Modify this function based on your data loading logic
        file_list = sorted([os.path.join(self.rec_path, x) for x in os.listdir(self.rec_path) if x.endswith(AUDIO_EXT)], key=lambda x: x.split('/')[-1])
        #file_list = [file_list[-1]] + file_list[1:-1] #to account for split_{}.npy
        data_list = [np.load(file_path, mmap_mode='r') for file_path in file_list]

        indices = np.cumsum([0] + [x.shape[0] for x in data_list])

        return (data_list, indices)
     
    @staticmethod
    def from_dirs_vb(dirs: list[str], radius: Literal[1,3,6,9] = 6, segment_length = 10,
                  transform :Literal['vggish', 'vggish-mel', 'ast', None] = None, n_classes= 2) -> ConcatDataset:
        
        ASPEDDataset.n_classes = n_classes
        dataset = []
        
        for d in tqdm(dirs):
            for s in os.listdir(d):
                path = os.path.join(d,s)
                #print(f"path: {path}") # /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C
                #input()
                try:
                    # /media/backup_SSD/ASPED_v2_npy_labels_with_bus/Session_07262023/FifthSt_A/0001.csv
                    if path.split('/')[-3] == 'ASPED_v2_npy': # ASPED v_b
                        label_files = list(filter(lambda x: x.endswith(LABELS_SUFFIX[-3:]), [os.path.join(os.path.join(path.replace('ASPED_v2_npy', 'ASPED_v2_npy_labels_with_bus')), x) for x in os.listdir(os.path.join(path.replace('ASPED_v2_npy', 'ASPED_v2_npy_labels_with_bus')))]))
                    else: # ASPED v_a
                        label_files = list(filter(lambda x: x.endswith(LABELS_SUFFIX[-3:]) and not
                                        'processed' in x, [os.path.join(path,'Labels', x) for x in os.listdir(os.path.join(path, 'Labels'))]))
                    #print(f"label_files: {label_files}") # label_files: ['/media/backup_SSD/ASPED_v2_npy_labels_with_bus/Session_07262023/FifthSt_C/0008.csv',

                    labels = [pd.read_csv(x) for x in sorted(label_files)]
                    labels = pd.concat(labels, axis=0)
                except Exception as e:
                    print(e)
                    continue
                audio_path = os.path.join(path, 'Audio')
                
                for i, rec in enumerate(sorted(os.listdir(audio_path))):
                    views_key = VIEW_PREFIX + LABEL_HEADER.format(i + 1, radius)

                    if views_key in labels:
                        views = torch.Tensor(labels[views_key].values)
                    else:
                        views = torch.zeros(len(labels))  # fallback if no view info

                    label = torch.Tensor(labels[LABEL_HEADER.format(i + 1, radius)].values)

                    # # IF BusFrame detected, Mask with IGNORE_INDEX (-1)
                    if 'busFrame' in labels.columns:
                        bus_mask = torch.Tensor(labels['busFrame'].values) == 1
                        label[bus_mask] = IGNORE_INDEX
                        #print(f"busFrame detected in {rec} at {path}")
                        #input()
                    
                    ASPEDDataset.max_val = max(label.max().item(), ASPEDDataset.max_val)
                    working_dir = os.path.join(audio_path, rec)
                    dataset.append(ASPEDDataset(working_dir, label, segment_length=segment_length, n_classes=n_classes))
                    
        c_dataset = ConcatDataset(dataset)

        if transform == 'vggish':
            ASPEDDataset._transform = VGGish()#.to(ASPEDDataset.device)
        elif transform == 'vggish-mel':
            ASPEDDataset._transform = VGGish_PreProc()#.to(ASPEDDataset.device)
        elif transform == 'ast':
            ASPEDDataset._transform = AST_PreProc()
        else:
            ASPEDDataset._transform = torch.nn.Identity()
        return c_dataset

if __name__ == '__main__':
    '''
    Testing
    '''
    import time
    from random import randint

    #TEST_DIR = ["/media/fast_drive/audio_data/Test_7262023", "/media/fast_drive/audio_data/Test_08092023", 
                #"/media/fast_drive/audio_data/Test_10242023", "/media/fast_drive/audio_data/Test_11072023"]
    TEST_DIR = ["/media/fast_drive/audio_data_aligned/Session_5242023", "/media/fast_drive/audio_data_aligned/Session_6012023", 
                "/media/fast_drive/audio_data_aligned/Session_6072023", "/media/fast_drive/audio_data_aligned/Session_6212023",
                "/media/fast_drive/audio_data_aligned/Session_6282023"]
    X = ASPEDDataset.from_dirs_vb(TEST_DIR, segment_length=10, transform='vggish-mel')
    print(type(X), len(X))

    lim = 100 #len(X)

    start = time.time()

    num_zero = 0
    err = list()
    print("testing....")
    for x in tqdm(range(lim)):
        idx = randint(0, len(X))
        batch = X[idx]
        if batch[0].sum() == 0:
            num_zero += 1
            err.append(idx)
    end = time.time()
    print([x.shape for x in batch])
    print(err)

    print(f'time per execution: {((len(X)*((end - start)/lim)))/60:.2e}m', len(err))
