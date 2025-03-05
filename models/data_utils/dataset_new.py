# ONGOING

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
    
    _transform = torch.nn.Identity() #Pre-process the raw waveform as input to n_classes
    n_classes = 'cls'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    min_val, max_val = 1, 1
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
  
    # v.b
    def __init__(self, rec_path, labels, segment_length=10, n_classes=2, valid_segments=None):
        """
        Args:
            rec_path (str): 오디오 파일이 위치한 경로
            labels (torch.Tensor): 해당 오디오의 라벨
            segment_length (int): 세그먼트 길이 (초 단위)
            n_classes (int): 분류할 클래스 개수
            valid_segments (list[list[int]]): 사용할 세그먼트 리스트 (프레임 단위)
        """
        self.rec_path = rec_path
        self.labels = labels
        self.segment_length = segment_length
        self.n_classes = n_classes

        # 🚨 valid_segments 전달 여부 확인
        if valid_segments is None:
            raise ValueError(f"valid_segments must be provided for {rec_path}")

        # 📌 _load_data()를 valid_segments와 함께 호출
        """
        data_list(2): [memmap([-0.03724556, -0.06404861, -0.05645911, ...,  0.00273331,
         0.00312567,  0.00660427], dtype=float32), memmap([ 0.00594037,  0.0100211 ,  0.01075488, ..., -0.00521716,
        -0.00347648, -0.0021581 ], dtype=float32), memmap([ 0.00048022,  0.00277342,  0.00555613, ..., -0.00367861,
        -0.00478825, -0.00589047], dtype=float32), memmap([-0.00373731, -0.00567107, -0.00482633, ..., -0.00305535,
        -0.00257165, -0.00305301], dtype=float32), memmap([-0.00207552, -0.00388347, -0.00410776, ...,  0.0245531 ,
         0.02039828,  0.0183718 ], dtype=float32), memmap([0.00990254, 0.01569039, 0.01580276, ..., 0.06184042, 0.06341559,
        0.07375804], dtype=float32), memmap([ 0.05028185,  0.07935696,  0.07999854, ..., -0.02744678,
        -0.02725497, -0.02887899], dtype=float32), memmap([-0.0169642 , -0.02428075, -0.01931662, ...,  0.00572748,
         0.0070189 ,  0.01036328], dtype=float32), memmap([0.00905126, 0.0148844 , 0.01482448, ..., 0.01046601, 0.00816145,
        0.00484289], dtype=float32), memmap([ 0.00049787, -0.00220863, -0.00512875, ..., -0.03282483,
        -0.03312571, -0.03696375], dtype=float32), memmap([-0.02328174, -0.03738508, -0.03470192, ..., -0.00043505,
        -0.00590625, -0.01225242], dtype=float32), memmap([-0.00659331, -0.00716433, -0.00391278, ..., -0.05658222,
        -0.04804996, -0.04405289], dtype=float32)]
        indices: [         0  230400000  460800000  691200000  921600000 1152000000
        1382400000 1612800000 1843200000 2073600000 2304000000 2534400000
        2764800000]
        """
        # v.b (w/o bus frames)
        self.data, self.indices = self._load_data(valid_segments)

        # 라벨 크기 조정 (음성 길이에 맞추기)
        self.labels = self.labels[:self.indices[-1] // SR]
       
    # v.a       
    # def __init__(self, rec_path, labels, segment_length = 10, n_classes = 2):
    #     #Instance vars
    #     self.rec_path = rec_path
    #     self.labels = labels

    #     #Class Vars
    #     ASPEDDataset.segment_length = segment_length
    #     ASPEDDataset.n_classes = n_classes #1 sets model to regression mode
         
    #     # Load data into a NumPy memory-mapped array
    #     self.data, self.indices = self._load_data()

    #     #for locations with uneven audio-labels
    #     self.labels = self.labels[:self.indices[-1] // SR]

    def __len__(self):
        print(f"len(self.labels): {len(self.labels)}") # why 10 in v.b?
        print(f"len(self.labels) - (self.segment_length + 1): {len(self.labels) - (self.segment_length + 1)}") # -1
        return len(self.labels) - (self.segment_length + 1)
    
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
                item = torch.cat((item_1, item_2), axis = 0)
            else:
                item = torch.Tensor(self.data[c][file_offset:file_offset + (SR * self.segment_length)].copy())
        except Exception as e:
            raise Exception(idx, self.indices, len(self),file_offset,len(self.indices), c, idx * SR, self.indices[c], self.indices[-1], idx * SR + (SR * self.segment_length), len(self.data), (c < len(self.data) - 1))

        labels = self.labels[idx:idx+self.segment_length]
        if not ASPEDDataset.do_transform:
            return item.view(self.segment_length, SR), ASPEDDataset.threshold(labels)

        return ASPEDDataset.transform(item.view(self.segment_length, SR), SR), ASPEDDataset.threshold(labels)
    
    def _zeropad(self, X, num):
        if len(X) >= num:
            return X
        return torch.cat([X, torch.zeros((num - X.shape[0],))])

    # ASPED v.b (New)
    def _load_data(self, valid_segments):
        """
        특정 segment에 해당하는 오디오 데이터만 Lazy Loading 방식으로 로드하되, 
        버스 프레임(busFrame)이 포함된 프레임을 제외한다.
        
        Args:
            valid_segments (dict): 파일별 valid_segments 매핑 (busFrame 제외된 상태)
            
        Returns:
            tuple: (메모리 매핑된 오디오 데이터 리스트, 프레임 인덱스 리스트)
        """
        if os.path.isfile(self.rec_path):  
            file_list = [self.rec_path]
        elif os.path.isdir(self.rec_path):  
            file_list = sorted(
                [os.path.join(self.rec_path, x) for x in os.listdir(self.rec_path) if x.endswith(AUDIO_EXT)],
                key=lambda x: x.split('/')[-1]
            )
        else:
            raise ValueError(f"Invalid path: {self.rec_path}")

        data_list = []
        indices = [0]

        # ✅ Lazy Loading 적용 (mmap_mode='r' 활용, busFrame 제외)
        for file_path in file_list:
            try:
                # 🚀 Lazy Loading: np.load(mmap_mode='r') 활용하여 전체 데이터를 RAM에 올리지 않음
                audio_data = np.load(file_path, mmap_mode='r')  

                # 🔹 현재 파일에서 사용할 유효한 세그먼트만 필터링 (busFrame 제외된 상태)
                file_valid_segments = valid_segments.get(file_path, [])

                for segment in file_valid_segments:
                    start_frame = max(0, segment[0] * SR)  # 시작 프레임 (초 → 샘플 변환)
                    end_frame = min(len(audio_data), (segment[-1] + 1) * SR)  # 끝 프레임

                    if start_frame < end_frame:
                        # ✅ 데이터를 복사하지 않고, 해당 파일과 인덱스를 저장 (Lazy Loading 유지)
                        data_list.append((audio_data, start_frame, end_frame))
                        indices.append(indices[-1] + (end_frame - start_frame))

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue

        return data_list, np.array(indices)

    # ASPED v.b
    # def _load_data(self, valid_segments):
    #     """
    #     특정 segment에 해당하는 오디오 데이터만 로드하는 함수.
    #     Args:
    #         valid_segments (list[list[int]]): 선택된 segment의 시작 및 끝 프레임 리스트.
    #     Returns:
    #         tuple: (오디오 데이터 리스트, 프레임 인덱스 리스트)
    #     """
    #     if os.path.isfile(self.rec_path):  # 🔹 만약 self.rec_path가 파일이라면 직접 로드
    #         file_list = [self.rec_path]
    #     elif os.path.isdir(self.rec_path):  # 🔹 폴더라면 내부의 .npy 파일 목록을 가져오기
    #         file_list = sorted(
    #             [os.path.join(self.rec_path, x) for x in os.listdir(self.rec_path) if x.endswith(AUDIO_EXT)],
    #             key=lambda x: x.split('/')[-1]
    #         )
    #     else:
    #         raise ValueError(f"Invalid path: {self.rec_path}")

    #     # print(f"file_list: {file_list}")
    #     '''
    #     file_list: ['/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0001.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0002.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0003.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0004.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0005.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0006.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0007.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0008.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0009.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0010.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0011.npy', '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11/0012.npy']
    #     '''
    #     # print(f"valid_segments: {valid_segments}")

    #     data_list = []
    #     indices = [0]

    #     total_audio_length = 0  # 🔹 전체 오디오 길이 누적 변수
    #     file_segment_map = {}  # 🔹 파일별 valid_segments 저장

    #     '''
    #     (지금 이게 문제)
    #     같은 valid_segments를 모든 파일에서 참조중. 이러면 안됨.
    #     애초에 valid_ssegment가 파일별로 지정이 되어있어야 함.
    #     valid_segments: [[4019, 4028]]
    #     file_path: /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_B/Audio/recorder1_DR-05X-10/0011.npy
    #     valid_segments: [[4019, 4028]]
    #     file_path: /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_B/Audio/recorder1_DR-05X-10/0001.npy
    #     valid_segments: [[4323, 4332]]
    #     file_path: /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_B/Audio/recorder1_DR-05X-10/0002.npy
    #     valid_segments: [[4323, 4332]]
    #     file_path: /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_B/Audio/recorder1_DR-05X-10/0003.npy
    #     valid_segments: [[4323, 4332]]
    #     file_path: /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_B/Audio/recorder1_DR-05X-10/0004.npy
    #     valid_segments: [[4323, 4332]]
    #     file_path: /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_B/Audio/recorder1_DR-05X-10/0005.npy
    #     valid_segments: [[4323, 4332]]
    #     '''

    #     # ✅ 각 파일에 대해 유효한 valid_segments를 저장
    #     for file_path in file_list:
    #         # print(f"file_path: {file_path}")
    #         try:
    #             audio_data = np.load(file_path, mmap_mode='r', allow_pickle=True)
    #             if isinstance(audio_data, tuple):
    #                 audio_data = audio_data[0]

    #             # print(f"file_path: {file_path}") # /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_B/Audio/recorder1_DR-05X-10/0008.npy
                
    #             file_length = len(audio_data) # 230400000 (npy 4시간)
    #             file_index = int(str(file_path)[-8:-4]) # 0001, 0002 ...
    #             start_time_offset = (file_index-1) * file_length
    #             # print(F"start_time_offset: {start_time_offset}")
    #             # print(f"valid_segments: {valid_segments}")
    #             # print(f"segment: {segment}")

    #             # ✅ 현재 파일과 관련된 segment만 필터링
    #             file_valid_segments = []
    #             for segment in valid_segments:
    #                 # print(f"segment: {segment}")
    #                 seg_start = segment[0] * SR
    #                 seg_end = (segment[-1]+1) * SR

    #                 # 🔹 현재 파일 내에서 유효한 segment만 선택
    #                 if seg_start >= start_time_offset + file_length or seg_end <= start_time_offset:
    #                     # print(f"------")
    #                     # print(f"SKIP")
    #                     # print(f"------")
    #                     continue  # 현재 파일 범위를 벗어난 segment는 무시
                    
    #                 # # 📌 파일 내 상대적 위치로 변환
    #                 # print(f"SUCCESS")
                    
    #                 # print(f"file_path: {file_path}") # file_path: /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0001.npy
    #                 # print(F"segment: {segment}") # [0, 910]
    #                 # print(f"seg_start: {seg_start}") # 0
    #                 # print(f"start_time_offset: {start_time_offset}") # 0
    #                 # # print(f"file_length: {file_length}")
    #                 # print(f"seg_end: {seg_end}") # 14576000 (911 seconds)
    #                 # 참고로 [0, 1, ..., 910] 이 valid_segments였음. (911 프레임 직전까지 --> 910.9999 초)
                    
    #                 local_start = max(0, seg_start - start_time_offset)
    #                 local_end = min(file_length, seg_end - start_time_offset)
    #                 #print(f"local_start: {local_start}") # 0
    #                 #print(F"local_end: {local_end}") # 14576000 
    #                 # input() # DEBUG

    #                 if local_end > local_start:
    #                     file_valid_segments.append((local_start, local_end))
                    
    #             file_segment_map[file_path] = file_valid_segments  # 🔹 파일별 valid_segments 저장
    #             #print(f"file_segment_map: {file_segment_map}")
    #             #input()
    #             # 각 파일에 유효한 오디오 샘플 구간을 담습니다. 
    #             ''' (14565000 직전까지)
    #             file_segment_map: {'/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0001.npy': [(0, 14576000)], 
    #             '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0002.npy': [], 
    #             '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0003.npy': [], 
    #             '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0004.npy': [], 
    #             '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0005.npy': [], 
    #             '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0006.npy': [], 
    #             '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0007.npy': [], 
    #             '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0008.npy': [], 
    #             '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0009.npy': [], 
    #             '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0010.npy': []}
    #             '''
    #         except Exception as e:
    #             #print(f"Error processing {file_path}: {e}")
    #             continue

    #     # ✅ 파일별 valid_segments 적용하여 오디오 데이터 추출
    #     for file_path in file_list:
    #         try:
    #             audio_data = np.load(file_path, mmap_mode='r', allow_pickle=True)
    #             if isinstance(audio_data, tuple):
    #                 audio_data = audio_data[0]

    #             file_length = len(audio_data)
    #             file_start = 0  # 🔹 파일의 시작점을 0으로 설정
    #             file_end = file_length

    #             segment_audio = []
    #             file_valid_segments = file_segment_map.get(file_path, [])  # 🔹 해당 파일과 관련된 segment만 사용

    #             #print(f"\n📌 Processing {file_path}")
    #             #print(f"  - file_valid_segments: {file_valid_segments}")  # 🔥 개별 파일의 valid_segments 출력

    #             for segment in file_valid_segments:
    #                 #print(f"segment: {segment}") # segment: [3516, 3525]
    #                 start_frame = max(0, segment[0] * SR - file_start)
    #                 end_frame = min(file_length, (segment[-1] + 1) * SR - file_start)

    #                 # 🔹 유효한 segment만 추가
    #                 if start_frame < file_length and end_frame > start_frame:
    #                     #print(f"start_frame: {start_frame}") # 56256000
    #                    #print(f"end_frame: {end_frame}") # end_frame: 56400000
    #                     segment_audio.append(audio_data[start_frame:end_frame])

    #             #print(f"segment_audio: {segment_audio}")
    #             # 🔹 segment가 비어있지 않다면 리스트에 추가
    #             if segment_audio:
    #                 segment_audio = np.concatenate(segment_audio, axis=0)
    #                 #print(f"segment_audio.shape: {segment_audio.shape}") # 160000
    #                 #print(f"indices: {indices}") # [0]
    #                 data_list.append(segment_audio)
    #                 indices.append(indices[-1] + segment_audio.shape[0])

    #         except Exception as e:
    #             print(f"Error loading {file_path}: {e}")
    #             continue

    #     # print(f"data_list shapes: {[d.shape for d in data_list]}")
    #     #print(f'data_list: {data_list}')
    #     #print(f"indices: {indices}")

    #     return data_list, np.array(indices)


    # def _load_data(self, valid_segments):
    #     """
    #     특정 segment에 해당하는 오디오 데이터만 로드하는 함수.

    #     Args:
    #         valid_segments (list[list[int]]): 선택된 segment의 시작 및 끝 프레임 리스트.

    #     Returns:
    #         tuple: (오디오 데이터 리스트, 프레임 인덱스 리스트)
    #     """
        
    #     if not os.path.isdir(self.rec_path):  # 🔹 폴더인지 확인
    #         raise ValueError(f"Expected a directory but got a file: {self.rec_path}")

    #     file_list = sorted(
    #         [os.path.join(self.rec_path, x) for x in os.listdir(self.rec_path) if x.endswith(AUDIO_EXT)],
    #         key=lambda x: x.split('/')[-1]
    #     )

    #     print(f"file_list: {file_list}")
    #     '''
    #     file_list: ['/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0001.npy', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0002.npy', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0003.npy', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0004.npy', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0005.npy', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0006.npy', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0007.npy', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0008.npy', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0009.npy', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0010.npy', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0011.npy']
    #     '''
    #     data_list = []
    #     indices = [0]

    #     # 🚨 valid_segments에 해당하는 오디오만 로드
    #     for file_path in file_list:
    #         try:
    #             audio_data = np.load(file_path, mmap_mode='r', allow_pickle=True)
    #             #print(f"file_path: {file_path}") # /media/backup_SSD/ASPED_v2_npy/Session_11282023/FifthSt_F/Audio/recorder2_DR-05X-06/0005.npy
    #             # ✅ tuple로 저장된 경우 첫 번째 값만 가져오기
    #             if isinstance(audio_data, tuple):
    #                 audio_data = audio_data[0]
    #             #print(f"audio_data: {audio_data}") # [-0.0086803  -0.01495952 -0.0150315  ... -0.00076886 -0.0003035  0.00012606]
    #             #print(f"len(audio_data): {len(audio_data)}") # 230400000 (14400 seconds == 4 hours)
    #             # ✅ segment에 해당하는 프레임만 필터링
    #             segment_audio = []
    #             print(f"segment_audio (1): {segment_audio}") #[]
    #             print(f"valid_segments: {valid_segments}")
    #             for segment in valid_segments:
    #                 start_frame, end_frame = segment[0] * SR, segment[-1] * SR  # 초 단위 → 샘플 단위 변환
    #                 print(f"start_frame: {start_frame}, end_frame: {end_frame}")
    #                 print(f"len_")
    #                 print(f"len(audio_data[start_frame:end_frame]): {len(audio_data[start_frame:end_frame])}")
    #                 '''
    #                 valid_segments: [[89434, 89435, 89436, 89437, 89438, 89439, 89440, 89441, 89442, 89443]]
    #                 start_frame: 1430944000, end_frame: 1431088000
    #                 '''
    #                 print(f"audio_data: {audio_data}")
    #                 print(f"audio_data.shape: {audio_data.shape}") # audio_data.shape: (230400000,)
    #                 segment_audio.append(audio_data[start_frame:end_frame])
    #                 print(f"segment_audio (2): {segment_audio}")
                    
    #             print(f"segment_audio: {segment_audio}")
    #             # ✅ segment_audio를 합쳐서 저장
    #             if segment_audio:
    #                 segment_audio = np.concatenate(segment_audio, axis=0)
    #                 print(f"segment_audio (3): {segment_audio}")
    #                 data_list.append(segment_audio)
    #                 print(f"data_list: {data_list}")
    #                 print(f'indices (1): {indices}') # [0, 144000]
    #                 indices.append(indices[-1] + segment_audio.shape[0])
    #                 print(f"indices (2): {indices}") # [0, 144000, 288000]

    #         except Exception as e:
    #             print(f"Error loading {file_path}: {e}")
    #             continue

    #     print(f"data_list shapes: {[d.shape for d in data_list]}")
    #     print(f"indices: {indices}")
    #     '''
    #     data_list shapes: [(144000,), (144000,), (144000,), (144000,), (144000,), (144000,), (144000,), (144000,), (144000,), (144000,), (144000,)]
    #     indices: [0, 144000, 288000, 432000, 576000, 720000, 864000, 1008000, 1152000, 1296000, 1440000, 1584000]
    #     '''
    #     raise 'STOP'
    #     return data_list, np.array(indices)

    # # ORIGINAL (v.a)
    # def _load_data(self):
    #     # Assuming each file contains data that can be loaded into a NumPy array
    #     # Modify this function based on your data loading logic
    #     file_list = sorted([os.path.join(self.rec_path, x) for x in os.listdir(self.rec_path) if x.endswith(AUDIO_EXT)], key=lambda x: x.split('/')[-1])
    #     #print(f"file_list: {file_list}")
    #     data_list = [np.load(file_path, mmap_mode='r', allow_pickle=True) for file_path in file_list]
    #     #print(f"data_list: {data_list}")
    #     data_list = [d[0] if isinstance(d, tuple) else d for d in data_list if d is not None]  # Unpack tuple

    #     indices = np.cumsum([0] + [x.shape[0] for x in data_list])
    #     #print(f"data_list(2): {data_list}")
    #     #print(f"indices: {indices}\n")
    #     #raise 'STOP'
    #     return (data_list, indices)
     
    @staticmethod
    def from_dirs(dirs: list[str], radius: Literal[1,3,6,9] = 6, segment_length = 1,
                  transform :Literal['vggish', 'vggish-mel', 'ast', None] = None, n_classes= 2) -> ConcatDataset:
        
        if transform == 'vggish':
            ASPEDDataset._transform = VGGish()
        elif transform == 'vggish-mel':
            ASPEDDataset._transform = VGGish_PreProc()
        elif transform == 'ast':
            ASPEDDataset._transform = torch.nn.Identity()
        
        ASPEDDataset.n_classes = n_classes

        dataset = []
        
        for d in tqdm(dirs):
            with open(os.path.join(d, METADATA_PATH), 'r') as f:
                metadata = json.load(f)
            for cam, recs in metadata.items():
                try:
                    labels = pd.read_csv(*[os.path.join(d, 'Labels', x) for x in 
                                        os.listdir(os.path.join(d, 'Labels')) if x.endswith(LABELS_SUFFIX.format(cam))])
                except:
                    continue
                
                for i in range(1, len(recs) + 1):
                    working_dir = os.path.join(d, AUDIO_PREFIX.format(recs[i - 1]))

                    label = torch.Tensor(labels[LABEL_HEADER.format(i, radius)].values)
                    views = torch.Tensor(labels[VIEW_PREFIX+LABEL_HEADER.format(i, radius)].values)
                    label[views == 1] = IGNORE_INDEX
                    ASPEDDataset.max_val = max(label.max().item(), ASPEDDataset.max_val)
                    dataset.append(ASPEDDataset(working_dir, label, segment_length=segment_length, n_classes=n_classes))
        
        return ConcatDataset(dataset)
    
    # V.b (New)
    @staticmethod
    def from_dirs_v1(dirs: list[str], radius: Literal[1,3,6,9] = 6, segment_length = 10,
                    transform: Literal['vggish', 'vggish-mel', 'ast', None] = None, n_classes=2) -> ConcatDataset:
        
        ASPEDDataset.n_classes = n_classes
        dataset = []

        print(f"dirs: {dirs}")

        for d in tqdm(dirs):
            for s in sorted(os.listdir(d)):
                path = os.path.join(d, s)
                new_path = path.replace("ASPED_v2_npy", "ASPED_v2_npy_labels_with_bus").replace('/Labels', '/')

                try:
                    # ✅ Labels 폴더에서 모든 CSV 파일 로드
                    label_files = sorted([os.path.join(new_path, x) for x in os.listdir(new_path) if x.endswith(LABELS_SUFFIX[-3:])])
                    labels = pd.concat([pd.read_csv(x) for x in label_files], axis=0)

                    # 🚨 버스 프레임 제외 (`busFrame == 0`인 프레임만 유지)
                    valid_frames = labels[labels['busFrame'] == 0]['frame'].tolist()

                    # ✅ valid_segments를 파일별 `dict`로 변환
                    valid_segments = {}
                    current_segment = []

                    for frame in valid_frames:
                        if not current_segment or frame == current_segment[-1] + 1:
                            current_segment.append(frame)
                        else:
                            if len(current_segment) >= segment_length:
                                file_index = current_segment[0] // (4 * 3600)  # 4시간 단위 분할
                                if file_index not in valid_segments:
                                    valid_segments[file_index] = []
                                valid_segments[file_index].append(current_segment)
                            current_segment = [frame]

                    if len(current_segment) >= segment_length:
                        file_index = current_segment[0] // (4 * 3600)
                        if file_index not in valid_segments:
                            valid_segments[file_index] = []
                        valid_segments[file_index].append(current_segment)

                except Exception as e:
                    print(f"❌ Error reading labels in {new_path}: {e}")
                    continue

                audio_path = os.path.join(path, 'Audio')

                for recorder in sorted(os.listdir(audio_path)):
                    recorder_path = os.path.join(audio_path, recorder)
                    if not os.path.isdir(recorder_path):
                        continue  # 폴더가 아닌 경우 무시

                    recorder_id = recorder.split('_')[0][-1]
                    label_column = LABEL_HEADER.format(recorder_id, radius)

                    if label_column not in labels.columns:
                        print(f"⚠️ Warning: {label_column} not found in labels for {recorder_path}")
                        continue

                    label_values = torch.tensor(labels[label_column].values, dtype=torch.float32)

                    # ✅ 개별 .npy 파일을 기준으로 valid_segments를 다시 나누기
                    file_list = sorted([os.path.join(recorder_path, x) for x in os.listdir(recorder_path) if x.endswith(AUDIO_EXT)])

                    file_offsets = {}
                    offset = 0
                    for file_path in file_list:
                        try:
                            audio_data = np.load(file_path, mmap_mode='r', allow_pickle=True)
                            file_length = len(audio_data)
                            file_offsets[file_path] = (offset, offset + file_length)
                            offset += file_length
                        except Exception as e:
                            print(f"❌ Error loading {file_path}: {e}")
                            continue

                    # 🚀 valid_segments를 개별 파일 기준으로 변환
                    for file_path, (file_start, file_end) in file_offsets.items():
                        file_index = int(file_path[-8:-4]) - 1  # 파일 번호 (0001 → 0, 0002 → 1)

                        if file_index not in valid_segments:
                            continue  # 현재 파일과 관련된 valid_segments 없음

                        file_valid_segments = []
                        for seg in valid_segments[file_index]:
                            seg_start, seg_end = seg[0] * SR, seg[-1] * SR
                            if seg_start >= file_end or seg_end <= file_start:
                                continue

                            local_start = max(0, seg_start - file_start)
                            local_end = min(file_end - file_start, seg_end - file_start)

                            if local_end > local_start:
                                file_valid_segments.append((local_start, local_end))

                        if file_valid_segments:
                            local_label = label_values[seg[0]:seg[0] + segment_length]
                            dataset.append(ASPEDDataset(
                                os.path.dirname(file_path),
                                local_values,
                                segment_length=segment_length,
                                n_classes=n_classes,
                                valid_segments={file_path: file_valid_segments}  # ✅ dict로 변환된 valid_segments 사용
                            ))

        # 🚨 dataset이 비어 있는 경우 예외 처리
        if not dataset:
            raise ValueError("❌ No valid datasets found. Check if audio files and corresponding labels exist.")

        print(f"type(dataset): {type(dataset)}")  # list <class 'list'>
        print(f"len(dataset): {len(dataset)}")

        # 🚨 개별 데이터셋 타입과 길이 확인
        for i, d in enumerate(dataset):
            print(f"Dataset {i}: type={type(d)}, len={len(d) if hasattr(d, '__len__') else 'No __len__ method'}")
            
            # 🚨 문제 있는 데이터셋 찾기
            if hasattr(d, '__len__'):
                length = len(d)
                if length < 0:
                    print(f"⚠️  Warning: Dataset {i} has len={length}, which is invalid!")
            else:
                print(f"⚠️  Warning: Dataset {i} does not have a __len__() method!")
                
        c_dataset = ConcatDataset(dataset)
        print(f"c_dataset[0]: {c_dataset[0]}")
        print(f"c_dataset[1]: {c_dataset[1]}")
        
        raise 'STOP'

        # 🚨 Transform 적용
        if transform == 'vggish':
            ASPEDDataset._transform = VGGish()
        elif transform == 'vggish-mel':
            ASPEDDataset._transform = VGGish_PreProc()
        elif transform == 'ast':
            ASPEDDataset._transform = AST_PreProc()
        else:
            ASPEDDataset._transform = torch.nn.Identity()

        return c_dataset

    
    
    # V.b
    # @staticmethod
    # def from_dirs_v1(dirs: list[str], radius: Literal[1,3,6,9] = 6, segment_length = 10,
    #                 transform: Literal['vggish', 'vggish-mel', 'ast', None] = None, n_classes=2) -> ConcatDataset:

    #     ASPEDDataset.n_classes = n_classes
    #     dataset = []

    #     print(f"dirs: {dirs}")
    #     '''
    #     dirs: ['/media/backup_SSD/ASPED_v2_npy/Session_07262023', '/media/backup_SSD/ASPED_v2_npy/Session_08092023', 
    #     '/media/backup_SSD/ASPED_v2_npy/Session_11072023', '/media/backup_SSD/ASPED_v2_npy/Session_11282023']
    #     '''

    #     for d in tqdm(dirs):
    #         for s in sorted(os.listdir(d)):
    #             path = os.path.join(d, s)
    #             new_path = path.replace("ASPED_v2_npy", "ASPED_v2_npy_labels_with_bus").replace('/Labels', '/')
    #             # print(f'new_path: {new_path}') #  /media/backup_SSD/ASPED_v2_npy_labels_with_bus/Session_07262023/FifthSt_A
    #             # print(sorted(os.listdir(d))) # ['FifthSt_A', 'FifthSt_B', 'FifthSt_C', 'FifthSt_D', 'FifthSt_E', 'FifthSt_F']
    #             try:
    #                 # Labels 폴더에서 모든 CSV 파일 로드
    #                 label_files = list(filter(lambda x: x.endswith(LABELS_SUFFIX[-3:]),
    #                                         [os.path.join(new_path, x) for x in sorted(os.listdir(new_path))]))
    #                 # print(f"sorted(label_files): {sorted(label_files)}")
    #                 '''
    #                 sorted(label_files): ['/media/backup_SSD/ASPED_v2_npy_labels_with_bus/Session_07262023/FifthSt_A/0001.csv', 
    #                 '/media/backup_SSD/ASPED_v2_npy_labels_with_bus/Session_07262023/FifthSt_A/0002.csv', ... ,
    #                 '/media/backup_SSD/ASPED_v2_npy_labels_with_bus/Session_07262023/FifthSt_A/0010.csv']
    #                 '''
    #                 labels = [pd.read_csv(x) for x in sorted(label_files)]
    #                 labels = pd.concat(labels, axis=0)
                    
    #                 # # DEBUG
    #                 # output_txt_path = "/media/backup_SSD/ASPED_v2_npy_labels_with_bus/labels_output.txt"

    #                 # with open(output_txt_path, "a") as f:  # "a" (append) 모드로 기존 파일에 추가 저장
    #                 #     f.write(f"Labels for {new_path}:\n")
    #                 #     labels.to_csv(f, sep="\t", index=False)  # 탭 구분자로 저장

                
    #                 # 버스 프레임이 있는 경우 제외한 세그먼트 찾기
    #                 valid_frames = labels[labels['busFrame'] == 0]['frame'].tolist() # 'frame column'    
    #                 # # DEBUG
    #                 # output_txt_path = "/media/backup_SSD/ASPED_v2_npy_labels_with_bus/valid_frames_output.txt"
    #                 # with open(output_txt_path, "a") as f:  # "a" (append) 모드로 저장
    #                 #     f.write(f"Valid frames for {new_path}:\n")
    #                 #     f.write(", ".join(map(str, valid_frames)))  # 리스트를 쉼표로 구분된 문자열로 변환
    #                 #     f.write("\n" + "=" * 50 + "\n")  # 구분선 추가
    #                 # raise "STOP"
                
    #                 # valid_frames 리스트에서 연속된 값들을 valid_segments로 그룹화
    #                 valid_segments = []
    #                 current_segment = []
    #                 for frame in valid_frames:
    #                     if not current_segment or frame == current_segment[-1] + 1:
    #                         current_segment.append(frame)
    #                     else:
    #                         # 현재 current_segment 길이가 segment_length 이상이어야지만 valid_segments에 추가
    #                         if len(current_segment) >= segment_length:
    #                             valid_segments.append(current_segment)
    #                         # 새로운 current_segment 시작
    #                         current_segment = [frame]
    #                 # for 루프가 끝난 후 마지막 그룹이 segment_length 이상이면 valid_segments에 추가.
    #                 if len(current_segment) >= segment_length:
    #                     valid_segments.append(current_segment)
    #                 output_txt_path = "/media/backup_SSD/ASPED_v2_npy_labels_with_bus/valid_segments_output.txt"

    #                 # # DEBUG
    #                 # with open(output_txt_path, "a") as f:  # "a" (append) 모드로 저장
    #                 #     f.write(f"Valid segments for {new_path}:\n")
    #                 #     for segment in valid_segments:
    #                 #         f.write(f"{segment}\n")  # 리스트 그대로 저장
    #                 #     f.write("=" * 50 + "\n")  # 구분선 추가
    #                 # raise 'stop'
                
    #                 #print(f"segment_length: {segment_length}") # 10

    #             except Exception as e:
    #                 print(f"Error reading labels in {new_path}: {e}")
    #                 input("Press any key to continue...")
    #                 continue

    #             audio_path = os.path.join(path, 'Audio')
    #             # print(F"audio_path: {audio_path}") # /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio

    #             # 각 레코더별로 처리
    #             # print(f"os.listdir(audio_path): {os.listdir(audio_path)}") # ['recorder1_DR-05X-01']
    #             for recorder in sorted(os.listdir(audio_path)):
    #                 # print(f"recorder: {recorder}") # recorder1_DR-05X-01
    #                 recorder_path = os.path.join(audio_path, recorder)
    #                 # print(f"recorder_path: {recorder_path}")
    #                 # recorder_path: /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01
                    
    #                 if not os.path.isdir(recorder_path):
    #                     continue  # 폴더가 아닌 경우 무시

    #                 # 리코더 ID 추출 (예: "recorder1_DR-05X-01" → "recorder1" --> "1")
    #                 recorder_id = recorder.split('_')[0][-1]
    #                 # print(f"recorder_id: {recorder_id}") # 1
                    
    #                 label_column = LABEL_HEADER.format(recorder_id, radius)
    #                 # print(f"label_column: {label_column}") # recorder1_6m

    #                 # print(f"label.columns: {labels.columns}")
    #                 '''
    #                 label.columns: Index(['timestamp', 'frame', 'recorder1_1m', 'recorder1_3m', 'recorder1_6m',
    #                             'recorder1_9m', 'view_recorder1_1m', 'view_recorder1_3m',
    #                             'view_recorder1_6m', 'view_recorder1_9m', 'busFrame'],
    #                             dtype='object')
    #                 '''
    #                 if label_column not in labels.columns:
    #                     print(f"Warning: {label_column} not found in labels for {recorder_path}")
    #                     continue

    #                 label_values = labels[label_column].values
    #                 # print(f"label_values: {label_values}") # label_values: [0 0 0 ... 0 0 0] # recorder1_6m

    #                 # 🚨 개별 .npy 파일을 기준으로 valid_segments를 다시 나누기
    #                 file_list = sorted([os.path.join(recorder_path, x) for x in os.listdir(recorder_path) if x.endswith(AUDIO_EXT)])
    #                 # print(f"file_list: {file_list}")
    #                 '''
    #                 file_list: ['/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0001.npy', 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0002.npy', 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0003.npy', 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0004.npy', 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0005.npy', 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0006.npy', 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0007.npy', 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0008.npy', 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0009.npy', 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0010.npy']
    #                 '''

    #                 file_offsets = {}  # 각 파일의 오디오 오프셋 추적
    #                 offset = 0

    #                 for file_path in file_list:
    #                     file_index = int(str(file_path)[-8:-4])
    #                     print(f"** file_path: {file_path}")
    #                     #input()
    #                     try:
    #                         audio_data = np.load(file_path, mmap_mode='r', allow_pickle=True)
    #                         if isinstance(audio_data, tuple):
    #                             audio_data = audio_data[0]

    #                         file_length = len(audio_data)
    #                         file_offsets[file_path] = (offset, offset + file_length)  # 오프셋 저장
    #                         offset += file_length
                            
    #                         # print(f"offset: {offset}")

    #                     except Exception as e:
    #                         print(f"Error loading {file_path}: {e}")
    #                         continue
    #                 # print(f"file_offsets: {file_offsets}")
    #                 '''
    #                 # 230400000 samples == 14400 seconds (16kHz) == 240 minutes == 4 hours
    #                 file_offsets: {'/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0001.npy': (0, 230400000), 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0002.npy': (230400000, 460800000), 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0003.npy': (460800000, 691200000), 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0004.npy': (691200000, 921600000), 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0005.npy': (921600000, 1152000000), 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0006.npy': (1152000000, 1382400000), 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0007.npy': (1382400000, 1612800000), 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0008.npy': (1612800000, 1843200000), 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0009.npy': (1843200000, 2073600000), 
    #                 '/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0010.npy': (2073600000, 2304000000)}
    #                 '''
                
    #                 # 🚨 valid_segments를 개별 파일 기준으로 변환
    #                 for seg in valid_segments:
    #                     # print(f"seg: {seg}") # [0, 1, ..., 910]
    #                     # print(f"type(seg): {type(seg)}") # <class 'list'>
    #                     seg_start, seg_end = seg[0] * SR, seg[-1] * SR # sample 단위로 변경; 시작 샘플과 끝 샘플
    #                     # print(f"seg_start: {seg_start}") # 0
    #                     # print(f"seg_end: {seg_end}") # 14560000
                        
    #                     # print(f"file_offsets.items(): {file_offsets.items()}")
    #                     '''
    #                     file_offsets.items(): 
    #                     dict_items([('/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0001.npy', (0, 230400000)), 
    #                     ('/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0002.npy', (230400000, 460800000)), 
    #                     ('/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0003.npy', (460800000, 691200000)), 
    #                     ('/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0004.npy', (691200000, 921600000)), 
    #                     ('/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0005.npy', (921600000, 1152000000)), 
    #                     ('/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0006.npy', (1152000000, 1382400000)), 
    #                     ('/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0007.npy', (1382400000, 1612800000)), 
    #                     ('/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0008.npy', (1612800000, 1843200000)), 
    #                     ('/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0009.npy', (1843200000, 2073600000)), 
    #                     ('/media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_A/Audio/recorder1_DR-05X-01/0010.npy', (2073600000, 2304000000))])
    #                     '''
                        
    #                     for file_path, (file_start, file_end) in file_offsets.items():
    #                         if seg_start >= file_end or seg_end <= file_start:
    #                             continue  # 현재 파일 범위를 벗어나는 segment는 무시

    #                         # print(f"seg_start: {seg_start}") # 0
    #                         # print(f"seg_end: {seg_end}") # 14560000
                            
    #                         # 파일 내 상대적 프레임 계산
    #                         local_start = max(0, seg_start - file_start)
    #                         local_end = min(file_end - file_start, seg_end - file_start)
                            
    #                         # print(f"local_start: {local_start}") # 0
    #                         # print(f"local_end: {local_end}") # 14560000

    #                         # print(f"seg[0]: {seg[0]}") # 0
    #                         # print(f"seg[0] + segment_length: {seg[0]+segment_length}") # 10
    #                         if local_end > local_start:
    #                             local_label = torch.tensor(label_values[seg[0]:seg[0] + segment_length].astype(np.float32))
    #                             # print(f"local_label: {local_label}") # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    #                             # print(f"local_label.numel(): {local_label.numel()}") # 10
    #                             if local_label.numel() != 10:
    #                                 print(f"Warning: Empty label tensor for {file_path} and segment {seg}")
    #                                 raise 'stop'
    #                                 continue
    #                             #print (f"local_label.max().item(): {local_label.max()}") 
    #                             #print(f"ASPEDDataset.max_val: {ASPEDDataset.max_val}") # 1 (Default; minimum)
                                
    #                             ASPEDDataset.max_val = max(local_label.max().item(), ASPEDDataset.max_val)
    #                             #print(F'ASPEDDataset.max_val: {ASPEDDataset.max_val}') # ASPEDDataset.meax_val: 1.0

    #                             # 🔹 파일이 아닌, 해당 파일이 있는 폴더를 전달
    #                             # __init__ 
    #                             # print(f"dataset: {dataset}") # initial value is []
    #                             dataset.append(ASPEDDataset(
    #                                 os.path.dirname(file_path),  # 🔹 폴더 경로를 전달
    #                                 local_label,
    #                                 segment_length=segment_length,
    #                                 n_classes=n_classes,
    #                                 valid_segments=[[(file_start)//230400000 + local_start//SR, (file_start)//230400000 + local_end // SR]]
    #                             ))
    #                             # # 📌 valid_segments를 전달하여 ASPEDDataset 객체 생성
    #                             # dataset.append(ASPEDDataset(
    #                             #     file_path, local_label, segment_length=segment_length, n_classes=n_classes, valid_segments=[[local_start // SR, local_end // SR]]
    #                             # ))
    #                             # print(f"dataset: {dataset}") 
    #                             # dataset: [<models.data_utils.dataset.ASPEDDataset object at 0x7c5e7a47c2b0>]
    #                             # raise 'stop'
    #                         # print(F'ASPEDDataset.max_val: {ASPEDDataset.max_val}')
    #                         #raise 'stop'

    #     # 🚨 dataset이 비어 있는 경우 예외 처리
    #     if not dataset:
    #         raise ValueError("❌ No valid datasets found. Check if audio files and corresponding labels exist.")

    #     print(f"type(dataset): {type(dataset)}") # list <class 'list'>
    #     print(f"len(dataset): {len(dataset)}") # 12568
        
    #     # 🚨 개별 데이터셋 타입과 길이 확인
    #     for i, d in enumerate(dataset):
    #         print(f"Dataset {i}: type={type(d)}, len={len(d) if hasattr(d, '__len__') else 'No __len__ method'}")
            
    #         # 🚨 문제 있는 데이터셋 찾기
    #         if hasattr(d, '__len__'):
    #             length = len(d)
    #             if length < 0:
    #                 print(f"⚠️  Warning: Dataset {i} has len={length}, which is invalid!")
    #         else:
    #             print(f"⚠️  Warning: Dataset {i} does not have a __len__() method!")
                
    #     c_dataset = ConcatDataset(dataset)
    #     print(f"c_dataset[0]: {c_dataset[0]}")
    #     print(f"c_dataset[1]: {c_dataset[1]}")
        
    #     raise 'STOP'

    #     # 🚨 Transform 적용
    #     if transform == 'vggish':
    #         ASPEDDataset._transform = VGGish()
    #     elif transform == 'vggish-mel':
    #         ASPEDDataset._transform = VGGish_PreProc()
    #     elif transform == 'ast':
    #         ASPEDDataset._transform = AST_PreProc()
    #     else:
    #         ASPEDDataset._transform = torch.nn.Identity()

    #     return c_dataset

    # # V.A
    # @staticmethod
    # def from_dirs_v1(dirs: list[str], radius: Literal[1,3,6,9] = 6, segment_length = 10,
    #               transform :Literal['vggish', 'vggish-mel', 'ast', None] = None, n_classes= 2) -> ConcatDataset:
        
    #     ASPEDDataset.n_classes = n_classes
    #     dataset = []
        
    #     print(f"dirs: {dirs}")
    #     for d in tqdm(dirs):
    #         for s in os.listdir(d):
    #             path = os.path.join(d,s)
    #             try:
    #                 label_files = list(filter(lambda x: x.endswith(LABELS_SUFFIX[-3:]) and not
    #                                     'processed' in x, [os.path.join(path,'Labels', x) for x in os.listdir(os.path.join(path, 'Labels'))]))
    #                 labels = [pd.read_csv(x) for x in sorted(label_files)]
    #                 labels = pd.concat(labels, axis=0)
    #             except Exception as e:
    #                 print(e)
    #                 continue
    #             audio_path = os.path.join(path, 'Audio')
                
    #             for i, rec in enumerate(sorted(os.listdir(audio_path))):
    #                 label = torch.Tensor(labels[LABEL_HEADER.format(i + 1, radius)].values)
    #                 ASPEDDataset.max_val = max(label.max().item(), ASPEDDataset.max_val)
    #                 working_dir = os.path.join(audio_path, rec)
    #                 '''
    #                 working_dir: /media/backup_SSD/ASPED_v2_npy/Session_07262023/FifthSt_C/Audio/recorder1_DR-05X-11
    #                 label: tensor([0., 0., 0.,  ..., 0., 0., 0.])
    #                 '''
    #                 #raise "STOP"
    #                 dataset.append(ASPEDDataset(working_dir, label, segment_length=segment_length, n_classes=n_classes))
                    
    #     #print(f"dataset: {dataset}")
    #     '''
    #     dataset: [<models.data_utils.dataset.ASPEDDataset object at 0x7a4412ebb5e0>, <models.data_utils.dataset.ASPEDDataset object at 0x7a4412e43f10>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a4412ebb3a0>, <models.data_utils.dataset.ASPEDDataset object at 0x7a4412ebb6a0>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a4407f36580>, <models.data_utils.dataset.ASPEDDataset object at 0x7a4412eb3f10>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a4412ebbb20>, <models.data_utils.dataset.ASPEDDataset object at 0x7a440eeaf430>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a44125fe490>, <models.data_utils.dataset.ASPEDDataset object at 0x7a4407f36520>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a4412ebb160>, <models.data_utils.dataset.ASPEDDataset object at 0x7a44125fe760>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a440eeafee0>, <models.data_utils.dataset.ASPEDDataset object at 0x7a4407f36190>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a4412e43610>, <models.data_utils.dataset.ASPEDDataset object at 0x7a440ccde850>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a4412e436a0>, <models.data_utils.dataset.ASPEDDataset object at 0x7a440eeb60d0>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a440d2cd1f0>, <models.data_utils.dataset.ASPEDDataset object at 0x7a440dde6b80>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a440ccde3d0>, <models.data_utils.dataset.ASPEDDataset object at 0x7a440ccdeca0>, 
    #     <models.data_utils.dataset.ASPEDDataset object at 0x7a4411bae940>, <models.data_utils.dataset.ASPEDDataset object at 0x7a4411bae1c0>]
    #     '''
    #     #raise "STOP"
    #     c_dataset = ConcatDataset(dataset)

    #     if transform == 'vggish':
    #         ASPEDDataset._transform = VGGish()
    #     elif transform == 'vggish-mel':
    #         ASPEDDataset._transform = VGGish_PreProc()
    #     elif transform == 'ast':
    #         ASPEDDataset._transform = AST_PreProc()
    #     else:
    #         ASPEDDataset._transform = torch.nn.Identity()
    #     return c_dataset

if __name__ == '__main__':
    # Testing 
    import time
    from random import randint

    TEST_DIR = ["/media/chan/backup_SSD2/ASPED_v1_npy/Session_5242023", "/media/chan/backup_SSD2/ASPED_v1_npy/Session_6012023", 
                "/media/chan/backup_SSD2/ASPED_v1_npy/Session_6072023", "/media/chan/backup_SSD2/ASPED_v1_npy/Session_6212023",
                "/media/chan/backup_SSD2/ASPED_v1_npy/Session_6282023"]
    
    X = ASPEDDataset.from_dirs_v1(TEST_DIR, segment_length=10, transform='vggish-mel')
    print(type(X), len(X))

    lim = 100

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