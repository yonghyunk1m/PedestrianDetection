import numpy as np
import os
import shutil
from tqdm import tqdm
import torchaudio
import torchaudio.functional as F
import torch
import sys

'''
Script to convert the audio from wav to 16Khz downsampled, 1-channel (mono) numpy arrays
for easy dataloading (memmap, etc.)

structure of source directory will be replicated in the destination directory

Use: python convert_to_npy.py [source directory of audio data for session] [destination directory of numpy arrays]

    OR update SOURCE_PATH, DESTINATION_PATH below 

    update AUDIO_EXT depending on audio file extension

Dependencies: torch, torchaudio, tqdm
'''

AUDIO_EXT = '.flac' 


if __name__ == '__main__':
    # Modify the list, SOURCE_PATH, DESTINATION_PATH as you wanted.
    for session in ["/Session_08092023","/Session_11072023","/Session_11282023"]:
        SOURCE_PATH = '/media/chan/backup_SSD2/ASPED.b_Aligned' + session
        DESTINATION_PATH = '/media/chan/My Passport2/ASPED_v2_npy' + session
        if len(sys.argv) == 3:
            SOURCE_PATH, DESTINATION_PATH = tuple(sys.argv[1:])

        try:
            os.mkdir(DESTINATION_PATH)

        except FileExistsError as e:
            pass

        pbar = tqdm(os.walk(SOURCE_PATH), total=len(list(os.walk(SOURCE_PATH))))
        for root, dirs, files in pbar:
            prefix = os.path.relpath(root, SOURCE_PATH)
            for dir in dirs:
                os.makedirs(os.path.join(DESTINATION_PATH, prefix, dir), exist_ok=True)

            label_files = sorted([x for x in files if x.endswith(".csv")])
            audio_files = sorted([x for x in files if x.endswith(AUDIO_EXT) and x != 'bandpassed.wav'])
            
            if len(label_files) > 0:
                if "recorder" not in root:
                    for filename in label_files:
                        shutil.copyfile(
                            os.path.join(root, filename),
                            os.path.join(DESTINATION_PATH, prefix, filename)
                        )


            for idx, x in enumerate(audio_files):
                pbar.set_description(f'{round(idx / len(audio_files) * 100, 2)}%')
                
                waveform, sr = torchaudio.load(os.path.join(root, x))

                waveform = F.resample(waveform, orig_freq=sr, new_freq=16000)

                waveform = torch.mean(waveform, axis=0).numpy()

                waveform = waveform.astype(np.float32)

                with open(f'{os.path.join(DESTINATION_PATH, prefix, x[:-len(AUDIO_EXT)])}.npy', 'wb') as f:
                    np.save(f, waveform)
            
            