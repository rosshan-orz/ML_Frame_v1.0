from torch.utils.data import Dataset
import os, torch
import numpy as np

class EEGDataset_DTU(Dataset):
    def __init__(self, root, file_name, mode='none'):
        self.file_path = os.path.join(root, file_name)
        self.data = np.load(self.file_path, allow_pickle=True)

        self.eeg_data = self.data['eeg_slices']
        self.audioA_data = self.data['wavA_slices']
        self.audioB_data = self.data['wavB_slices']
        self.event_data = self.data['event_slices']
        self.mode = mode
                
    def __len__(self):
        return len(self.eeg_data)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        eeg = self.eeg_data[idx].astype(np.float32)[:64,:]
        eeg = torch.Tensor(eeg)
        # 归一化处理
        eeg_min,_ = torch.min(eeg, dim=1, keepdim=True)
        eeg_max,_ = torch.max(eeg, dim=1, keepdim=True)
        eeg = (eeg-eeg_min)/(eeg_max-eeg_min)

        event_np = self.event_data[idx]
        event_values = []
        for item in event_np:
            event_values.append(item[0][0][0][0])
        values_array = np.array(event_values, dtype=np.uint8)
        event = torch.tensor(values_array)
        event = event-1
        
        return eeg, event
    
class EEGDataset_KUL(Dataset):
    def __init__(self, root, file_name):
        self.file_path = os.path.join(root, file_name)
        self.data = np.load(self.file_path)
        
        self.eeg_data = self.data['eeg']
        self.audio_data = self.data['audio']
        self.ear_data = self.data['ear']

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        eeg = self.eeg_data[idx].astype(np.float32)
        eeg = torch.Tensor(eeg)
        # 归一化处理
        eeg_min,_ = torch.min(eeg, dim=1, keepdim=True)
        eeg_max,_ = torch.max(eeg, dim=1, keepdim=True)
        eeg = (eeg-eeg_min)/(eeg_max-eeg_min)
        
        event = self.ear_data[idx]
        event = torch.Tensor(event)
        
        return eeg, event