import torch
from torch.utils.data import Dataset
import numpy as np

class DICOMSliceDataset(Dataset):
    def __init__(self, data_dict, segmentation_dict, transform=None):
        self.slices = []
        self.masks = []
        for patient_id, data in data_dict.items():
            if patient_id not in segmentation_dict:
                raise KeyError(f"No segmentation for {patient_id}")
            num_slices = data.shape[0]
            for slice_idx in range(num_slices):
                self.slices.append(data[slice_idx, :, :])
                self.masks.append(segmentation_dict[patient_id][slice_idx, :, :])

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_img = self.slices[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32)

        # Конвертация в тензоры
        slice_tensor = torch.from_numpy(slice_img).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return slice_tensor, mask_tensor