import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset

class FingerDataset(Dataset):
    def __init__(self, data_file='./prompts.json'):
        self.data = []
        with open(data_file, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # Ensure the image file exists and is readable
        if not os.path.exists(source_filename) or not os.path.exists(target_filename):
            raise FileNotFoundError(f"One or more image files not found: {source_filename}, {target_filename}")

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Check if images are loaded
        if source is None or target is None:
            raise Exception(f"Failed to load one or more images: {source_filename}, {target_filename}")

        # Convert color from BGR to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize images
        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))

        # Normalize source images to [0, 1]
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1]
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
