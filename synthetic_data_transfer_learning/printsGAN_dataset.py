import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class PrintsGANDataset(Dataset):
    def is_image_filename(self, filename):
        return any(extension in filename for extension in ['.png', '.PNG', '.pneg', '.PNEG', '.jpg', '.JPG', '.jpeg', '.JPEG'])

    def load_images(self, root_dir, max_labels, max_images_per_label):
        self.classes = []  # self.classes[i] = the name of class with index i
        self.img_labels = []
        self.images = []
        self.class_to_images = []

        labels_loaded = 0

        for pid in tqdm(os.listdir(root_dir)):
            # print(pid)
            if labels_loaded >= max_labels:
                break  # Stop if the max number of labels has been reached
            
            self.classes.append(pid)
            self.class_to_images.append([])
            curr_person_folder = os.path.join(root_dir, pid)
            
            images_loaded = 0
            for sample in os.listdir(curr_person_folder):
                if not self.is_image_filename(sample):
                    continue
                if images_loaded >= max_images_per_label:
                    break  # Stop if the max number of images per label has been reached

                curr_image = os.path.join(curr_person_folder, sample)
                self.img_labels.append(pid)
                self.images.append(curr_image)
                self.class_to_images[-1].append(curr_image)
                images_loaded += 1

            labels_loaded += 1

        self.len = len(self.img_labels)

        print(f"Total labels loaded: {len(self.classes)}")
        print(f"Total images loaded: {self.len}")

        return

    def __init__(self, root_dir,  max_labels, max_images_per_label, train=True):
        self.root_dir = root_dir
        self.train = train

        self.load_images(root_dir, max_labels=max_labels, max_images_per_label=max_images_per_label)

        if self.train:
            self.train_labels = self.img_labels
            self.train_data = self.images
        else:
            self.test_labels = self.img_labels
            self.test_data = self.images

        return

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        raise RuntimeError('not supposed to call getitem() from fingerprint dataset')
