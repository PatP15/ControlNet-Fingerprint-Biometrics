import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class FingerprintDataset(Dataset):
    def is_image_filename(self, filename):
        return any(extension in filename for extension in ['.png', \
            '.PNG', '.pneg', '.PNEG', '.jpg', '.JPG', '.jpeg', '.JPEG'])

    """
    Loads the unpaired images from the folder
    1) Stores indexed unpaired images with their corresponding labels in self.images, self.img_labels
    2) Stores all the possible classes in self.classes
    3) Stores images separated by class in self.class_to_images (has image filepath, not actual image)
    """
    def load_images(self, root_dir):
        self.classes = list() # self.classes[i] = the name of class with index i
        self.img_labels = list()
        self.images = list()
        self.class_to_images = list()

        for pid in tqdm(os.listdir(root_dir)):
            self.classes.append(pid)
            self.class_to_images.append(list())
            curr_person_folder = os.path.join(root_dir, pid)
            for sample in os.listdir(curr_person_folder):
                if not self.is_image_filename(sample):
                    continue
                curr_image = os.path.join(curr_person_folder, sample)

                self.img_labels.append(pid)
                self.images.append(curr_image)
                self.class_to_images[-1].append(curr_image)

        self.len = len(self.img_labels)

        return

    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train

        self.load_images(root_dir)

        if self.train:
            self.train_labels = self.img_labels
            self.train_data = self.images
        else:
            self.test_labels = self.img_labels
            self.test_data = self.images

        return

    def __len__(self):
        return self.len

    # returns image, label, filepath
    def __getitem__(self, idx):
        raise RuntimeError('not supposed to call getitem() from fingerprint dataset')