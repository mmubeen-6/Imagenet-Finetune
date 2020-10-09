import os
from PIL import Image, ImageFile

import torch 
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):

    def __init__(self, root_dir, model_save_path, testing=False, transform=None):
        """
        Constructor for The Dataset we are using for Imagenet Finetune.
        :param root_dir: Base path of the directory/folder where data is placed.
        :param model_save_path: path where model will be or is saved
        :param testing: whether to load data for training or testing
        :param transform: torch.Transform variable to apply transformation.
        """
        self.root_dir = root_dir
        self.model_save_path = model_save_path
        self.transform = transform
        
        class_labels_file = os.path.join(self.model_save_path, "class_labels.txt")
        if not testing:
            self.class_labels = os.listdir(self.root_dir)
            self.class_labels.sort()
            self.num_classes = len(self.class_labels)

            with open(class_labels_file, "w") as file_:
                for class_label in self.class_labels:
                    file_.write("{}\n".format(class_label))
        else:
            with open(class_labels_file) as file_:
                content = file_.readlines()
                self.class_labels = [x.strip() for x in content]
        
        self.images_path = []
        self.images_labels = []

        for class_label in self.class_labels:
            for image_name in os.listdir(os.path.join(self.root_dir, class_label)):
                self.images_path.append(os.path.join(self.root_dir, class_label, image_name))
                self.images_labels.append(self.class_labels.index(class_label))
        
        self.images_path = self.images_path[0:500]
        self.images_labels = self.images_labels[0:500]

        # print(self.images_path)
        # print(self.images_labels)

    def get_total_classes(self):
        """
        :return: total classes in dataset
        """
        return self.num_classes

    def __len__(self):
        """
        :return: length of the training/valid sets.
        """
        return len(self.images_path)

    def __getitem__(self, index):
        """
        :param index: id of the sample example.
        :return: tuple containing the torch image and the torch index of the class.
        """

        image = Image.open(self.images_path[index])
        class_id = int(self.images_labels[index])
        class_id = torch.tensor(class_id)

        if self.transform:
            image = self.transform(image)

        return image, class_id