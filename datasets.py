import os
import ipdb
import glob
import numpy as np
import pandas as pd
import cv2

from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
	
	def __init__(self, data_set_path, transforms=None, type_='train'):
		self.data_set_path = data_set_path
		self.type_ = type_
		self.image_files_path, self.length = self.read_data_set()
		self.transforms = transforms
	
	def read_data_set(self):

		img_files  = glob.glob(self.data_set_path + '/images/*.png')
		img_files = np.array(img_files)
		
		labels_raw = pd.read_csv(self.data_set_path + 'responses.csv')
		self.all_labels = labels_raw.set_index('id').to_dict()['corr']
		
		all_img_files = []
		proposal = []

		np.random.seed(123)
		proposal = np.random.choice(len(img_files), size=int(len(img_files) * 0.8), replace=False)
		if self.type_ == 'train':
			all_img_files = img_files[proposal]
		else:
			mask = np.zeros_like(img_files, bool)
			mask[proposal] = True
			all_img_files = img_files[~mask]		

		return all_img_files, len(all_img_files)


	def __getitem__(self, index):

		# gray scale with coordinate
		img = cv2.imread(self.image_files_path[index], 0)
		(h,w) = img.shape[:2]
		vertical, parallel = (np.indices((h,w)) / 149 * 255).astype(np.uint8)
		img = np.stack((img, vertical, parallel), axis=-1)

		# rgb image
		# img = cv2.imread(self.image_files_path[index])
		
		# get label name
		key = os.path.splitext(os.path.basename(self.image_files_path[index]))[0]
		

		if self.transforms is not None:
			image = self.transforms(img)

		return {'image': image, 'label': self.all_labels[key]}

	def __len__(self):
		return self.length

