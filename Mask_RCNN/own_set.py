from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# class that defines and loads the kangaroo dataset
class DataSetWhiteTrain(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "Schijf_zilver")
		self.add_class("dataset", 2, "Schoepen_groot")
		# define data locations
		images_dir = dataset_dir  + '/images/'
		annotations_dir = dataset_dir + '/annotations/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2])

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//object'):
			name = box.find('name').text
			xmin = int(box.find('./bndbox/xmin').text)
			ymin = int(box.find('./bndbox/ymin').text)
			xmax = int(box.find('./bndbox/xmax').text)
			ymax = int(box.find('./bndbox/ymax').text)
			coors = [xmin, ymin, xmax, ymax, name]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			if (box[4] == 'Schijf_zilver'):
				masks[row_s:row_e, col_s:col_e, i] = 1
				class_ids.append(self.class_names.index('Schijf_zilver'))
			else:
				masks[row_s:row_e, col_s:col_e, i] = 2
				class_ids.append(self.class_names.index('Schoepen_groot'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# class that defines and loads the kangaroo dataset
class DataSetWhiteTest(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=False):
		# define one class
		self.add_class("dataset", 1, "Schijf_zilver")
		self.add_class("dataset", 2, "Schoepen_groot")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annotations/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2])

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//object'):
			name = box.find('name').text
			xmin = int(box.find('./bndbox/xmin').text)
			ymin = int(box.find('./bndbox/ymin').text)
			xmax = int(box.find('./bndbox/xmax').text)
			ymax = int(box.find('./bndbox/ymax').text)
			coors = [xmin, ymin, xmax, ymax, name]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			if (box[4] == 'Schijf_zilver'):
				masks[row_s:row_e, col_s:col_e, i] = 1
				class_ids.append(self.class_names.index('Schijf_zilver'))
			else:
				masks[row_s:row_e, col_s:col_e, i] = 2
				class_ids.append(self.class_names.index('Schoepen_groot'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define a configuration for the model
class TrainSetWitteConfig(Config):
	# Give the configuration a recognizable name
	NAME = "witte_cfg"
	# Number of classes (background + schijf_zilver + Schoepen_groot)
	NUM_CLASSES = 1 + 2
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 200

class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "witte_cfg"
	# number of classes (background + schijf_zilver + Schoepen_groot)
	NUM_CLASSES = 1 + 2
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# train set
train_set = DataSetWhiteTrain()
train_set.load_dataset('DatasetWhite/train', is_train=True) #pad hier aanpassen
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# test/val set
test_set = DataSetWhiteTest()
test_set.load_dataset('DatasetWhite/val', is_train=False) #pad hier aanpassen
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# load an image
# image_id =1
# image = train_set.load_image(image_id)
# print(image.shape)
# # load image mask
# mask, class_ids = train_set.load_mask(image_id)
# print(mask.shape)
# # plot image
# pyplot.imshow(image)
# # plot mask
# pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
# pyplot.show()

##############################
# prepare config
config = TrainSetWitteConfig()
config.display()

# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)

# load weights (mscoco)
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

