# imports
import os
# Flask
from flask import Flask, render_template, jsonify, send_file, request, Response, flash, redirect, url_for
from werkzeug.utils import secure_filename
# Sessions
from uuid import uuid4
# Time
import time
# Numpy
import numpy as np
# OpenCV
import cv2
# Skimage
import skimage.io
# Torch
import torch, torchvision
from torchvision.utils import save_image
# Detectron2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
# Send multiple files
from requests_toolbelt import MultipartEncoder
# Logging
import logging
logging.basicConfig(level=logging.DEBUG)

# APP
app = Flask(__name__)
app.config['SECRET_KEY'] = 'this is another secret key!'

# Static path
static_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"))
# Uploads path
uploads_path = os.path.join(static_path, "uploads")
app.config['UPLOAD_FOLDER'] = uploads_path

# Colors
COLORS = [
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
]

# Categories
# https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
id_to_category = {
	0: u'__background__',
	1: u'person',
	2: u'bicycle',
	3: u'car',
	4: u'motorcycle',
	5: u'airplane',
	6: u'bus',
	7: u'train',
	8: u'truck',
	9: u'boat',
	10: u'traffic light',
	11: u'fire hydrant',
	12: u'stop sign',
	13: u'parking meter',
	14: u'bench',
	15: u'bird',
	16: u'cat',
	17: u'dog',
	18: u'horse',
	19: u'sheep',
	20: u'cow',
	21: u'elephant',
	22: u'bear',
	23: u'zebra',
	24: u'giraffe',
	25: u'backpack',
	26: u'umbrella',
	27: u'handbag',
	28: u'tie',
	29: u'suitcase',
	30: u'frisbee',
	31: u'skis',
	32: u'snowboard',
	33: u'sports ball',
	34: u'kite',
	35: u'baseball bat',
	36: u'baseball glove',
	37: u'skateboard',
	38: u'surfboard',
	39: u'tennis racket',
	40: u'bottle',
	41: u'wine glass',
	42: u'cup',
	43: u'fork',
	44: u'knife',
	45: u'spoon',
	46: u'bowl',
	47: u'banana',
	48: u'apple',
	49: u'sandwich',
	50: u'orange',
	51: u'broccoli',
	52: u'carrot',
	53: u'hot dog',
	54: u'pizza',
	55: u'donut',
	56: u'cake',
	57: u'chair',
	58: u'couch',
	59: u'potted plant',
	60: u'bed',
	61: u'dining table',
	62: u'toilet',
	63: u'tv',
	64: u'laptop',
	65: u'mouse',
	66: u'remote',
	67: u'keyboard',
	68: u'cell phone',
	69: u'microwave',
	70: u'oven',
	71: u'toaster',
	72: u'sink',
	73: u'refrigerator',
	74: u'book',
	75: u'clock',
	76: u'vase',
	77: u'scissors',
	78: u'teddy bear',
	79: u'hair drier',
	80: u'toothbrush'
}

# Params
COLOR_TEXT = (255, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
thickness = 2

# Detector
print('[DETECTOR APP] Load Detectron\'s config...')
cfg = get_cfg()
cfg.merge_from_file("/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
# Run it on the CPU
cfg.MODEL.DEVICE = 'cpu'
# Predictor
print('[DETECTOR APP] Load predictor...')
predictor = DefaultPredictor(cfg)
print('[DETECTOR APP] Detectron has been loaded!')


@app.route('/', methods=['POST'])
def process_files():

	# Create a unique "session ID" for this particular batch of uploads
	upload_key = str(uuid4())
	# Pattern for file names
	directory = time.strftime("%Y%m%d-%H%M%S") + '-' + upload_key
	upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], directory)
	images_dir = os.path.join(upload_dir, 'images')
	rois_dir = os.path.join(upload_dir, 'rois')
	masks_dir = os.path.join(upload_dir, 'masks')
	# Create dirs
	os.makedirs(images_dir)
	os.makedirs(rois_dir)
	os.makedirs(masks_dir)

	# Files to send
	files_to_send = []

	# Receive the images and store them
	images = request.files.to_dict()
	
	for image in images:
		
		# Filename
		filename = secure_filename(images[image].filename)
		print('[HUMAN-MASK] File received: {}'.format(filename))
		logging.info('File received: {}'.format(filename))
		filename_and_path = os.path.join(images_dir, filename)
		name = filename.split('.')[0]
		
		# Save it!
		with open(filename_and_path, 'wb') as f:
			f.write(images[image].read())
		
		# Image
		original_img = cv2.imread(filename_and_path)

		# Size
		original_size = original_img.shape
		print('[HUMAN-MASK] Image shape: {}'.format(original_size))

		# Overlay
		overlay = original_img.copy()
		
		# Predict!
		print('[HUMAN-MASK] Predicting...')
		outputs = predictor(original_img)

		# Instances
		# https://detectron2.readthedocs.io/_modules/detectron2/structures/instances.html
		print('[HUMAN-MASK] Analysing predictions...')
		instances = outputs["instances"]

		# Image size
		pred_image_size = instances.image_size

		# Resize it
		# https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/postprocessing.py
		scale_x, scale_y = (original_size[1] / pred_image_size[1], original_size[0] / pred_image_size[0])

		# Get attributes
		pred_boxes = instances.get('pred_boxes')
		pred_boxes.scale(scale_x, scale_y)
		pred_boxes.clip(pred_image_size)
		pred_boxes = pred_boxes.tensor
		scores = instances.get('scores').numpy()
		pred_classes = instances.get('pred_classes').numpy()
		pred_masks = instances.pred_masks

		for j in range(len(instances)):

			print('[HUMAN-MASK] Reading prediction {} / {}...'.format(j + 1, len(instances)))

			# Get attributes for each instance
			category_id = int(pred_classes[j]) + 1
			category_text = id_to_category[category_id].replace(' ', '_') if category_id in id_to_category else 'unknown'
			# Bounding box (rescaled)
			bbox = pred_boxes[j].numpy()
			bbox_str = '_'.join([ str(int(x)) for x in bbox ])
			# Score (confidence)
			score =  float(scores[j])

			# ROI

			# Bounding box
			# https://github.com/facebookresearch/detectron2/blob/d1afdf4b7fc2efd58d342c11a3882a10f16f04d1/detectron2/utils/visualizer.py#L802
			# box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
			# are the coordinates of the image's top left corner. x1 and y1 are the
			# coordinates of the image's bottom right corner.
			x1, y1, x2, y2 = bbox
			w = x2 - x1
			h = y2 - y1

			# Bounding box is empty
			if w == 0 or h == 0:
				continue

			# New image using only the bounding box
			roi_image = original_img[int(y1):int(y1 + h), int(x1):int(x1 + w)]

			# New image path
			roi_image_name = '{}-detection-roi-{}-{:.4f}-{}.jpg'.format(name, category_text, score, bbox_str)
			roi_image_path = os.path.join(app.config['UPLOAD_FOLDER'], directory, 'rois', roi_image_name)

			# Write
			cv2.imwrite(roi_image_path, roi_image)

			# Add it to send!
			files_to_send.append(roi_image_path)

			# MASK

			# Predicted Mask: convert from True / False to 255 / 0
			pred_mask = pred_masks[j].numpy().astype(np.uint8) * 255
			# Apply mask to image
			masked_image = cv2.bitwise_and(original_img, original_img, mask = pred_mask)
			# Reverse mask
			reverse_mask = cv2.bitwise_not(pred_mask)
			# White background
			background = np.full(original_img.shape, 255, dtype = np.uint8)
			# Apply reverse mask to image
			reverse_masked_image = cv2.bitwise_and(background, background, mask = reverse_mask)
			white_mask = cv2.bitwise_or(masked_image, reverse_masked_image)

			# Get ROI
			mask_only_roi = pred_mask[int(y1):int(y1 + h), int(x1):int(x1 + w)]
			masked_roi_image = masked_image[int(y1):int(y1 + h), int(x1):int(x1 + w)]
			reverse_masked_roi_image = reverse_masked_image[int(y1):int(y1 + h), int(x1):int(x1 + w)]
			white_masked_roi_image = white_mask[int(y1):int(y1 + h), int(x1):int(x1 + w)]

			# Create a directory per image
			custom_mask_dir = os.path.join(app.config['UPLOAD_FOLDER'], directory, 'masks', str(j))
			os.mkdir(custom_mask_dir)
			# Save image
			mask_only_image_path = os.path.join(custom_mask_dir, '{}-mask-{}-{:.4f}-{}.jpg'.format(name, category_text, score, bbox_str))
			mask_image_path = os.path.join(custom_mask_dir, '{}-detection-mask-{}-{:.4f}-{}.jpg'.format(name, category_text, score, bbox_str))
			reverse_mask_image_path = os.path.join(custom_mask_dir, '{}-reverse-mask-{}-{:.4f}-{}.jpg'.format(name, category_text, score, bbox_str))
			white_mask_image_path = os.path.join(custom_mask_dir, '{}-detection-reverse-mask-{}-{:.4f}-{}.jpg'.format(name, category_text, score, bbox_str))

			# Write
			cv2.imwrite(mask_only_image_path, mask_only_roi)
			cv2.imwrite(mask_image_path, masked_roi_image)
			cv2.imwrite(reverse_mask_image_path, reverse_masked_roi_image)
			cv2.imwrite(white_mask_image_path, white_masked_roi_image)

			# Add it to send!
			files_to_send.append(mask_only_image_path)
			files_to_send.append(mask_image_path)
			files_to_send.append(reverse_mask_image_path)
			files_to_send.append(white_mask_image_path)

			# ANNOTATED IMAGE

			# Color
			color = COLORS[j] if j < len(COLORS) else COLORS[len(COLORS) - 1]

			# Draw contours for predicted masks
			results = cv2.findContours(pred_masks[j].numpy().astype(np.uint8), 
				cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
			polygons = results[-2]
			polygons = [x.flatten() for x in polygons]
			polygons = [x for x in polygons if len(x) >= 6]
			for polygon in polygons:
				pts = polygon.reshape((-1, 1, 2))
				cv2.fillPoly(overlay, [ pts ], color)

			# Draw rectangles for ROIs
			cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

			# Put text for labels and scores
			label = "{} {:.0f}%".format(category_text, score * 100)
			cv2.putText(overlay, label, 
				(int(x1 + 10), int(y1 + 30)), font, fontScale, COLOR_TEXT, thickness)

	# Annotated image path
	annotated_image_name = '{}-annotated.jpg'.format(name)
	annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], directory, annotated_image_name)

	# apply the overlay
	alpha = 0.5
	cv2.addWeighted(overlay, alpha, original_img, 1 - alpha, 0, original_img)

	# Write
	cv2.imwrite(annotated_image_path, original_img)

	# Add it to send!
	files_to_send.append(annotated_image_path)

	# # Return one file
	# response = send_file(annotated_image_path, mimetype='image/png')
	# return response
	
	# Return multiple files
	# https://github.com/requests/toolbelt
	files = {}
	for file_to_send in files_to_send:
		name = file_to_send.split('/')[-1]
		files[name] = (name, open(file_to_send, 'rb'), 'text/plain')
	m = MultipartEncoder(files)
	return Response(m.to_string(), mimetype=m.content_type)


if __name__ == '__main__':

	app.run(host='0.0.0.0', debug=True, port=5000)