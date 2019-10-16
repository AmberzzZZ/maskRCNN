import os
import sys
import numpy as np
import cv2
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,4"


# Configuration
class CurveConfig(Config):

    # Give the configuration a recognizable name
    NAME = "curve"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPU_COUNT * IMAGES_PER_GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1       # 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1      # background + 1 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    IMAGE_CHANNEL_COUNT = 1      # default 3

    # Image mean (default RGB)
    MEAN_PIXEL = np.array([128])

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 30

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


# Dataset
class CurveDataset(utils.Dataset):

    def load_curve(self, dataset_dir, mask_dir, subset):
        """Load a subset of the Curve dataset
        """
        # Add classes.
        self.add_class("curve", 1, "curve")

        # choose dataset
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add images
        for file in os.listdir(dataset_dir):
            self.add_image(
                "curve",
                image_id=file,
                path=os.path.join(dataset_dir, file),
                mask_path=os.path.join(mask_dir, file),
                width=512, height=512)

    def load_mask(self, image_id):
        """Generate instance masks for an image
           Returns:
           masks: A bool array of shape [height, width, instance count] with
                  a binary mask per instance.
           class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        maskfile_dir = image_info['mask_path']
        mask = cv2.imread(maskfile_dir, 0)
        mask = np.expand_dims(mask, axis=-1)       # must keep the [INSTANCE COUNT] dim!

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,C] Numpy array
        """
        image_info = self.image_info[image_id]
        imgfile_dir = image_info['path']
        image = cv2.imread(imgfile_dir, 0)
        image = np.expand_dims(image, axis=-1)       # must keep the [CHANNEL] dim!

        return image


def train(model, all=False):
    """Train the model."""
    image_dir = "/Users/amber/workspace/maskRcnn/data/origin"
    mask_dir = "/Users/amber/workspace/maskRcnn/data/mask"

    # Training dataset.
    dataset_train = CurveDataset()
    dataset_train.load_curve(image_dir, mask_dir, "train")
    dataset_train.prepare()          # multi-classification  ---->  multi binary-classification

    # Validation dataset.
    dataset_val = CurveDataset()
    dataset_val.load_curve(image_dir, mask_dir, "val")
    dataset_val.prepare()

    # Training schedule
    if not all:
        print("Trainning network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers='heads')
    else:
        print("Trainning network all")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=50,               # this number should follow the last training step
                    layers='all')


if __name__ == '__main__':

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    config = CurveConfig()
    # config.display()

    # training model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # train stage 1
    # model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
    #         'mrcnn_class_logits', 'mrcnn_bbox_fc',
    #         'mrcnn_bbox', 'mrcnn_mask', 'conv1'])       # conv1
    # train stage 2
    model.load_weights(model.find_last(), by_name=True)

    train(model, all=True)




