import os
import sys
import numpy as np
import cv2
import glob

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib
from mrcnn import visualize
from train import CurveConfig


# os.environ["CUDA_VISIBLE_DEVICES"] = "1,4"

class InferenceConfig(CurveConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == '__main__':

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    MODEL_PATH = os.path.join(MODEL_DIR, "curve20191016T0943/mask_rcnn_curve_0011.h5")

    config = InferenceConfig()
    # config.display()

    # inference model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    model.load_weights(MODEL_PATH, by_name=True)

    class_names = ['BG', 'curve']

    test_dir = "/Users/amber/workspace/maskRcnn/data/origin/test"
    for file in glob.glob(test_dir + "/*png"):
        image = cv2.imread(file, 0)
        image = np.expand_dims(image, axis=-1)

        # run detection
        results = model.detect([image], verbose=1)

        # visualize results
        r = results[0]            # batch[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])




