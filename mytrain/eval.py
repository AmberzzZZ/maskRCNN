import os
import sys
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib
from mrcnn import utils
from infer import InferenceConfig
from train import CurveDataset


# os.environ["CUDA_VISIBLE_DEVICES"] = "1,4"


if __name__ == '__main__':

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    MODEL_PATH = os.path.join(MODEL_DIR, "curve20191016T0943/mask_rcnn_curve_0011.h5")

    config = InferenceConfig()
    # config.display()

    # inference model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    model.load_weights(MODEL_PATH, by_name=True)

    # test dataset
    image_dir = "/Users/amber/workspace/maskRcnn/data/origin"
    mask_dir = "/Users/amber/workspace/maskRcnn/data/mask"
    dataset_test = CurveDataset()
    dataset_test.load_curve(image_dir, mask_dir, "test")
    dataset_test.prepare()

    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_test.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_test, config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))



