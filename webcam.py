import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import custom 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

custom_WEIGHTS_PATH = "logs/damage_major/mask_rcnn_damage_0040.h5"  # TODO: make arg

def main():

    # Config
    config = custom.CustomConfig()
    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 0

    config = InferenceConfig()

    # Device to load the neural network on.
    # Useful if you're training a model on the same 
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"


    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode=TEST_MODE,
                                model_dir=MODEL_DIR,
                                config=config)

    # load the last model you trained
    # weights_path = model.find_last()[1]

    # Load weights
    print("Loading weights ", custom_WEIGHTS_PATH)
    model.load_weights(custom_WEIGHTS_PATH, by_name=True)

    image_id = random.choice(glob.glob("car_accident/val/*.*"))
    custom.detect_and_color_splash(model, image_path=image_id)

if __name__ == "__main__":
    main()