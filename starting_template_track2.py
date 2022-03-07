################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-02-2022                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
Starting template for the 2nd challenge track (object detection)

Mostly based on Avalanche's "getting_started.py" example.

The template is organized as follows:
- The template is split in sections (CONFIG, TRANSFORMATIONS, ...) that can be
    freely modified (apart from the BENCHMARK CREATION one).
- Don't remove the mandatory metric (in charge of storing the test output).
- You will write most of the logic as a Strategy or as a Plugin. By default,
    the Naive (plain fine tuning) strategy is used.
- The train/eval loop should be left as it is.
- The Naive strategy already has a default logger + the detection metrics. You
    are free to add more metrics or change the logger.
"""

import argparse
import logging
from pathlib import Path
from typing import List

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor

from avalanche.benchmarks.utils import Compose
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from devkit_tools.benchmarks import demo_detection_benchmark
from devkit_tools.benchmarks.classification_benchmark import \
    demo_classification_benchmark
from devkit_tools.challenge_constants import DEFAULT_DEMO_CLASS_ORDER_SEED


from devkit_tools.metrics.detection_output_exporter import EgoMetrics
from devkit_tools.templates.detection_template import ObjectDetectionTemplate

# TODO: change this to the path where you downloaded (and extracted) the dataset
DATASET_PATH = Path.home() / '3rd_clvision_challenge' / 'demo_dataset'

# Don't change this (unless you want to experiment with different class orders)
# Note: it won't be possible to change the class order in the real challenge
CLASS_ORDER_SEED = DEFAULT_DEMO_CLASS_ORDER_SEED

# This sets the root logger to write to stdout (your console).
# Customize the logging level as you wish.
logging.basicConfig(level=logging.NOTSET)


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if args.cuda >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    # ---------

    # --- TRANSFORMATIONS
    # Add additional transformations here
    # You can take some detection transformations here:
    # https://github.com/pytorch/vision/blob/main/references/detection/transforms.py
    # Beware that:
    # - transforms found in torchvision.transforms.transforms will only act on
    #    the image and they will not adjust bounding boxes accordingly: don't
    #    use them (apart from ToTensor)!
    # - make sure you are using the "Compose" from avalanche.benchmarks.utils,
    #    not the one from torchvision or from the aforementioned link.
    train_transform = Compose(
        [ToTensor()]
    )

    # Don't add augmentation transforms to the eval transformations!
    eval_transform = Compose(
        [ToTensor()]
    )
    # ---------

    # --- BENCHMARK CREATION
    benchmark = demo_detection_benchmark(
        dataset_path=DATASET_PATH,
        class_order_seed=CLASS_ORDER_SEED,
        train_transform=train_transform,
        eval_transform=eval_transform
    )
    # ---------

    # --- MODEL CREATION
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # By disabling the grad on current params, only the box predictor is tuned
    # (because it's created later, in the next few lines of code)
    for p in model.parameters():
        p.requires_grad = False

    num_classes = benchmark.n_classes + 1  # N classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)
    print('Num classes', num_classes)
    # --- OPTIMIZER AND SCHEDULER CREATION

    # Create the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # Define the scheduler
    train_mb_size = 5
    warmup_factor = 1.0 / 1000
    warmup_iters = min(
        1000, len(benchmark.train_stream[0].dataset) // train_mb_size - 1
    )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )
    # ---------

    # TODO: ObjectDetectionTemplate == Naive == plain fine tuning without
    #  replay, regularization, etc.
    # For the challenge, you'll have to implement your own strategy (or a
    # strategy plugin that changes the behaviour of the ObjectDetectionTemplate)

    # --- PLUGINS CREATION
    # Avalanche already has a lot of plugins you can use!
    # Many mainstream continual learning approaches are available as plugins:
    # https://avalanche-api.continualai.org/en/latest/training.html#training-plugins
    #
    mandatory_plugins = []
    plugins: List[SupervisedPlugin] = [
        LRSchedulerPlugin(lr_scheduler)
        # ...
    ] + mandatory_plugins
    # ---------

    # --- METRICS AND LOGGING
    mandatory_metrics = [EgoMetrics(save_folder='./track2_results')]
    evaluator = EvaluationPlugin(
        mandatory_metrics,
        timing_metrics(
            experience=True, stream=True
        ),
        loggers=[InteractiveLogger()]
    )
    # ---------

    # --- CREATE THE STRATEGY INSTANCE
    cl_strategy = ObjectDetectionTemplate(
        model=model,
        optimizer=optimizer,
        train_mb_size=train_mb_size,
        train_epochs=1,
        eval_mb_size=train_mb_size,
        device=device,
        plugins=plugins,
        evaluator=evaluator
    )
    # ---------

    # TRAINING LOOP
    print("Starting experiment...")
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)

        cl_strategy.train(experience, num_workers=4)
        print("Training completed")

        print("Computing accuracy on the full test set")
        cl_strategy.eval(benchmark.test_stream[0], num_workers=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
