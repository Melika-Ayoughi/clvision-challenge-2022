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
Starting template for the "object classification - instances" track

Mostly based on Avalanche's "getting_started.py" example.

The template is organized as follows:
- The template is split in sections (CONFIG, TRANSFORMATIONS, ...) that can be
    freely modified.
- Don't remove the mandatory plugin (in charge of storing the test output).
- You will write most of the logic as a Strategy or as a Plugin. By default,
    the Naive (plain fine tuning) strategy is used.
- The train/eval loop should be left as it is.
- The Naive strategy already has a default logger + the accuracy metric. You
    are free to add more metrics or change the logger.
- The use of Avalanche training and logging code is not mandatory. However,
    you are required to use the given benchmark generation procedure. If not
    using Avalanche, make sure you are following the same train/eval loop and
    please make sure you are able to export the output in the expected format.
"""
import sys
import argparse
import datetime
from pathlib import Path
from typing import List

import torch
from torch.optim import SGD
from torchvision import transforms
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop
import random
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.benchmarks.utils import Compose
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    timing_metrics, forgetting_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, LwF, EWC, GEM, AGEM
from avalanche.training.supervised.icarl import ICaRL

from devkit_tools.benchmarks import challenge_classification_benchmark, demo_classification_benchmark
from devkit_tools.metrics.classification_output_exporter import \
    ClassificationOutputExporter
from devkit_tools.challenge_constants import DEFAULT_CHALLENGE_CLASS_ORDER_SEED
from os.path import expanduser

from avalanche.benchmarks.datasets import CIFAR100
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.training.plugins import EvaluationPlugin
from pytorchcv.model_provider import get_model as ptcv_get_model
from avalanche.training.plugins import CoPEPlugin


DATASET_PATH = Path('/project/mayoughi/dataset')
EXP_NAME = "cope_splitcifar"

def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if args.cuda >= 0 and torch.cuda.is_available()
        else "cpu"
    )

    torchvision_normalization = transforms.Normalize(
        mean=[0.485,  0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # Add additional transformations here
    train_transform = Compose(
        [RandomCrop(224, padding=10, pad_if_needed=True),
         ToTensor(),
         torchvision_normalization]
    )

    # Don't add augmentation transforms to the eval transformations!
    eval_transform = Compose(
        [ToTensor(), torchvision_normalization]
    )
    # ---------

    # benchmark = challenge_classification_benchmark(
    #     dataset_path=DATASET_PATH,
    #     class_order_seed=DEFAULT_CHALLENGE_CLASS_ORDER_SEED,
    #     train_transform=train_transform,
    #     eval_transform=eval_transform,
    #     n_validation_videos=0,
    # )
    def icarl_cifar100_augment_data(img):
        img = img.numpy()
        padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode="constant")
        random_cropped = np.zeros(img.shape, dtype=np.float32)
        crop = np.random.randint(0, high=8 + 1, size=(2,))

        # Cropping and possible flipping
        if np.random.randint(2) > 0:
            random_cropped[:, :, :] = padded[
                                      :, crop[0]: (crop[0] + 32), crop[1]: (crop[1] + 32)
                                      ]
        else:
            random_cropped[:, :, :] = padded[
                                      :, crop[0]: (crop[0] + 32), crop[1]: (crop[1] + 32)
                                      ][:, :, ::-1]
        t = torch.tensor(random_cropped)
        return t

    from avalanche.benchmarks import Experience, SplitCIFAR100, SplitCIFAR110
    benchmark = SplitCIFAR100(10, shuffle=False, return_task_id=False, train_transform=Compose(
        [  # RandomCrop(224, padding=10, pad_if_needed=True),
            ToTensor(),
            icarl_cifar100_augment_data]
    ), eval_transform=Compose(
        [ToTensor()]
    ))
    print(benchmark.task_labels)
    # ---------

    # --- MODEL CREATION
    # model = SimpleMLP(
    #     input_size=3*224*224,
    #     num_classes=benchmark.n_classes)
    # model = ptcv_get_model("resnet50", pretrained=True) #imagenet pretrained
    # model.output = Linear(in_features=2048, out_features=benchmark.n_classes, bias=True)
    model = SimpleMLP(
        input_size=3*32*32,
        num_classes=benchmark.n_classes)


    mandatory_plugins = [
        ClassificationOutputExporter(
            benchmark, save_folder=f'./output/baseline/{EXP_NAME}/instance_classification_results')
    ]
    plugins: List[SupervisedPlugin] = [
        # ...
    ] + mandatory_plugins
    # ---------

    # --- METRICS AND LOGGING
    evaluator = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=False, experience=True, stream=True),
        timing_metrics(epoch=False, epoch_running=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=False),
        confusion_matrix_metrics(num_classes=benchmark.n_classes, save_image=True, stream=True),
        disk_usage_metrics(minibatch=False, epoch=False, experience=False, stream=False),
        # collect_all=True,
        loggers=[InteractiveLogger(), TextLogger(open(f'./output/baseline/{EXP_NAME}/log.txt', 'a')),
                 TensorboardLogger(tb_log_dir=f'./output/baseline/{EXP_NAME}/exp_' + datetime.datetime.now().isoformat())],
    )
    # ---------

    # CoPE PLUGIN
    cope = CoPEPlugin(
        mem_size=2000, alpha=0.99, p_size=benchmark.n_classes, n_classes=benchmark.n_classes
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE) WITH CoPE PLUGIN
    cl_strategy = Naive(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        cope.ppp_loss,  # CoPE PPP-Loss
        train_mb_size=10,
        train_epochs=5, #70,
        eval_mb_size=100,
        device=device,
        plugins=plugins + [cope],
        evaluator=evaluator,
    )

    # ---------

    # TRAINING LOOP
    results = []
    print("Starting experiment...")
    for experience in benchmark.train_stream:
        current_experience_id = experience.current_experience
        print("Start of experience: ", current_experience_id)
        print("Current Classes: ", experience.classes_in_this_experience)

        data_loader_arguments = dict(
            num_workers=8,
            persistent_workers=True
        )

        if 'valid' in benchmark.streams:
            # Each validation experience is obtained from the training
            # experience directly. We can't use the whole validation stream
            # (because that means accessing future or past data).
            # For this reason, validation is done only on
            # `valid_stream[current_experience_id]`.
            cl_strategy.train(
                experience,
                eval_streams=[benchmark.valid_stream[current_experience_id]],
                **data_loader_arguments)
        else:
            cl_strategy.train(
                experience,
                **data_loader_arguments)
        print("Training completed")

        print("Computing accuracy on the complete test set")
        # cl_strategy.eval(benchmark.test_stream, num_workers=10, persistent_workers=True)
        # results = evaluator.get_all_metrics()
        results.append(cl_strategy.eval(benchmark.test_stream, num_workers=8, persistent_workers=True))
        # print(f"All metrics: ", {results})
        # for k, v in results.items():
        #     print(k, v)


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
