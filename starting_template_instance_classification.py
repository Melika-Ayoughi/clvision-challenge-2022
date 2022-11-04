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
# from avalanche.models.pytorchcv_wrapper import (
#     vgg,
#     resnet,
#     densenet,
#     pyramidnet,
#     get_model,
# )

# TODO: change this to the path where you downloaded (and extracted) the dataset
# DATASET_PATH = Path.home() / 'clvision-challenge-2022' / 'dataset'
DATASET_PATH = Path('/project/mayoughi/dataset')
# EXP_NAME = "icarl_ego_debug" #"log_ewc_8workers"


def reproduce():
    random.seed(DEFAULT_CHALLENGE_CLASS_ORDER_SEED)
    np.random.seed(DEFAULT_CHALLENGE_CLASS_ORDER_SEED)
    torch.manual_seed(DEFAULT_CHALLENGE_CLASS_ORDER_SEED)
    torch.cuda.manual_seed(DEFAULT_CHALLENGE_CLASS_ORDER_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def main(args):

    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if args.cuda >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    # ---------

    # --- TRANSFORMATIONS
    # This is the normalization used in torchvision models
    # https://pytorch.org/vision/stable/models.html
    torchvision_normalization = transforms.Normalize( #todo
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

    # --- BENCHMARK CREATION
    # benchmark = demo_classification_benchmark(
    #     dataset_path=DATASET_PATH,
    #     class_order_seed=DEFAULT_CHALLENGE_CLASS_ORDER_SEED,
    #     train_transform=train_transform,
    #     eval_transform=eval_transform
    # )
    benchmark = challenge_classification_benchmark(
        dataset_path=DATASET_PATH,
        class_order_seed=DEFAULT_CHALLENGE_CLASS_ORDER_SEED,
        train_transform=train_transform,
        eval_transform=eval_transform,
        # n_validation_videos=1,
        unlabeled_test_set=False,
    )
    # def icarl_cifar100_augment_data(img):
    #     img = img.numpy()
    #     padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode="constant")
    #     random_cropped = np.zeros(img.shape, dtype=np.float32)
    #     crop = np.random.randint(0, high=8 + 1, size=(2,))
    #
    #     # Cropping and possible flipping
    #     if np.random.randint(2) > 0:
    #         random_cropped[:, :, :] = padded[
    #                                   :, crop[0]: (crop[0] + 32), crop[1]: (crop[1] + 32)
    #                                   ]
    #     else:
    #         random_cropped[:, :, :] = padded[
    #                                   :, crop[0]: (crop[0] + 32), crop[1]: (crop[1] + 32)
    #                                   ][:, :, ::-1]
    #     t = torch.tensor(random_cropped)
    #     return t

    # transforms_group = dict(
    #     eval=(
    #         transforms.Compose(
    #             [
    #                 transforms.ToTensor(),
    #                 # lambda img_pattern: img_pattern - per_pixel_mean,
    #             ]
    #         ),
    #         None,
    #     ),
    #     train=(
    #         transforms.Compose(
    #             [
    #                 transforms.ToTensor(),
    #                 # lambda img_pattern: img_pattern - per_pixel_mean,
    #                 icarl_cifar100_augment_data,
    #             ]
    #         ),
    #         None,
    #     ),
    # )

    # train_set = CIFAR100(
    #     expanduser("~") + "/.avalanche/data/cifar100/",
    #     train=True,
    #     download=True,
    # )
    # test_set = CIFAR100(
    #     expanduser("~") + "/.avalanche/data/cifar100/",
    #     train=False,
    #     download=True,
    # )
    #
    # train_set = AvalancheDataset(
    #     train_set,
    #     transform_groups=transforms_group,
    #     initial_transform_group="train",
    # )
    # test_set = AvalancheDataset(
    #     test_set,
    #     transform_groups=transforms_group,
    #     initial_transform_group="eval",
    # )
    #
    # benchmark = nc_benchmark(
    #     train_dataset=train_set,
    #     test_dataset=test_set,
    #     n_experiences=10,
    #     task_labels=False,
    #     shuffle=False,
    # )
    # from avalanche.benchmarks import Experience, SplitCIFAR100, SplitCIFAR110
    # benchmark = SplitCIFAR100(10, shuffle=False, return_task_id=False, train_transform=Compose(
    #     [#RandomCrop(224, padding=10, pad_if_needed=True),
    #      ToTensor(),
    #      icarl_cifar100_augment_data]
    # ), eval_transform=Compose(
    #     [ToTensor()]
    # ))
    # print(benchmark.task_labels)
    # ---------

    # --- MODEL CREATION
    # model = SimpleMLP(
    #     input_size=3*224*224,
    #     num_classes=benchmark.n_classes)
    # ---------
    # For the challenge, you'll have to implement your own strategy (or a
    # strategy plugin that changes the behaviour of the SupervisedTemplate)

    # --- PLUGINS CREATION
    # Avalanche already has a lot of plugins you can use!
    # Many mainstream continual learning approaches are available as plugins:
    # https://avalanche-api.continualai.org/en/latest/training.html#training-plugins
    mandatory_plugins = [
        ClassificationOutputExporter(
            benchmark, save_folder=f'./output/baseline/{args.EXP_NAME}/instance_classification_results')
    ]
    plugins: List[SupervisedPlugin] = [
    ] + mandatory_plugins

    evaluator = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=False, experience=True, stream=True),
        timing_metrics(epoch=False, epoch_running=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=False),
        confusion_matrix_metrics(num_classes=benchmark.n_classes, save_image=True, stream=True),
        disk_usage_metrics(minibatch=False, epoch=False, experience=False, stream=False),
        # collect_all=True,
        loggers=[InteractiveLogger(), TextLogger(open(f'./output/baseline/{args.EXP_NAME}/log.txt', 'a')),
                 TensorboardLogger(tb_log_dir=f'./output/baseline/{args.EXP_NAME}/exp_' + datetime.datetime.now().isoformat())],
        benchmark=benchmark,
    )
    # ---------

    # --- CREATE THE STRATEGY INSTANCE
    # In Avalanche, you can customize the training loop in 3 ways:
    #   1. Adapt the make_train_dataloader, make_optimizer, forward,
    #   criterion, backward, optimizer_step (and other) functions. This is the
    #   clean way to do things!
    #   2. Change the loop itself by reimplementing training_epoch or even
    #   _train_exp (not recommended).
    #   3. Create a Plugin that, by implementing the proper callbacks,
    #   can modify the behavior of the strategy.
    #  -------------
    #  Consider that popular strategies (EWC, LwF, Replay) are implemented
    #  as plugins. However, writing a plugin from scratch may be a tad
    #  tedious. For the challenge, we recommend going with the 1st option.
    #  In particular, you can create a subclass of the SupervisedTemplate
    #  (Naive is mostly an alias for the SupervisedTemplate) and override only
    #  the methods required to implement your solution.
    # cl_strategy = Naive(
    #     model,
    #     SGD(model.parameters(), lr=0.001, momentum=0.9),
    #     CrossEntropyLoss(),
    #     train_mb_size=100,
    #     train_epochs=10,
    #     eval_mb_size=100,
    #     device=device,
    #     plugins=plugins,
    #     evaluator=evaluator,
    #     eval_every=0 if 'valid' in benchmark.streams else -1
    # )
    # cl_strategy = EWC(
    #     model,
    #     SGD(model.parameters(), lr=0.001, momentum=0.9),
    #     CrossEntropyLoss(),
    #     0.4, #args.ewc_lambda,
    #     "online", #args.ewc_mode ["separate", "online"]
    #     decay_factor=0.1, #args.decay_factor,
    #     train_epochs=10, #args.epochs,
    #     device=device,
    #     train_mb_size=100 , #args.minibatch_size,
    #     evaluator=evaluator,
    # )

    if args.baseline == "icarl":
        def icarl_egoobjects_augment_data(img):
            img = img.numpy()
            padded = np.pad(img, ((0, 0), (4, 4), (4, 4)),
                            mode="constant")  # no padding in rgb axis and just in width and height
            random_cropped = np.zeros(img.shape, dtype=np.float32)
            crop = np.random.randint(0, high=8 + 1, size=(2,))

            # Cropping and possible flipping
            if np.random.randint(2) > 0:
                random_cropped[:, :, :] = padded[
                                          :, crop[0]: (crop[0] + 224), crop[1]: (crop[1] + 224)
                                          ]
            else:
                random_cropped[:, :, :] = padded[
                                          :, crop[0]: (crop[0] + 224), crop[1]: (crop[1] + 224)
                                          ][:, :, ::-1]
            t = torch.tensor(random_cropped)
            return t

        model: IcarlNet = make_icarl_net(num_classes=benchmark.n_classes)
        model.apply(initialize_icarl_net)

        optim = SGD(
            model.parameters(),
            lr=2.0,
            weight_decay=0.00001,
            momentum=0.9,
        )

        sched = LRSchedulerPlugin(
            MultiStepLR(optim, [49, 63], gamma=1.0 / 5.0)
        )

        cl_strategy = ICaRL(
            model.feature_extractor,
            model.classifier,
            optim,
            memory_size=2000,
            buffer_transform=transforms.Compose([icarl_egoobjects_augment_data]),
            device=device,
            train_mb_size=args.batch_size,
            fixed_memory=True,
            train_epochs=args.train_epochs,
            plugins=plugins + [sched],
            evaluator=evaluator,
        )
    elif args.baseline == "naive":
        # Naive == plain fine tuning without replay, regularization, etc.

        model = ptcv_get_model("resnet34", pretrained=False)  # imagenet pretrained
        model.output = Linear(in_features=512, out_features=benchmark.n_classes, bias=True)
        cl_strategy = Naive(
            model,
            SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(),
            train_mb_size=args.batch_size,
            train_epochs=args.train_epochs,
            eval_mb_size=100,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=0 if 'valid' in benchmark.streams else -1
        )
    elif args.baseline == "cope":
        model = ptcv_get_model("resnet34", pretrained=False)  # imagenet pretrained
        model.output = Linear(in_features=512, out_features=benchmark.n_classes, bias=True)

        from avalanche.training.plugins import CoPEPlugin
        # CoPE PLUGIN
        cope = CoPEPlugin(
            mem_size=2000, alpha=0.99, p_size=benchmark.n_classes, n_classes=benchmark.n_classes
        )

        # CREATE THE STRATEGY INSTANCE (NAIVE) WITH CoPE PLUGIN
        cl_strategy = Naive(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            cope.ppp_loss,  # CoPE PPP-Loss
            train_mb_size=args.batch_size,
            train_epochs=args.train_epochs,
            eval_mb_size=100,
            device=device,
            plugins=plugins + [cope],
            evaluator=evaluator,
        )


    # TRAINING LOOP
    results = []
    print("Starting experiment...")
    for experience in benchmark.train_stream:
        current_experience_id = experience.current_experience
        print("Start of experience: ", current_experience_id)
        print("Current Classes: ", experience.classes_in_this_experience)

        data_loader_arguments = dict(
            num_workers=1,
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
        results.append(cl_strategy.eval(benchmark.test_stream, num_workers=1, persistent_workers=True))
        # print(f"All metrics: ", {results})
        # for k, v in results.items():
        #     print(k, v)
        torch.save(model, f'./output/baseline/{args.EXP_NAME}/model.pt')


if __name__ == "__main__":
    reproduce()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--EXP_NAME",
        type=str,
        default="cope_ego",
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="cope",
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="The batch size for training.",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=100,
        help="The number of training epochs.",
    )

    args = parser.parse_args()
    main(args)
