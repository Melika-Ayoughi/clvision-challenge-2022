from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from devkit_tools.benchmarks import challenge_classification_benchmark
import sys
import argparse
import datetime
from pathlib import Path
from typing import List

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks.utils import Compose
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    timing_metrics, confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, LwF, EWC, GEM, AGEM
from avalanche.training.supervised.icarl import ICaRL

from devkit_tools.benchmarks import challenge_classification_benchmark, demo_classification_benchmark
from devkit_tools.metrics.classification_output_exporter import \
    ClassificationOutputExporter
from devkit_tools.challenge_constants import DEFAULT_CHALLENGE_CLASS_ORDER_SEED


# scenario = SplitMNIST( n_experiences=5, seed=DEFAULT_CHALLENGE_CLASS_ORDER_SEED, return_task_id=True)
# model = SimpleMLP(num_classes=scenario.n_classes)

DATASET_PATH = Path('/project/mayoughi/dataset')
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
scenario = challenge_classification_benchmark(
    dataset_path=DATASET_PATH,
    train_transform=train_transform,
    eval_transform=eval_transform,
    n_validation_videos=0,
    unlabeled_test_set=False,
    )
#
# MODEL CREATION
model = SimpleMLP(
    input_size=3 * 224 * 224,
    num_classes=scenario.n_classes)

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=False, experience=False, stream=False),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=False),
    confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False,
                             stream=True),
    disk_usage_metrics(minibatch=False, epoch=False, experience=False, stream=False),
    loggers=[interactive_logger, text_logger, tb_logger]
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
    evaluator=eval_plugin)

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(scenario.test_stream))