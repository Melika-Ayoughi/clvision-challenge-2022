from torchvision.transforms import Resize

DEFAULT_DEMO_CLASS_ORDER_SEED = 20220307
DEFAULT_DEMO_TRAIN_JSON = 'ego_objects_demo_train.json'
DEFAULT_DEMO_TEST_JSON = 'ego_objects_demo_test.json'

DEFAULT_CHALLENGE_CLASS_ORDER_SEED = DEFAULT_DEMO_CLASS_ORDER_SEED
DEFAULT_CHALLENGE_TRAIN_JSON = 'train_.json'
DEFAULT_CHALLENGE_TEST_JSON = 'validation_.json'

DEMO_CLASSIFICATION_FORCED_TRANSFORMS = Resize((224, 224))
DEMO_DETECTION_FORCED_TRANSFORMS = None

CHALLENGE_CLASSIFICATION_FORCED_TRANSFORMS = \
    DEMO_CLASSIFICATION_FORCED_TRANSFORMS
CHALLENGE_DETECTION_FORCED_TRANSFORMS = \
    DEMO_DETECTION_FORCED_TRANSFORMS


DEMO_CLASSIFICATION_EXPERIENCES = 15
DEMO_DETECTION_EXPERIENCES = 4

CHALLENGE_CLASSIFICATION_EXPERIENCES = 15
CHALLENGE_DETECTION_EXPERIENCES = 5


__all__ = [
    'DEFAULT_DEMO_CLASS_ORDER_SEED',
    'DEFAULT_DEMO_TRAIN_JSON',
    'DEFAULT_DEMO_TEST_JSON',
    'DEFAULT_CHALLENGE_CLASS_ORDER_SEED',
    'DEFAULT_CHALLENGE_TRAIN_JSON',
    'DEFAULT_CHALLENGE_TEST_JSON',
    'DEMO_CLASSIFICATION_FORCED_TRANSFORMS',
    'DEMO_DETECTION_FORCED_TRANSFORMS',
    'CHALLENGE_CLASSIFICATION_FORCED_TRANSFORMS',
    'CHALLENGE_DETECTION_FORCED_TRANSFORMS',
    'DEMO_CLASSIFICATION_EXPERIENCES',
    'DEMO_DETECTION_EXPERIENCES',
    'CHALLENGE_CLASSIFICATION_EXPERIENCES',
    'CHALLENGE_DETECTION_EXPERIENCES'
]
