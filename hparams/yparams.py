""" Yaml file parser derivative of tensorflow.contrib.training.HParams """
""" Original code: https://hanxiao.github.io/2017/12/21/Use-HParams-and-YAML-to-Better-Manage-Hyperparameters-in-Tensorflow/ """

from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams

class YParams(HParams):
    """ Yaml file parser derivative of HParams """

    def __init__(self, yaml_fn, config_name, print_params=True):

        super(YParams, self).__init__()
        with open(yaml_fn) as yamlfile:
            for key, val in YAML().load(yamlfile)[config_name].items():
                if print_params:
                    print(key, val)
                self.add_hparam(key, val)
