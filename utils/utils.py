from base.meters import BaseMeters
import time
import random
import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary

class Summary:

    def __init__(self, model, device, train_dataset, args,  dev_dataset=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model
        self.hyper_params = vars(args)

        self.model.to(device)
        self.args = args

    def __call__(self):

        print('Model Summary')
        summary(self.model, input_size=(
            3, self.args.image_size, self.args.image_size))

        print('Training Image: {}', len(self.train_dataset))

        if self.dev_dataset:
            print('Validation Image: {}'.format(len(self.dev_dataset)))

        print('Hyper Parameters')

        for key, value in self.hyper_params.items():
            print('{0} : {1}'.format(key, value))


class Loss(BaseMeters):

    def __init__(self):
        super(Loss, self).__init__()


class Timer:

    def __init__(self):
        pass

    def __call__(self, function):

        def wrapper(*args, **kwargs):
            start = time.time()
            result = function(*args, **kwargs)
            end = time.time()
            print('function:%r took: %2.2f sec' %
                  (function.__name__,  end - start))
            return result

        return wrapper


class EarlyStopping:

    def __init__(self, not_improve_step,  verbose=True):

        self.not_improve_step = not_improve_step
        self.verbose = verbose
        self.best_val = 10000
        self.count = 0

    def step(self, val):
        if val <= self.best_val:
            self.best_val = val
            self.count = 0
            return False
        else:
            self.count += 1
            if self.count > self.not_improve_step:
                if self.verbose:
                    print('Performance not Improve after {0}, Early Stopping Execute .......'.format(
                        self.count))
                return True
            else:
                print('Performance not improve, count: {}'.format(self.count))
                return False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)