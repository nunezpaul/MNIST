import keras as k
import numpy as np
import tensorflow as tf


class ModelParams(object):
    def __init__(self):
        self.embed_dim = 256
        self.digits_embedding = k.layers.Embedding(input_dim=10, output_dim=self.digits_embedding)



class TrainLoss(object):
    def __init__(self, ModelParams):
        self.model_params = ModelParams

    def get_probability(self, positive):
        return

class TrainRun(object):
    def __init__(self):
        pass

