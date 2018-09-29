import tensorflow as tf

from basic_model import ModelParams, DataConfig, TrainLoss, TrainRun


class ModelParams(ModelParams):
    def __init__(self):
        super(ModelParams, self).__init__()
        self.name = 'CNN'
        self.filter1 = 32
        self.filter2 = 64
        self.kernel_size = (5,5)
        self.pool_size=(2,2)
        self.pool_strides=2
        self.img_size = list(self.img_size)

    def embed(self, img):
        # Reshape the image to include a channel
        img_expanded = tf.expand_dims(img, axis=-1)

        # First convolution layer and pooling
        conv1 = tf.layers.conv2d(
            img_expanded,
            filters=self.filter1,
            kernel_size=self.kernel_size,
            padding='same',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=self.pool_size, strides=self.pool_strides)

        # Second convolution layer and pooling
        conv2 = tf.layers.conv2d(
            pool1,
            filters=self.filter2,
            kernel_size=self.kernel_size,
            padding='same',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=self.pool_size, strides=self.pool_strides)

        # Give pool2 to original embedding function of basic_model
        img_embed = super(ModelParams, self).embed(img=pool2, check_shapes=False)

        # Check that the shapes are as we would expect
        assert img.shape[1:] == self.img_size
        assert img_expanded.shape[1:] == self.img_size + [1]
        assert conv1.shape[1:] == self.img_size + [self.filter1]
        assert pool1.shape[1:] == (14, 14, self.filter1)
        assert conv2.shape[1:] == (14, 14, self.filter2)
        assert pool2.shape[1:] == (7, 7, self.filter2)
        assert img_embed.shape[1:] == self.num_classes

        return img_embed


class TrainLoss(TrainLoss):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.model_params = ModelParams()


class TrainRun(TrainRun):
    def __init__(self, lr=0.001):
        super(TrainRun, self).__init__(lr)
        self.train_loss = TrainLoss()


if __name__ == '__main__':
    tr = TrainRun()
    sess = tf.Session()
    tr.initialize(sess)
    tr.train(sess)
