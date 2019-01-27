import tensorflow as tf

from basic_model import BasicModel, TrainLoss, TrainRun
from config import Config


class CNNModel(BasicModel):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.name = 'CNN'
        self.filter1 = 32
        self.filter2 = 64
        self.kernel_size = (5, 5)
        self.pool_size = (2, 2)
        self.pool_strides = 2
        self.img_size = list(self.img_size)

        # Conv layers
        self.conv1 = tf.layers.Conv2D(filters=self.filter1,
                                      kernel_size=self.kernel_size,
                                      padding='same',
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(filters=self.filter2,
                                      kernel_size=self.kernel_size,
                                      padding='same',
                                      activation=tf.nn.relu)

    def _forward(self, img):
        # Reshape the image to include a channel
        img_expanded = tf.expand_dims(img, axis=-1)

        # First convolution and pool layer
        with tf.name_scope('Conv1') as scope:
            conv1 = self.conv1(img_expanded)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=self.pool_size, strides=self.pool_strides)

        # Second convolution layer and pooling
        with tf.name_scope('Conv2') as scope:
            conv2 = self.conv2(pool1)
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=self.pool_size, strides=self.pool_strides)

        # Give pool2 to original embedding function of basic_model
        img_embed = super(CNNModel, self)._forward(img=pool2, check_shapes=False)

        # Check that the shapes are as we would expect
        assert img.shape[1:] == self.img_size
        assert img_expanded.shape[1:] == self.img_size + [1]
        assert conv1.shape[1:] == self.img_size + [self.filter1]
        assert pool1.shape[1:] == (14, 14, self.filter1)
        assert conv2.shape[1:] == (14, 14, self.filter2)
        assert pool2.shape[1:] == (7, 7, self.filter2)
        assert img_embed.shape[1:] == self.num_classes

        return img_embed


if __name__ == '__main__':
    config = Config()
    with tf.Session() as sess:
        model = CNNModel()
        tr = TrainRun(model=model, sess=sess, load_dir=config.params['load_dir'], lr=config.params['lr'])
        tr.train(sess, save_dir=config.params['save_dir'])
