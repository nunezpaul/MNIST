import tensorflow as tf

from basic_model import ModelParams, DataConfig, TrainLoss, TrainRun

class ModelParams(ModelParams):
    def __init__(self):
        super(ModelParams, self).__init__()
        self.name = 'autoencoder_model'

    def embed(self, img):
        # Create an embedding based on given image
        flattened = tf.layers.flatten(img)
        layer_1 = tf.layers.dense(flattened, self.embed_dim, activation=tf.nn.relu)
        bottle_neck = tf.layers.dropout(layer_1, 0.8, training=self.is_training)
        img_embed = tf.layers.dense(bottle_neck, self.num_classes)

        # Auto-encoding layer
        print(dir(flattened.shape[1:].as_list()))
        reconstruction = tf.layers.dense(bottle_neck, flattened.shape.as_list()[1])
        print(reconstruction)

        # Check that the shapes are as we would expect
        assert img.shape[1:] == self.img_size
        assert flattened.shape[1:] == self.img_size[0] * self.img_size[1]
        assert img_embed.shape[1:] == self.num_classes

        return img_embed, reconstruction

mp = ModelParams()
img = tf.placeholder(tf.float32, (64, 28, 28))
result = mp.embed(img)
print(result)
