import tensorflow as tf

from basic_model import ModelParams, DataConfig, TrainLoss, TrainRun


class ModelParams(ModelParams):
    def __init__(self):
        super(ModelParams, self).__init__()
        self.name = 'autoencoder'

        self.dense_decode = tf.layers.Dense(self.flat_size)

    def embed(self, img, noise_factor=0.8):
        # Adding noise to the img
        noise = tf.truncated_normal(shape=tf.shape(img), mean=0.0, stddev=0.5) * tf.to_float(self.is_training)
        noisy_img = tf.clip_by_value(img + noise_factor * noise, clip_value_min=0.0, clip_value_max=1.0)

        # Create an embedding based on given image
        flattened = tf.layers.flatten(noisy_img)
        layer_1 = self.dense1(flattened)
        bottle_neck = tf.layers.dropout(layer_1, 0.8, training=self.is_training)
        img_embed = self.dense2(bottle_neck)

        # Decoding layer
        decode = self.dense_decode(bottle_neck)

        # Check that the shapes are as we would expect
        assert img.shape[1:] == self.img_size
        assert flattened.shape[1:] == self.img_size[0] * self.img_size[1]
        assert img_embed.shape[1:] == self.num_classes

        return img_embed, decode


class TrainLoss(TrainLoss):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.model_params = ModelParams()

    def eval(self):
        metrics = {}
        # Loss will be on the negative log likelihood that the img embed belongs to the correct class
        img, lbl = self.data_config.next_element
        logits, decode_img = self.model_params.embed(img)

        # Determine the log loss and probability of positive sample
        metrics['Log_loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lbl))
        metrics['Neg_prob'] = 1 - tf.exp(-metrics['Log_loss'])

        # Determine decode loss
        metrics['Decode_loss'] = tf.losses.mean_squared_error(tf.layers.flatten(img), decode_img)

        # Get the accuracy of prediction from logits compared to the label
        prediction = tf.argmax(logits, -1)
        metrics['Inaccuracy'] = tf.reduce_mean(tf.to_float(tf.not_equal(prediction, lbl)))

        # Check that shapes are as expected
        assert logits.shape[1:] == self.model_params.num_classes
        assert metrics['Log_loss'].shape == ()
        assert metrics['Neg_prob'].shape == ()
        assert metrics['Inaccuracy'].shape == ()
        assert metrics['Decode_loss'].shape == ()
        assert prediction.shape[1:] == lbl.shape[1:]

        return metrics, prediction, lbl


class TrainRun(TrainRun):
    def __init__(self, lr=0.001):
        super(TrainRun, self).__init__(lr)
        self.train_loss = TrainLoss()
        self.eval_metrics = self.train_loss.eval()
        self.metrics, self.pred, self.lbl = self.eval_metrics
        self.metrics['Total_loss'] = self.metrics['Decode_loss'] + self.metrics['Log_loss']
        self.train_op = self.optimizer.minimize(self.metrics['Total_loss'])


if __name__ == '__main__':
    tr = TrainRun()
    sess = tf.Session()
    tr.initialize(sess)
    tr.train(sess)
