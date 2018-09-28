import tensorflow as tf

from basic_model import ModelParams, DataConfig, TrainLoss, TrainRun


class ModelParams(ModelParams):
    def __init__(self):
        super(ModelParams, self).__init__()
        self.name = 'MoS'
        self.num_components = 2
        self.mixture_weight = tf.Variable(tf.random_uniform((self.img_size[0] * self.img_size[1], self.num_components)))

    def embed(self, img, noise_factor=0.8):
        # Project flattened img into different spaces
        flat = tf.layers.flatten(img)
        proj_flat = tf.layers.dense(flat, flat.shape[-1] * self.num_components, activation=tf.nn.tanh)
        n_flat = tf.split(proj_flat, self.num_components, axis=-1)
        proj_imgs = [tf.reshape(element, (-1, self.img_size[0], self.img_size[1])) for element in n_flat]

        # Determine the logits from using the basic model
        n_img_embeds = tf.stack([super(ModelParams, self).embed(proj_img) for proj_img in proj_imgs], axis=-1)

        # Calculate component weighting for each sample
        comp_weight = tf.matmul(flat, self.mixture_weight)

        # Combine img_embeds since they have gone through non-linear units
        img_embed = tf.einsum('bcn,bn->bc', n_img_embeds, comp_weight)

        # Check that the shapes are as we would expect
        assert img.shape[1:] == self.img_size
        assert flat.shape[1:] == self.img_size[0] * self.img_size[1]
        assert proj_flat.shape[1:] == self.img_size[0] * self.img_size[1] * self.num_components
        assert n_flat[0].shape[1:] == n_flat[1].shape[1:] == (self.img_size[0] * self.img_size[1])
        assert comp_weight.shape[1:] == self.num_components
        assert n_img_embeds.shape[1:] == (self.num_classes, self.num_components)
        assert img_embed.shape[1:] == self.num_classes

        return img_embed


class TrainLoss(TrainLoss):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.model_params = ModelParams()
        print(dir(self.model_params))
        print(dir(self))


class TrainRun(TrainRun):
    def __init__(self, lr=0.001):
        super(TrainRun, self).__init__(lr)
        self.train_loss = TrainLoss()
        self.writer = {}
        self.create_writers()
        self.eval_metrics = self.train_loss.eval()
        self.metrics, self.pred, self.lbl = self.eval_metrics
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.metrics['Log_loss'])
        self.step = 0


if __name__ == '__main__':
    tr = TrainRun()
    sess = tf.Session()
    tr.initialize(sess)
    print(tr.train_loss.model_params.name)
    tr.train(sess)
