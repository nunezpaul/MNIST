import tensorflow as tf

from basic_model import BasicModel, TrainLoss, TrainRun
from config import Config


class MOSModel(BasicModel):
    def __init__(self):
        super(MOSModel, self).__init__()
        self.num_components = 3
        self.name = 'MoS'
        self.dense_mixture_weight = tf.layers.Dense(self.num_components, name='mixture_weight')
        self.dense_projected = tf.layers.Dense(self.flat_size * self.num_components, activation=tf.nn.tanh, name='proj')

    def _forward(self, img):
        # Project flattened img into different spaces
        flat = tf.layers.flatten(img)
        proj_flat = self.dense_projected(flat)
        n_flat = tf.split(proj_flat, self.num_components, axis=-1)
        proj_imgs = [tf.reshape(element, (-1, self.img_size[0], self.img_size[1])) for element in n_flat]

        # Determine the logits from using the basic model
        n_img_embeds = tf.stack([super(MOSModel, self)._forward(proj_img) for proj_img in proj_imgs], axis=-1)

        # Calculate component weighting for each sample (also known as gating function)
        comp_logits = self.dense_mixture_weight(flat)
        normalized_comp_logits = comp_logits - tf.expand_dims(tf.reduce_max(comp_logits, axis=-1), axis=-1)
        comp_weight = tf.nn.softmax(normalized_comp_logits)

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

        self.comp_weight = comp_weight
        return img_embed


class TrainLoss(TrainLoss):
    def __init__(self, model, train_data, test_data=None):
        super(TrainLoss, self).__init__(self, model, train_data, test_data=None)

    def eval(self, img, label):
        metrics = {}

        # Loss will be on the negative log likelihood that the img embed belongs to the correct class
        logits = self.model(img)

        # Determine the output of the component weights
        avg_comp_weight = tf.reduce_mean(self.model.comp_weight, axis=0)
        for i in range(self.model.num_components):
            metrics['Comp_weight_{num}'.format(num=i)] = avg_comp_weight[i]

        # Determine the log loss and probability of positive sample
        metrics['Log_loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lbl))
        metrics['Neg_prob'] = 1 - tf.exp(-metrics['Log_loss'])

        # Get the accuracy of prediction from logits compared to the label
        prediction = tf.argmax(logits, -1)
        metrics['Inaccuracy'] = tf.reduce_mean(tf.to_float(tf.not_equal(prediction, lbl)))

        # Check that shapes are as expected
        assert logits.shape[1:] == self.model.num_classes
        assert prediction.shape[1:] == label.shape[1:]
        assert metrics['Log_loss'].shape == ()
        assert metrics['Neg_prob'].shape == ()
        assert metrics['Inaccuracy'].shape == ()

        return metrics, prediction, label


class TrainRun(TrainRun):
    def __init__(self, model, sess, load_dir, lr=0.001):
        super(TrainRun, self).__init__(model, sess, load_dir, lr)


if __name__ == '__main__':
    config = Config()
    with tf.Session() as sess:
        model = MOSModel()
        tr = TrainRun(model=model, sess=sess, load_dir=config.params['load_dir'], lr=config.params['lr'])
        tr.train(sess, save_dir=config.params['save_dir'])
