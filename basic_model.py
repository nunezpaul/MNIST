import tensorflow as tf

import os
import uuid

from config import Config
from data_config import TestData, TrainData


class BasicModel(object):
    def __call__(self, img, *args, **kwargs):
        return self._forward(img)

    def __init__(self):
        self.img_size = (28, 28)
        self.flat_size = self.img_size[0] * self.img_size[1]
        self.embed_dim = 256
        self.num_classes = 10
        self.id = uuid.uuid4()
        self.name = 'basic'
        self.model_dir = os.path.abspath(__file__).split(os.path.basename(__file__))[0]
        self.is_training = tf.placeholder_with_default(True, shape=(), name='is_training')
        tf.add_to_collection('Dropout_switch', self.is_training)

        self.dense1 = tf.layers.Dense(self.embed_dim, activation=tf.nn.relu, name='dense1')
        self.dense2 = tf.layers.Dense(self.num_classes, name='dense2')

    def _forward(self, img, check_shapes=True):
        # Create an embedding based on given image
        with tf.name_scope('model',) as scope:
            flattened = tf.layers.flatten(img)
            layer_1 = self.dense1(flattened)
            layer_1_nl = tf.layers.dropout(layer_1, 0.8, training=self.is_training)
            img_embed = self.dense2(layer_1_nl)

        # Check that the shapes are as we would expect
        if check_shapes:
            assert img.shape[1:] == self.img_size
            assert flattened.shape[1:] == self.img_size[0] * self.img_size[1]
            assert img_embed.shape[1:] == self.num_classes

        return img_embed


class TrainLoss(object):
    def __init__(self, model, train_data, test_data=None):
        self.data_init = [train_data.iter_init] + ([test_data.iter_init] if test_data is not None else [])
        self.model = model
        self.use_train_data = tf.placeholder_with_default(True, shape=(), name='use_train_data')
        self.use_placeholder = tf.placeholder_with_default(False, shape=(), name='use_placeholder')
        self.outputs = self.eval(*self.create_data_switches(train_data, test_data))
        tf.add_to_collection('Placeholder_switch', self.use_placeholder)

    def create_data_switches(self, train_data, test_data):
        img_ph, label_ph = self.create_placeholders(train_data)

        # Overwrite is_training in order to switch where the data is coming from
        if test_data is not None:
            img, label = tf.cond(self.use_placeholder,
                                 lambda: [img_ph, label_ph],
                                 lambda: self.use_datasets(train_data, test_data))
        else:
            img, label = tf.cond(self.use_placeholder,
                                 lambda: [img_ph, label_ph],
                                 lambda: [train_data.img, train_data.label])

        return img, label

    def eval(self, img, label):
        metrics = {}
        # Loss will be on the negative log likelihood that the img embed belongs to the correct class
        logits = self.model(img)

        # Determine the log loss and probability of positive sample
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))
        metrics['Log_loss'] = loss
        metrics['Neg_prob'] = 1 - tf.exp(-metrics['Log_loss'])

        # Get the accuracy of prediction from logits compared to the label
        prediction = tf.argmax(logits, -1, name='prediction')
        metrics['Inaccuracy'] = tf.reduce_mean(tf.to_float(tf.not_equal(prediction, label)))

        # Check that shapes are as expected
        assert logits.shape[1:] == self.model.num_classes
        assert prediction.shape[1:] == label.shape[1:]
        assert metrics['Log_loss'].shape == ()
        assert metrics['Neg_prob'].shape == ()
        assert metrics['Inaccuracy'].shape == ()

        return {'loss': loss, 'metrics': metrics, 'prediction': prediction, 'label': label}

    def use_datasets(self, train_data, test_data):
        img, label = tf.cond(self.use_train_data,
                             lambda: [train_data.img, train_data.label],
                             lambda: [test_data.img, test_data.label])
        return [img, label]

    def create_placeholders(self, dataset):
        img_ph = tf.placeholder_with_default(tf.cast(tf.random_uniform(shape=[64, 28, 28], maxval=255), tf.uint8),
                                             name='img_input', shape=[None, 28, 28])
        label_ph = tf.placeholder_with_default(tf.cast(tf.random_uniform(shape=[64, ], maxval=255), tf.uint8),
                                               name='label_input', shape=[None, ])

        tf.add_to_collection(name='Inputs', value=img_ph)
        tf.add_to_collection(name='Inputs', value=label_ph)

        img_ph = dataset._img_preprocessing(img_ph)
        label_ph = dataset._label_preprocessing(label_ph)

        return [img_ph, label_ph]


class TrainRun(object):
    def __init__(self, model, sess, load_dir, lr=0.001):
        self.train_loss = TrainLoss(model=model, train_data=TrainData(), test_data=TestData())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.train_loss.outputs['loss'])
        self.saver = tf.train.Saver()
        self.writer = {}
        self.step = 0
        self.epochs = 0
        self.initialize(sess, load_dir)

    def create_writers(self):
        phases = ['train', 'test']
        base_dir = self.train_loss.model.model_dir
        id = self.train_loss.model.id
        name = self.train_loss.model.name
        for phase in phases:
            tensorboard_dir = f'{base_dir}/logs/{name}/{id}/{phase}'
            self.writer[phase] = tf.summary.FileWriter(tensorboard_dir, tf.get_default_graph())

    def initialize(self, sess, load_dir=None):
        self.count_number_trainable_parameters()
        self.create_writers()
        print('Initializing...')
        init_op = self.train_loss.data_init + [tf.report_uninitialized_variables()]
        init_op += [] if load_dir else [tf.global_variables_initializer()]
        if load_dir:
            print('Reloading model...')
            self.saver.restore(sess, load_dir)
            print('Model restored!')
        output = sess.run(init_op)
        init_vals = output[-1]
        print('Initialized Values: \n{init_vals}'.format(init_vals=init_vals))

    def train(self, sess, save_dir='saved_models'):
        try:
            for step in range(60 * 10**3):
                _ = sess.run([self.train_op])
                self.step += 1
                if step % (60 * 10 ** 3 // 64) == 0:
                    self.epochs += 1
                    print(f'Evaluating metrics at epoch {self.epochs}...')
                    self.report_metrics(sess)
        except KeyboardInterrupt:
            print('Saving graph params!')
            self.saver.save(sess, f"{save_dir}/{self.train_loss.model.name}_model_epoch_{self.epochs}.ckpt")
            print('Saved!')

    def report_metrics(self, sess):
        # Evaluate using training data here and switch to testing data
        train_outputs = sess.run(self.train_loss.outputs,
                                 feed_dict={self.train_loss.model.is_training: False})
        test_outputs = sess.run(self.train_loss.outputs,
                                feed_dict={self.train_loss.model.is_training: False,
                                           self.train_loss.use_train_data: False})
        # Write metrics to tensorboard
        for key in train_outputs['metrics'].keys():
            self.tensorboard_logger(self.writer['test'], tag=key, value=test_outputs['metrics'][key])
            self.tensorboard_logger(self.writer['train'], tag=key, value=train_outputs['metrics'][key])
            print('Train {tag}: {train:.3f} \t\t\t Test {tag}: {test:.3f}'.format(
                tag=key, train=train_outputs['metrics'][key], test=test_outputs['metrics'][key]))

        print('Train Predict {train} \t Test Predict {test}'.format(
            train=train_outputs['prediction'][:10], test=test_outputs['prediction'][:10]))
        print('Train Correct {train} \t Test Correct {test}'.format(
            train=train_outputs['label'][:10], test=test_outputs['label'][:10]))

    def tensorboard_logger(self, writer, tag, value):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, self.step)
        writer.flush()

    def count_number_trainable_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print(variable.name, variable_parameters)
            total_parameters += variable_parameters
        print(total_parameters)


if __name__ == '__main__':
    config = Config()
    with tf.Session() as sess:
        model = BasicModel()
        tr = TrainRun(model=model, sess=sess, load_dir=config.params['load_dir'], lr=config.params['lr'])
        tr.train(sess, save_dir=config.params['save_dir'])
