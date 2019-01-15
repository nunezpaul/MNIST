import tensorflow as tf

import os
import uuid

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
        self.is_training = tf.placeholder_with_default(True, shape=())

        self.dense1 = tf.layers.Dense(self.embed_dim, activation=tf.nn.relu, name='dense1')
        self.dense2 = tf.layers.Dense(self.num_classes, name='dense2')

    def _forward(self, img, check_shapes=True):
        # Create an embedding based on given image
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
        self.data_init = [train_data.iter_init]
        self.model = model
        self.train_outputs = self.eval(train_data)
        if test_data:
            self.data_init.append(test_data.iter_init)
            self.test_outputs = self.eval(test_data)

    def eval(self, data):
        metrics = {}

        # Loss will be on the negative log likelihood that the img embed belongs to the correct class
        img, label = data.img, data.label
        logits = self.model(img)

        # Determine the log loss and probability of positive sample
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))
        metrics['Log_loss'] = loss
        metrics['Neg_prob'] = 1 - tf.exp(-metrics['Log_loss'])

        # Get the accuracy of prediction from logits compared to the label
        prediction = tf.argmax(logits, -1)
        metrics['Inaccuracy'] = tf.reduce_mean(tf.to_float(tf.not_equal(prediction, label)))

        # Check that shapes are as expected
        assert logits.shape[1:] == self.model.num_classes
        assert prediction.shape[1:] == label.shape[1:]
        assert metrics['Log_loss'].shape == ()
        assert metrics['Neg_prob'].shape == ()
        assert metrics['Inaccuracy'].shape == ()

        return {'loss': loss, 'metrics': metrics, 'prediction': prediction, 'label': label}


class TrainRun(object):
    def __init__(self, model, lr=0.001):
        self.train_loss = TrainLoss(model=model, train_data=TrainData(), test_data=TestData())
        self.writer = {}
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.train_loss.train_outputs['loss'])
        self.step = 0
        self.epochs = 0

    def create_writers(self):
        phases = ['train', 'test']
        base_dir = self.train_loss.model.model_dir
        id = self.train_loss.model.id
        name = self.train_loss.model.name
        for phase in phases:
            tensorboard_dir = f'{base_dir}.{name}/{id}/{phase}'
            self.writer[phase] = tf.summary.FileWriter(tensorboard_dir, tf.get_default_graph())

    def initialize(self, sess):
        self.count_number_trainable_parameters()
        self.create_writers()
        print('Initializing...')
        init_op = [tf.report_uninitialized_variables(),
                   tf.global_variables_initializer()] + self.train_loss.data_init
        output = sess.run(init_op)
        init_vals = output[0]
        print('Initializing Values: \n{init_vals}'.format(init_vals=init_vals))
        print('Finished Initialization!')

    def train(self, sess):
        for step in range(60 * 10**3):
            _ = sess.run([self.train_op])
            self.step += 1
            if step % (60 * 10 ** 3 // 64) == 0:
                self.epochs += 1
                print(f'Evaluating metrics at epoch {self.epochs}...')
                self.report_metrics(sess)

    def report_metrics(self, sess):
        # Evaluate using training data here and switch to testing data
        train_outputs, test_outputs = sess.run([self.train_loss.train_outputs, self.train_loss.test_outputs],
                                               feed_dict={self.train_loss.model.is_training: False})

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
            print(variable)
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print(total_parameters)


if __name__ == '__main__':
    with tf.Session() as sess:
        model = BasicModel()
        tr = TrainRun(model=model)
        tr.initialize(sess)
        tr.train(sess)
