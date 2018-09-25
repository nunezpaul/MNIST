import numpy as np
import tensorflow as tf
import uuid


class ModelParams(object):
    def __init__(self):
        self.img_size = (28, 28)
        self.batch_size = 64
        self.embed_dim = 256
        self.num_classes = 10
        self.id = uuid.uuid4()
        self.name = 'basic_model'
        self.is_training = tf.placeholder_with_default(True, shape=())

    def embed(self, img):
        # Create an embedding based on given image
        flattened = tf.layers.flatten(img)
        layer_1 = tf.layers.dense(flattened, self.embed_dim, activation=tf.nn.relu)
        layer_1_nl = tf.layers.dropout(layer_1, 0.8, training=self.is_training)
        img_embed = tf.layers.dense(layer_1_nl, self.num_classes)

        # Check that the shapes are as we would expect
        assert img.shape[1:] == self.img_size
        assert flattened.shape[1:] == self.img_size[0] * self.img_size[1]
        assert img_embed.shape[1:] == self.num_classes

        return img_embed


class DataConfig(object):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.iter_init = None
        self.next_element = None
        self.create_data_init()

    def create_data_init(self):
        keys = ['train', 'test']

        # Loading processed MNIST train and test data
        data = dict(zip(keys, [self.preprocess_data(data) for data in tf.keras.datasets.mnist.load_data()]))
        assert data['train'][0].shape[1:] == data['test'][0].shape[1:] == (28, 28)

        # Convert numpy data to tf.dataset
        dataset = dict(zip(data.keys(), [self.create_dataset(data[key], self.batch_size) for key in keys]))

        # Creating iterator, data inits and next_element
        iterator = tf.data.Iterator.from_structure(dataset['train'].output_types, dataset['train'].output_shapes)
        self.iter_init = dict(zip(data.keys(), [iterator.make_initializer(dataset[key], name=key) for key in keys]))
        self.next_element = iterator.get_next()

    def preprocess_data(self, data):
        # Converting data to usable data types
        img, lbl = data
        img = img.astype(np.float32) / 255.
        lbl = lbl.astype(np.int64)

        # Checking for correct type and shape
        assert type(data) == tuple
        assert img.shape[1:] == (28, 28)
        assert lbl.shape[1:] == ()

        return img, lbl

    def create_dataset(self, data, batch_size=None):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.repeat().shuffle(buffer_size=batch_size * 1000)
        dataset = dataset.batch(batch_size) if batch_size else dataset
        dataset = dataset.prefetch(batch_size * 4) if batch_size else dataset
        return dataset


class TrainLoss(object):
    def __init__(self):
        self.data_config = DataConfig()
        self.model_params = ModelParams()

    def eval(self):
        metrics = {}

        # Loss will be on the negative log likelihood that the img embed belongs to the correct class
        img, lbl = self.data_config.next_element
        logits = self.model_params.embed(img)

        # Determine the log loss and probability of positive sample
        metrics['Log_loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lbl))
        metrics['Pos_prob'] = tf.exp(-metrics['Log_loss'])

        # Get the accuracy of prediction from logits compared to the label
        prediction = tf.argmax(logits, -1)
        metrics['Accuracy'] = tf.reduce_mean(tf.to_float(tf.equal(prediction, lbl)))

        # Check that shapes are as expected
        assert logits.shape[1:] == self.model_params.num_classes
        assert metrics['Log_loss'].shape == metrics['Pos_prob'].shape == ()
        assert prediction.shape[1:] == ()
        assert metrics['Accuracy'].shape == ()

        return metrics, prediction, lbl


class TrainRun(object):
    def __init__(self, sess, lr=0.001):
        self.train_loss = TrainLoss()
        self.writer = {}
        self.create_writers()
        self.eval_metrics = self.train_loss.eval()
        self.metrics, self.pred, self.lbl = self.eval_metrics
        self.tags = self.metrics.keys()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.metrics['Log_loss'])
        self.initialize(sess)
        self.step = 0

    def create_writers(self):
        phases = ['train', 'test']
        id = self.train_loss.model_params.id
        name = self.train_loss.model_params.name
        for phase in phases:
            tensorboard_dir = './.{name}/{id}/{phase}'.format(name=name, id=id, phase=phase)
            self.writer[phase] = tf.summary.FileWriter(tensorboard_dir, tf.get_default_graph())

    def initialize(self, sess):
        print('Initializing Values: \n{init_vals}'.format(init_vals=sess.run(tf.report_uninitialized_variables())))
        init_op = [tf.global_variables_initializer(), self.train_loss.data_config.iter_init['train']]
        sess.run(init_op)
        print('Finished Initialization.')

    def train_step(self, sess):
        for step in range(60 * 10**3):
            _ = sess.run([self.train_op])
            self.step += 1
            if step % 10**3 == 0:
                print('Evaluating metrics...')
                self.report_metrics(sess)

    def report_metrics(self, sess):
        # Evaluate using training data here
        train_metrics, train_pred, train_lbl = sess.run(
            self.eval_metrics, feed_dict={self.train_loss.model_params.is_training: False})

        # Switch to testing data and get metrics
        sess.run(self.train_loss.data_config.iter_init['test'])
        test_metrics, test_pred, test_lbl = sess.run(
            self.eval_metrics, feed_dict={self.train_loss.model_params.is_training: False})

        # Write metrics to tensorboard
        for tag in self.tags:
            self.tensorboard_logger(self.writer['test'], tag=tag, value=test_metrics[tag])
            self.tensorboard_logger(self.writer['train'], tag=tag, value=train_metrics[tag])
            print('Train {tag}: {train:.3f} \t\t\t Test {tag}: {test:.3f}'.format(
                tag=tag, train=train_metrics[tag], test=test_metrics[tag]))

        print('Train Predict {train} \t Test Predict {test}'.format(
            train=train_pred[:10], test=test_pred[:10]))
        print('Train Correct {train} \t Test Correct {test}'.format(
            train=train_lbl[:10], test=test_lbl[:10]))

        # Switch data back to training data
        sess.run(self.train_loss.data_config.iter_init['train'])

    def tensorboard_logger(self, writer, tag, value):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, self.step)
        writer.flush()


if __name__ == '__main__':
    sess = tf.Session()
    tr = TrainRun(sess)
    tr.train_step(sess)
