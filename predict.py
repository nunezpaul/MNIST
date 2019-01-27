import cv2

import tensorflow as tf


def import_graph(name):
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(f'{name}.ckpt.meta')
    new_saver.restore(sess, f'{name}.ckpt')
    graph = tf.get_default_graph()
    return sess, graph


def get_all_collection_keys(graph):
    collection_keys = graph.get_all_collection_keys()
    for key in collection_keys:
        print(key, graph.get_collection(key))


def get_test_data():
    _, test = tf.keras.datasets.mnist.load_data()
    test_imgs, test_labels = test
    return test_imgs, test_labels


def make_prediction(imgs, sess, graph, labels=None):
    use_placeholder = graph.get_tensor_by_name("use_placeholder:0")
    is_training = graph.get_collection('Dropout_switch')[0]
    img_input = graph.get_collection('Inputs')[0]
    # label_input = graph.get_tensor_by_name('label_input:0')

    prediction = graph.get_tensor_by_name('prediction:0')
    sess.run(graph.get_collection('Iterator_init'))
    predictions, is_training = sess.run([prediction, is_training], feed_dict={use_placeholder: True,
                                                  is_training: False,
                                                  img_input: imgs})
    plot_results(imgs, predictions, labels)


def plot_results(imgs, predictions, labels):
    if labels is not None:
        for img, prediction, label in zip(imgs, predictions, labels):
            if prediction != label:
                print(prediction, label)
                cv2.imshow('image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        for img, prediction in zip(imgs, predictions):
            print(prediction)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    num = 2
    model = 'MoS'
    sess, graph = import_graph(f'saved_models/{model}_model_epoch_{num}')
    get_all_collection_keys(graph=graph)
    test_imgs, test_labels = get_test_data()

    make_prediction(imgs=test_imgs, labels=test_labels, sess=sess, graph=graph)