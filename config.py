import argparse

class Config(object):
    def __init__(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(description='Parameters for training model.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='set the learning rate')
        parser.add_argument('--load_dir', type=str, default=None,
                            help='file path to saved checkpoint file to load.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout percent to keep during training.')
        parser.add_argument('--save_dir', type=str, default='saved_models',
                            help='Where to save the trained model.')
        parser.add_argument('--epochs', type=int, default=100,
                            help='How many epochs to train the model for.')

        self.parser = parser
        self.params = vars(parser.parse_args())


if __name__ == '__main__':
    config = Config()
    for key, val in config.params.items():
        print(key, val)
