import os

from utils import TrainOptions
from train import Trainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.device_id
    trainer = Trainer(options)
    trainer.train()
