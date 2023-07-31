import configargparse
from magicpony import setup_runtime, Trainer, MagicPony

# runtime arguments
parser = configargparse.ArgumentParser(description='Training configurations.')
parser.add_argument('-c', '--config', default="config/train_multi_seq_dev.yml", type=str, is_config_file=True, help='Specify a config file path')
parser.add_argument('--gpu', default='0', type=str, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
args, _ = parser.parse_known_args()

# set up
cfgs = setup_runtime(args)
trainer = Trainer(cfgs, MagicPony)
run_train = cfgs.get('run_train', True)
run_test = cfgs.get('run_test', False)

# run
if run_train:
    trainer.train()
if run_test:
    trainer.test()
