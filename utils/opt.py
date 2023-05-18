import argparse
import yaml
import os

def get_args(): 
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

    parser.add_argument('-c', '--config', type=str, default='cfgs/default.yaml', 
                        metavar='FILE', help='Path to config file')

    parser = argparse.ArgumentParser('DENSPOUT Training script', add_help=False)

    # Model parameters
    parser.add_argument('--model', type=str, default='vanilla_gan', help='Model name')
    parser.add_argument('--nz', type=int, default=100, help='Size of latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='Size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=64, help='Size of feature maps in discriminator')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparameter for Adam optimizers')

    # data parameters
    parser.add_argument('--dataroot', type=str, default='/home/yinh4/DENSPOUT/data/', help='Root directory for dataset')
    parser.add_argument('--image_size', type=int, default=64, help='Size of images')
    parser.add_argument('--workers', type=int, default=128, help='Number of workers for dataloader')
    parser.add_argument('--nc', type=int, default=3, help='Number of channels in the training images')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for training')
    parser.add_argument('--print_every', type=int, default=50, help='Print losses every n iterations')
    parser.add_argument('--save_every', type=int, default=500, help='Save checkpoints every n iterations')

    # Do we have a config file to parse? 
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config: 
        with open(args_config.config, 'r') as f:
            config = yaml.safe_load(f)
        parser.set_defaults(**config)

    # the main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified
    args = parser.parse_args(remaining)

    args.workers = os.cpu_count()

    return args