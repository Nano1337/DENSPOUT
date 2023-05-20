import argparse
import yaml
import os

def get_args(): 
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

    parser.add_argument('-c', '--config', type=str, default='cfgs/default.yaml', 
                        metavar='FILE', help='Path to config file')

    parser = argparse.ArgumentParser('DENSPOUT Training script', add_help=False)
    parser.add_argument('--output_dir', type=str, default='/home/yinh4/DENSPOUT/outputs-fabric', help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=999, help='Random seed for reproducibility')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')

    # Model parameters
    parser.add_argument('--model', type=str, default='vanilla_gan', help='Model name')
    parser.add_argument('--nz', type=int, default=100, help='Size of latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='Size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=64, help='Size of feature maps in discriminator')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparameter for Adam optimizers')

    # Data parameters
    parser.add_argument('--dataroot', type=str, default='/home/yinh4/DENSPOUT/data/', help='Root directory for dataset')
    parser.add_argument('--dataset', type=str, default='EURO', help='Dataset name')
    parser.add_argument('--image_size', type=int, default=64, help='Size of images')
    parser.add_argument('--workers', type=int, default=128, help='Number of workers for dataloader')
    parser.add_argument('--nc', type=int, default=3, help='Number of channels in the training images')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--gpus', nargs='+', type=int, default=[1], help='List of GPUs')
    parser.add_argument('--print_every', type=int, default=50, help='Print losses every n iterations')
    parser.add_argument('--ckpt_dir', type=str, default='/home/yinh4/DENSPOUT/ckpt_dir', help='Directory to save checkpoints')
    parser.add_argument('--ckpt_name', type=str, default=None, help='Checkpoint file name')
    parser.add_argument('--display_n', type=int, default=10, help='Number of images to save/display')

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