import argparse
import yaml

def get_args(): 
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

    parser.add_argument('-c', '--config', type=str, default='cfgs/default.yaml', 
                        metavar='FILE', help='Path to config file')

    parser = argparse.ArgumentParser('DENSPOUT Training script', add_help=False)
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')


    # Model parameters

    # Optimizer parameters

    # Dataset parameters
    parser.add_argument('--dataroot', type=str, default='/home/yinh4/DENSPOUT/data/', help='Root directory for dataset')
    parser.add_argument('--image_size', type=int, default=64, help='Size of images')

    # more parameters here

    # Do we have a config file to parse? 
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config: 
        with open(args_config.config, 'r') as f:
            config = yaml.safe_load(f)
        parser.set_defaults(**config)

    # the main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified
    args = parser.parse_args(remaining)

    return args