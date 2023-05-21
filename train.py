"""
DCGAN - Accelerated with Lightning Fabric

Code adapted from the official PyTorch DCGAN tutorial:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import os
import time
from pathlib import Path

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
from lightning.fabric import Fabric, seed_everything

import models
import dataset_utils
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.opt import get_args

torch.set_float32_matmul_precision('medium')

# Future work: add get_model method that is modular and customizable from opt


def main(args):
    # Set random seed for reproducibility
    seed_everything(args.seed)

    fabric = Fabric(accelerator="auto", devices=args.gpus)
    fabric.launch()

    # Create the dataset and dataloader
    dataset, dataloader = dataset_utils.get_dataloader(args.dataset, args)
    dataloader = fabric.setup_dataloaders(dataloader)
    dataset_size = len(dataset)
    print("Dataset size: {}".format(dataset_size))

    # Create output directory
    unique_dir_name = time.strftime("%Y%m%d-%H%M%S") + "-" + args.model
    output_dir = Path("outputs-fabric", unique_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot some training images from modality A
    real_batch = next(iter(dataloader))
    num_display = min(args.batch_size, args.display_n)
    vutils.save_image(
        real_batch['A'][:num_display],
        output_dir / "sample-data-A.png",
        padding=2,
        normalize=True,
    )

    # Plot some training images from modality A
    real_batch = next(iter(dataloader))
    vutils.save_image(
        real_batch['B'][:num_display],
        output_dir / "sample-data-B.png",
        padding=2,
        normalize=True,
    )

    # Get the model
    model = models.get_model(args.model, fabric, args)

    # Load checkpoint if exists
    if not os.path.isdir(args.ckpt_dir):
        raise ValueError("args.ckpt_dir does not exist")
    if args.ckpt_full_path and os.path.isfile(args.ckpt_full_path):
        loaded_epoch = model.load_networks()
        print("Loaded checkpoint at epoch {}".format(loaded_epoch))
    else:
        loaded_epoch = 0

    # Create checkpoint directory
    save_dir = Path(args.ckpt_dir, unique_dir_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.print_networks()

    total_iters = 0  # the total number of training iterations
    optimize_time = 0.1 # set initial value to 0.1 for smoothing

    for epoch in range(loaded_epoch, loaded_epoch + args.num_epochs + 1):
        epoch_start_time = time.time() # start timer for epoch
        iter_data_time = time.time() # start timer for data loading per iteration
        epoch_iter = 0 # start epoch iteration counter, will be reset to 0 at end of epoch
        model.set_epoch(epoch) 

        for i, data in enumerate(dataloader, 0):
            iter_start_time = time.time() # start timer for computation each iteration

            total_iters += args.batch_size
            epoch_iter += args.batch_size
            fabric.barrier()

            optimize_start_time = time.time()
            if i == 0: 
                model.data_dependent_initialize(data)

            # forward and backward passes
            model.set_input(data)
            model.optimize_parameters() # calculate losses, gradients, and update network weights; called in every training iteration
            fabric.barrier() # sync all processes before moving on to next iteration
            optimize_time = (time.time() - optimize_start_time) / args.batch_size * 0.005 + 0.995 * optimize_time

            # Output training stats # TODO: abstract this
            if i % args.print_every == 0:
                fabric.print(
                    f"[{epoch}/{loaded_epoch + args.num_epochs}][{i}/{len(dataloader)}]\t"
                    f"Loss_G: {model.loss_g.item():.4f}\t"
                    f"Loss_D_real: {model.loss_d_real.item():.4f}\t"
                    f"Loss_D_fake: {model.loss_d_fake.item():.4f}\t"
                    f"Loss_idt: {model.loss_idt.item():.4f}\t"
                    f"time: {optimize_time:.4f}\t"
                    f"data: {(iter_start_time - iter_data_time):.4f}\t"
                )
            
            iter_data_time = time.time() # end timer for data loading per iteration

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, loaded_epoch + args.num_epochs, time.time() - epoch_start_time))
        
        # Visualize training results every epoch
        visuals = model.get_current_visuals()

        if fabric.is_global_zero:
            print("Savings images at epoch {}".format(epoch))

            grids = []

            for _, v in visuals.items():
                grid = vutils.make_grid(v, nrow=min(args.batch_size, args.display_n), padding=2, normalize=True)
                grids.append(grid)
            
            # concat all grids vertically
            grid = torch.cat(grids, dim=1)

            vutils.save_image(
                grid,
                output_dir / f"fake-e{epoch:04d}.png",
                padding=2,
                normalize=True,
            )

        fabric.barrier() # same as torch.cuda.synchronize()

        if fabric.is_global_zero:
            # TODO: save checkpoints smarter using IMD for convergence detection
            print("Saving checkpoint at epoch {}".format(epoch))
            save_name = "{}-{}.pth".format(args.model, epoch)
            model.save_networks(os.path.join(save_dir, "latest.pth"))
            model.save_networks(os.path.join(save_dir, save_name))



if __name__ == "__main__":
    args = get_args()
    if args.output_dir: 
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)