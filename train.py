"""
DCGAN - Accelerated with Lightning Fabric

Code adapted from the official PyTorch DCGAN tutorial:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils
from torchvision.datasets import STL10
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

    # # process dataset download
    # dataset = STL10(
    #     root=args.dataroot,
    #     split="train",
    #     download=True,
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize(args.image_size),
    #             transforms.CenterCrop(args.image_size),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712)),
    #         ]
    #     ),
    # )

    # Create the dataset
    dataset = dataset_utils.get_dataset(args.dataset, args)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Create output directory
    output_dir = Path("outputs-fabric", time.strftime("%Y%m%d-%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot some training images from modality A
    real_batch = next(iter(dataloader))
    num_display = min(args.batch_size, args.display_n)
    torchvision.utils.save_image(
        real_batch['A'][:num_display],
        output_dir / "sample-data-A.png",
        padding=2,
        normalize=True,
    )

    # Plot some training images from modality A
    real_batch = next(iter(dataloader))
    torchvision.utils.save_image(
        real_batch['B'][:num_display],
        output_dir / "sample-data-B.png",
        padding=2,
        normalize=True,
    )

    # Create models
    generator = models.get_model("vanilla_gan", "generator", args)
    discriminator = models.get_model("vanilla_gan", "discriminator", args)
    
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=fabric.device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Set up Adam optimizers for both G and D
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    discriminator, optimizer_d = fabric.setup(discriminator, optimizer_d)
    generator, optimizer_g = fabric.setup(generator, optimizer_g)
    dataloader = fabric.setup_dataloaders(dataloader)

    # Lists to keep track of progress
    losses_g = []
    losses_d = []
    iteration = 0

    # Check if args.ckpt_dir exists and throw error if it doesnt
    if not os.path.isdir(args.ckpt_dir):
        raise ValueError("args.ckpt_dir does not exist")

    # Load checkpoint if it exists 
    if args.ckpt_name and os.path.isfile(os.path.join(args.ckpt_dir, args.ckpt_name)):
        loaded_epoch, losses_g, losses_d = load_checkpoint(fabric, os.path.join(args.ckpt_dir, args.ckpt_name), 
                                                            generator, discriminator, optimizer_g, optimizer_d)
        print("Loaded checkpoint at epoch {}".format(loaded_epoch))
    else:
        loaded_epoch = 0

    # Training loop
    for epoch in range(loaded_epoch, loaded_epoch + args.num_epochs):
        for i, data in enumerate(dataloader, 0):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) to train discriminator
            # (a) Train with all-real batch
            discriminator.zero_grad() # zero all gradients for learnable weights of discriminator
            real = data[0] # take first data tuple (image) from batch
            b_size = real.size(0) # get batch size
            label = torch.full((b_size,), real_label, dtype=torch.float, device=fabric.device) # create tensor of ones since data is all real images
            # Forward pass real batch through D
            output = discriminator(real).view(-1) # flatten output tensor
            # Calculate loss on all-real batch
            err_d_real = criterion(output, label)
            # Calculate gradients for D in backprop
            fabric.backward(err_d_real) 
            d_x = output.mean().item()

            # (b) Train with all-fake batch to train generator
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.nz, 1, 1, device=fabric.device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)  # detach gradients from computation graph otherwise we will backpropagate to generator
                                                            # Purpose is that we don't want to update gneerator's parameters based on discriminator 
                                                            # Ensures that generator is updated by optimizing on its own loss
            # Calculate D's loss on the all-fake batch
            err_d_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            fabric.backward(err_d_fake)
            d_g_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            err_d = err_d_real + err_d_fake
            # Update D, optimizer uses computed gradients to adjust parameters of discriminator
            optimizer_d.step()

            # (2) Update G network: maximize log(D(G(z)))
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            err_g = criterion(output, label)
            # Calculate gradients for G
            fabric.backward(err_g)
            d_g_z2 = output.mean().item()
            # Update G
            optimizer_g.step()

            # Output training stats
            if i % args.print_every == 0:
                fabric.print(
                    f"[{epoch}/{loaded_epoch + args.num_epochs}][{i}/{len(dataloader)}]\t"
                    f"Loss_D: {err_d.item():.4f}\t"
                    f"Loss_G: {err_g.item():.4f}\t"
                    f"D(x): {d_x:.4f}\t"
                    f"D(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}"
                )

            # Save Losses for plotting later
            losses_g.append(err_g.item())
            losses_d.append(err_d.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iteration % args.save_every == 0) or ((epoch == args.num_epochs + loaded_epoch - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()

                if fabric.is_global_zero:
                    print("Savings images at epoch {} and iteration {}".format(epoch, iteration))
                    torchvision.utils.save_image(
                        fake,
                        output_dir / f"fake-e{epoch:04d}-it{iteration:04d}.png",
                        padding=2,
                        normalize=True,
                    )

                fabric.barrier() # same as torch.cuda.synchronize()

            iteration += 1

        if fabric.is_global_zero:
            # TODO: save checkpoints smarter using IMD for convergence detection
            print("Saving checkpoint at epoch {}".format(epoch))
            save_name = "{}-{}.pth".format(args.model, epoch)
            save_checkpoint(fabric, generator, discriminator, optimizer_g, optimizer_d, epoch, losses_g, losses_d, os.path.join(args.ckpt_dir, save_name))


if __name__ == "__main__":
    args = get_args()
    if args.output_dir: 
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)