def save_checkpoint(fabric, generator, discriminator, optimizer_g, optimizer_d, epoch, losses_g, losses_d, path):
    state = {
        "generator": generator,
        "discriminator": discriminator,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d,
        "epoch": epoch,
        "losses_g": losses_g,
        "losses_d": losses_d
    }
    fabric.save(path, state)

def load_checkpoint(fabric, path, generator, discriminator, optimizer_g, optimizer_d):
    state = { 
        "generator": generator, # fabric API loads these in-place
        "discriminator": discriminator,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d
    }
    remainder = fabric.load(path, state)
    return remainder["epoch"], remainder["losses_g"], remainder["losses_d"]
