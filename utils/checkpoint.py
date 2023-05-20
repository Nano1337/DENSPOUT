def save_checkpoint(fabric, generator, discriminator, optimizer_g, optimizer_d, epoch, losses_g, losses_d, path):
    state = {
        "net_g": generator,
        "net_d": discriminator,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d,
        "epoch": epoch,
        "losses_g": losses_g,
        "losses_d": losses_d
    }
    fabric.save(path, state)

def load_checkpoint(fabric, path, generator, discriminator, optimizer_g, optimizer_d):
    state = { 
        "net_g": generator, # fabric API loads these in-place
        "net_d": discriminator,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d
    }
    remainder = fabric.load(path, state)
    return remainder["epoch"], remainder["losses_g"], remainder["losses_d"]
