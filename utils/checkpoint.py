def save_checkpoint(fabric, generator, discriminator, optimizer_g, optimizer_d, iteration, losses_g, losses_d, path):
    state = {
        "generator": generator,
        "discriminator": discriminator,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d,
        "iteration": iteration,
        "losses_g": losses_g,
        "losses_d": losses_d
    }
    fabric.save(path, state)

def load_checkpoint(fabric, path, generator, discriminator, optimizer_g, optimizer_d):
    state = {
        "generator": generator,
        "discriminator": discriminator,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d
    }
    fabric.load(path, state)
    return state["iteration"], state["losses_g"], state["losses_d"]
