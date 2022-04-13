import numpy as np


from ima_vae.data.data_generators.spriteworld import (
    factor_distributions as distribs,
    sprite_generators,
    renderers as spriteworld_renderers,
    tasks,
)


def random_sprites_config(beta_params, label, args, angle_params, shape_probs):
    factor_list = [
        distribs.Beta("x", beta_params[label][0][0], beta_params[label][0][1]),
        distribs.Beta("y", beta_params[label][1][0], beta_params[label][1][1]),
        distribs.Beta("scale", beta_params[label][2][0], beta_params[label][2][1]),
        # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
        distribs.Beta("c0", beta_params[label][3][0], beta_params[label][3][1]),
        distribs.Continuous("c1", 1.0, 1.0),
        distribs.Continuous("c2", 1.0, 1.0),
    ]

    if args.angle:
        angles = np.random.uniform(args.lower, args.upper, 2)
        angle_params[label] = angles
        factor_list.append(
            distribs.Beta("angle", angle_params[label][0], angle_params[label][1])
        )

    if args.shape:
        probs = np.random.uniform(0, 1, 3)
        probs = probs / probs.sum()
        shape_probs[label] = probs
        factor_list.append(
            distribs.Discrete(
                "shape", ["triangle", "square", "pentagon"], probs=shape_probs[label]
            )
        )
    else:
        factor_list.append(distribs.Discrete("shape", ["triangle"]))

    factors = distribs.Product(factor_list)
    sprite_gen = sprite_generators.generate_sprites(factors, num_sprites=1)

    renderers = {
        "image": spriteworld_renderers.PILRenderer(
            image_size=(64, 64),
            anti_aliasing=5,
            color_to_rgb=spriteworld_renderers.color_maps.hsv_to_rgb,
        ),
        "attributes": spriteworld_renderers.SpriteFactors(
            factors=("x", "y", "shape", "angle", "scale", "c0", "c1", "c2")
        ),
    }

    config = {
        "task": tasks.NoReward(),
        "action_space": None,
        "renderers": renderers,
        "init_sprites": sprite_gen,
        "max_episode_length": 1,
    }
    return config
