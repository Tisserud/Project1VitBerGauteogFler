"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    
    #Trening av nettverket
    pinn_params, loss_dict = train_pinn(sensor_data, cfg)

    #Prediksjon av data
    T_pred = predict_grid(pinn_params['nn'], x, y, t, cfg)

    #Plotting

    plot_snapshots(
        x,
        y,
        t, 
        T_pred, 
        save_path = "output/pinn/pinn_snapshots.png" )
    
    create_animation(
        x, y, t, T_pred, title="NN", save_path="output/pinn/pinn_animation.gif"
    )

    #Losses

    total = loss_dict['total']
    data = loss_dict['data']
    physics = loss_dict['physics']
    bc = loss_dict['bc']
    ic = loss_dict['ic']


    x = [x for x in range(len(total))]

    plt.figure()
    plt.plot(x, jnp.log(total), label = "total")
    plt.plot(x, jnp.log(data), label = "data")
    plt.plot(x, jnp.log(physics), label = "physics")
    plt.plot(x, jnp.log(bc), label = "bc")
    plt.plot(x, jnp.log(ic), label = "ic")
    plt.title("Log loss")
    plt.grid()
    plt.legend()
    plt.savefig('output/pinn/losses.png')


    print(f"""
alpha: {jnp.exp(pinn_params['log_alpha'])}
k: {jnp.exp(pinn_params['log_k'])}
h: {jnp.exp(pinn_params['log_h'])}
power: {jnp.exp(pinn_params['log_power'])}
""")

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
