"""Script for training and plotting the NN model."""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################
    #Initialisering av treningsdata
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    
    #Trening av nettverket
    nn_params, loss_dict = train_nn(sensor_data, cfg)

    #Prediksjon av data
    T_pred = predict_grid(nn_params, x, y, t, cfg)

    #Plotting

    plot_snapshots(
        x,
        y,
        t, 
        T_pred, 
        save_path = "output/nn/nn_snapshots.png" )
    
    create_animation(
        x, y, t, T_pred, title="NN", save_path="output/nn/nn_animation.gif"
    )

    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
