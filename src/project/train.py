"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior


def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg)
    adam_state = init_adam(nn_params)

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start
    #######################################################################
    @jit
    def objective_fn(
        nn_params : list[tuple[jnp.ndarray, jnp.ndarray]],
        )-> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """"Objective function of NN
        Args:
            nn_params: Network parameters
        Returns:
            objetive function: Objective funcion to be optimized
            partial losses: Tuple of individual losses
            """
            
        L_data = cfg.lambda_data*data_loss(nn_params, sensor_data, cfg)
        L_ic = cfg.lambda_ic*ic_loss(nn_params, ic_epoch, cfg)
        return L_data + L_ic, (L_data, L_ic)
    objektive_fn_jit=jit(objective_fn)

    for _ in tqdm(range(cfg.num_epochs), desc="Training NN"):
        ic_epoch, key = sample_ic(key, cfg)
        
        values, grads = jax.value_and_grad(objektive_fn_jit, has_aux=True)(nn_params)

        nn_params, adam_state = adam_step(nn_params, grads, adam_state, lr=cfg.learning_rate)

        # Update the nn_params and losses dictionary
        total, (data, ic) = values
        losses["total"].append(total)
        losses["data"].append(data)
        losses["ic"].append(ic)

    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################

    return nn_params, {k: jnp.array(v) for k, v in losses.items()}


def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################
    @jit
    def objective_fn(
        pinn_params : dict
        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """"Objective function of PINN            
        Args:
            pinn_param: Dictionary of system parameters
        Returns:                
            objetive function: Objective funcion to be optimized
            partial losses: Tuple of individual losses
            """
        nn_params = pinn_params['nn']

        L_data = cfg.lambda_data*data_loss(nn_params, sensor_data, cfg)
        L_ic = cfg.lambda_ic*ic_loss(nn_params, ic_epoch, cfg)
        L_ph = cfg.lambda_physics*physics_loss(pinn_params, interior_epoch, cfg)
        L_bc = cfg.lambda_bc * bc_loss(pinn_params, bc_epoch, cfg)

        L = L_data + L_ic + L_ph + L_bc

        return L, (L_data, L_ic, L_ph, L_bc)
    
    objektive_fn_jit=jit(objective_fn)

    for _ in tqdm(range(cfg.num_epochs), desc = "Training PINN"):
        interior_epoch, key = sample_interior(key, cfg)
        ic_epoch, key = sample_ic(key, cfg)
        bc_epoch, key = sample_bc(key, cfg)

        
        
        values, grads = jax.value_and_grad(objektive_fn_jit, has_aux=True)(pinn_params)

        pinn_params, opt_state = adam_step(pinn_params, grads, opt_state, lr=cfg.learning_rate)

        # Update the pinn_params and losses dictionary
        total, (data, ic, ph, bc) = values
        losses["total"].append(total)
        losses["data"].append(data)
        losses["ic"].append(ic)
        losses["physics"].append(ph)
        losses["bc"].append(bc)



    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
