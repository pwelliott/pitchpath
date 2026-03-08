import numpy as np
import pymc as pm
import pytensor.tensor as pt

def solve_time_pt(y0, vy0, ay):
    """
    Pytensor version of time-to-plate solver for use within PyMC models.
    """
    discriminant = vy0**2 - 2 * ay * y0
    return (-vy0 - pt.sqrt(discriminant)) / ay

def solve_time_np(y0, vy0, ay):
    """
    Numpy version of time-to-plate solver for final calculations.
    """
    discriminant = vy0**2 - 2 * ay * y0
    return (-vy0 - np.sqrt(discriminant)) / ay

def fit_trajectory(parameters, actual_plate, sigma_obs=0.01, sigma_prior=2.0):
    """
    Fits a 9-parameter kinematic model to the observed plate crossing.
    Returns the MAP estimate of refined accelerations.
    """
    x0, y0, z0 = parameters['x0'], parameters['y0'], parameters['z0']
    vx0, vy0, vz0 = parameters['vx0'], parameters['vy0'], parameters['vz0']
    ax_stat, ay_stat, az_stat = parameters['ax'], parameters['ay'], parameters['az']
    
    x_obs = actual_plate['x']
    z_obs = actual_plate['z']

    with pm.Model() as model:
        # Priors for acceleration
        ax = pm.Normal('ax', mu=ax_stat, sigma=sigma_prior)
        ay = pm.Normal('ay', mu=ay_stat, sigma=sigma_prior)
        az = pm.Normal('az', mu=az_stat, sigma=sigma_prior)
        
        # Physics model
        T = solve_time_pt(y0, vy0, ay)
        
        x_pred = x0 + vx0 * T + 0.5 * ax * T**2
        z_pred = z0 + vz0 * T + 0.5 * az * T**2
        
        # Likelihood
        pm.Normal('obs_x', mu=x_pred, sigma=sigma_obs, observed=x_obs)
        pm.Normal('obs_z', mu=z_pred, sigma=sigma_obs, observed=z_obs)
        
        # MAP Estimate
        map_estimate = pm.find_MAP(progressbar=False)
        
    ay_map = float(map_estimate['ay'])
    T_map = float(solve_time_np(y0, vy0, ay_map))

    return {
        'ax': float(map_estimate['ax']),
        'ay': ay_map,
        'az': float(map_estimate['az']),
        'time': T_map
    }

def predict_plate_crossing(parameters, time=None):
    """
    Utility to calculate x, z at home plate given parameters and time.
    """
    T = time if time is not None else solve_time_np(parameters['y0'], parameters['vy0'], parameters['ay'])
    x = parameters['x0'] + parameters['vx0'] * T + 0.5 * parameters['ax'] * T**2
    z = parameters['z0'] + parameters['vz0'] * T + 0.5 * parameters['az'] * T**2
    return x, z
