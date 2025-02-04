import pandas as pd
import numpy as np
import fortranformat as ff

# https://www2.mmm.ucar.edu/wrf/users/wrfda/OnlineTutorial/Help/littler.html
# https://pypi.org/project/fortranformat/

def saturation_water_vapor_pressure(T : float) -> float:
    """
    https://en.wikipedia.org/wiki/Vapour_pressure_of_water > August-Roche-Magnus 
    """
    return 0.61094*np.exp(17.625*T / (T+243.04)) * 1e3 # (Pa)

def saturation_vapor_density(T : float, molecular_weight : float = 18.) -> float:
    """Ideal Gas law with 18 as molecular weight for water (in g/mol)"""
    R = 8.31 # J/molK
    VP = saturation_water_vapor_pressure(T)
    return VP / (R*(T+273.15)) * molecular_weight

def u_from_vector(magnitude : float,degree : float):
    return np.cos(np.deg2rad(degree)) * magnitude

def v_from_vector(magnitude : float,degree : float):
    return np.sin(np.deg2rad(degree)) * magnitude
