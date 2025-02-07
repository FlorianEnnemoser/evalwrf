import numpy as np
from dataclasses import dataclass
# https://www2.mmm.ucar.edu/wrf/users/wrfda/OnlineTutorial/Help/littler.html
# https://pypi.org/project/fortranformat/

@dataclass
class Consts:
    R : float = 8.314462618
    """Molar Gas Constant"""

    Md : float = 28.96546e-3
    """Dry Air molecular weight"""

    g : float = 9.80665
    """Gravitational Force on Earth"""

    Rd : float = R / Md  
    """Dry Air Gas Constant"""



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

def u_from_vector(magnitude : float, degree : float) -> float:
    return np.cos(np.deg2rad(degree)) * magnitude

def v_from_vector(magnitude : float, degree : float) -> float:
    return np.sin(np.deg2rad(degree)) * magnitude

def slp_from_station_pressure(pressure : float, elevation : float, temperature : float) -> float:
    H = Consts.Rd * temperature / Consts.g
    return pressure * np.exp(elevation/H)