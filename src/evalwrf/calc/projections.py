import numpy as np
import xarray as xr
import cartopy.io.shapereader as shpreader

from .constants import WRF_EARTH_RADIUS, RAD_PER_DEG, DEG_PER_RAD, PI


def convert(tude: str) -> float:
    multiplier = 1 if tude[-1] in ["N", "E"] else -1
    return multiplier * sum(
        float(x) / 60**n for n, x in enumerate(tude[:-1].split("-"))
    )


def lon_switch(deltalon: float) -> float:
    if deltalon < -180.0:
        deltalon += 360.0
    if deltalon > 180.0:
        deltalon -= 360.0
    return deltalon


def latlon_to_ij(ds: xr.Dataset, lat: float, lon: float) -> dict[str, int]:
    """
    TODO: check https://github.com/NCAR/wrf-python/blob/develop/src/wrf/latlonutils.py
    as there is, depending on map projection, different methods to calculate ref_lat_val and ref_lon_val and
    latinc and loninc. Current implementation is only for MAP_PROJ == 1 ; Lambert Conformal.
    """
    if not isinstance(ds, xr.Dataset):
        raise ValueError(
            "Only for WRF xr.Dataset, as this contain information on projection etc."
        )

    ref_lat_val = np.ravel(ds.XLAT[..., 0, 0])[0]
    ref_lon_val = np.ravel(ds.XLONG[..., 0, 0])[0]

    _ll2ij_kwargs = dict(
        map_proj=ds.MAP_PROJ,
        truelat1=ds.TRUELAT1,
        truelat2=ds.TRUELAT2,
        stdlon=ds.STAND_LON,
        lat1=ref_lat_val,
        lon1=ref_lon_val,
        pole_lat=ds.POLE_LAT,
        pole_lon=ds.POLE_LON,
        knowni=1,
        knownj=1,
        dx=ds.DX,
        latinc=0,
        loninc=0,
    )

    return _latlon_to_ij(lat, lon, **_ll2ij_kwargs)


def _latlon_to_ij(
    lat: float,
    lon: float,
    map_proj: int,
    truelat1: float,
    truelat2: float,
    stdlon: float,
    lat1: float,
    lon1: float,
    pole_lat: float,
    pole_lon: float,
    knowni: float,
    knownj: float,
    dx: float,
    latinc: float,
    loninc: float,
) -> tuple[float, float]:
    """
    Sources are:

    https://github.com/NCAR/wrf-python/blob/develop/src/wrf/latlonutils.py
    and
    https://github.com/NCAR/wrf-python/blob/develop/fortran/wrf_user_latlon_routines.f90

    Convert a lat/lon coordinate to WRF grid (i, j) indices.

    Mirrors the Fortran DLLTOIJ subroutine, supporting all four WRF
    map projections:
        1 - Lambert Conformal
        2 - Polar Stereographic
        3 - Mercator
        6 - Lat/Lon (regular or rotated pole)

    Parameters
    ----------
        map_proj        : WRF map projection code (1, 2, 3, or 6)
        truelat1        : First true latitude (all projections)
        truelat2        : Second true latitude (Lambert only; ignored otherwise)
        stdlon          : Standard longitude parallel to the y-axis
        lat1, lon1      : Lat/lon of the SW corner (grid point 1,1)
        pole_lat        : Pole latitude  (lat/lon projection only)
        pole_lon        : Pole longitude (lat/lon projection only)
        knowni, knownj  : i/j of the known reference point (usually 1.0, 1.0)
        dx, dy          : Grid spacing in metres
        latinc, loninc  : Lat/lon increment (lat/lon projection only)
        lat, lon        : Target latitude/longitude to convert

    Returns
    -------
    (j, i) : tuple of floats - WRF row and column indices (1-based)

    Raises
    ------
    ValueError  : if map_proj is not one of {1, 2, 3, 6}
    ValueError  : if selected latitude and longitude is outside sw-corner

    Example
    --------
    >>> ref_lat_val = np.ravel(ds_inp.XLAT[...,0,0])[0]
    >>> ref_lon_val = np.ravel(ds_inp.XLONG[...,0,0])[0]

    >>> coords = ll_to_ij(
            map_proj=ds_inp.MAP_PROJ,
            truelat1=ds_inp.TRUELAT1,
            truelat2=ds_inp.TRUELAT2,
            stdlon=ds_inp.STAND_LON,

            lat1=ref_lat_val,
            lon1=ref_lon_val,

            pole_lat=ds_inp.POLE_LAT,
            pole_lon=ds_inp.POLE_LON,
            knowni=1,
            knownj=1,
            dx=ds_inp.DX,
            latinc=0, # für alle Projections bis auf wenn projection gleich LAT / LON ist
            loninc=0, # für alle Projections bis auf wenn projection gleich LAT / LON ist
            lat=47.5,
            lon=16.0
        )


    """

    if lat < lat1:
        raise ValueError("Given latitude is outside simulation domain!")

    if lon < lon1:
        raise ValueError("Given longitude is outside simulation domain!")

    rebydx = WRF_EARTH_RADIUS / dx
    hemi = 1.0 if truelat1 >= 0.0 else -1.0

    # ------------------------------------------------------------------ #
    # Mercator  (map_proj == 3)
    # ------------------------------------------------------------------ #
    if map_proj == 3:
        clain = np.cos(RAD_PER_DEG * truelat1)
        dlon = dx / (WRF_EARTH_RADIUS * clain)

        rsw = 0.0
        if lat1 != 0.0:
            rsw = np.log(np.tan(0.5 * (lat1 + 90.0) * RAD_PER_DEG)) / dlon

        deltalon = lon_switch(lon - lon1)

        i = knowni + (deltalon / (dlon * DEG_PER_RAD))
        j = knownj + np.log(np.tan(0.5 * (lat + 90.0) * RAD_PER_DEG)) / dlon - rsw

    # ------------------------------------------------------------------ #
    # Polar Stereographic  (map_proj == 2)
    # ------------------------------------------------------------------ #
    elif map_proj == 2:
        reflon = stdlon + 90.0
        scale_top = 1.0 + hemi * np.sin(truelat1 * RAD_PER_DEG)

        ala1 = lat1 * RAD_PER_DEG
        rsw = rebydx * np.cos(ala1) * scale_top / (1.0 + hemi * np.sin(ala1))

        alo1 = (lon1 - reflon) * RAD_PER_DEG
        polei = knowni - rsw * np.cos(alo1)
        polej = knownj - hemi * rsw * np.sin(alo1)

        ala = lat * RAD_PER_DEG
        rm = rebydx * np.cos(ala) * scale_top / (1.0 + hemi * np.sin(ala))
        alo = (lon - reflon) * RAD_PER_DEG

        i = polei + rm * np.cos(alo)
        j = polej + hemi * rm * np.sin(alo)

    # ------------------------------------------------------------------ #
    # Lambert Conformal  (map_proj == 1)
    # ------------------------------------------------------------------ #
    elif map_proj == 1:
        if abs(truelat2) > 90.0:
            truelat2 = truelat1

        if abs(truelat1 - truelat2) > 0.1:
            cone = (
                np.log(np.cos(truelat1 * RAD_PER_DEG))
                - np.log(np.cos(truelat2 * RAD_PER_DEG))
            ) / (
                np.log(np.tan((90.0 - abs(truelat1)) * RAD_PER_DEG * 0.5))
                - np.log(np.tan((90.0 - abs(truelat2)) * RAD_PER_DEG * 0.5))
            )
        else:
            cone = np.sin(abs(truelat1) * RAD_PER_DEG)

        deltalon1 = lon_switch(lon1 - stdlon)

        tl1r = truelat1 * RAD_PER_DEG
        ctl1r = np.cos(tl1r)

        rsw = (
            rebydx
            * ctl1r
            / cone
            * (
                np.tan((90.0 * hemi - lat1) * RAD_PER_DEG / 2.0)
                / np.tan((90.0 * hemi - truelat1) * RAD_PER_DEG / 2.0)
            )
            ** cone
        )

        arg = cone * (deltalon1 * RAD_PER_DEG)
        polei = hemi * knowni - hemi * rsw * np.sin(arg)
        polej = hemi * knownj + rsw * np.cos(arg)

        deltalon = lon_switch(lon - stdlon)

        rm = (
            rebydx
            * ctl1r
            / cone
            * (
                np.tan((90.0 * hemi - lat) * RAD_PER_DEG / 2.0)
                / np.tan((90.0 * hemi - truelat1) * RAD_PER_DEG / 2.0)
            )
            ** cone
        )

        arg = cone * (deltalon * RAD_PER_DEG)
        i = polei + hemi * rm * np.sin(arg)
        j = polej - rm * np.cos(arg)

        # Flip for southern hemisphere (SW-corner origin convention)
        i = hemi * i
        j = hemi * j

    # ------------------------------------------------------------------ #
    # Lat/Lon - regular or rotated pole  (map_proj == 6)
    # ------------------------------------------------------------------ #
    elif map_proj == 6:
        if pole_lat != 90.0:
            olat, olon = _rotate_coords(
                lat, lon, pole_lat, pole_lon, stdlon, direction=-1
            )
            lat = olat
            lon = olon + stdlon

            olat, olon = _rotate_coords(
                lat1, lon1, pole_lat, pole_lon, stdlon, direction=-1
            )
            lat1n = olat
            lon1n = olon + stdlon

            deltalat = lat - lat1n
            deltalon = lon - lon1n
        else:
            deltalat = lat - lat1
            deltalon = lon - lon1

        i = deltalon / loninc + knowni
        j = deltalat / latinc + knownj

    else:
        raise ValueError(f"Unknown map projection: {map_proj}. Must be 1, 2, 3, or 6.")

    return {
        "west_east": np.rint(i - 1).astype(int),
        "south_north": np.rint(j - 1).astype(int),
    }  # lon,lat idx; -1 as this function is fortran index based


def _rotate_coords(
    ilat: float,
    ilon: float,
    lat_np: float,
    lon_np: float,
    lon_0: float,
    direction: int,
) -> tuple[float, float]:
    """
    Rotate coordinates between computational and geographical systems.

    Parameters
    ----------
    ilat, ilon  : Input latitude/longitude (degrees)
    lat_np      : Latitude  of the North Pole in the other system (degrees)
    lon_np      : Longitude of the North Pole in the other system (degrees)
    lon_0       : Reference longitude (degrees)
    direction   : >= 0 → computational to geographical
                  <  0 → geographical  to computational

    Returns
    -------
    (olat, olon) : Rotated latitude/longitude (degrees)
    """
    phi_np = lat_np * RAD_PER_DEG
    lam_np = lon_np * RAD_PER_DEG
    lam_0 = lon_0 * RAD_PER_DEG
    rlat = ilat * RAD_PER_DEG
    rlon = ilon * RAD_PER_DEG

    dlam = (PI - lam_0) if direction < 0 else lam_np

    sinphi = np.cos(phi_np) * np.cos(rlat) * np.cos(rlon - dlam) + np.sin(
        phi_np
    ) * np.sin(rlat)
    cosphi = np.sqrt(1.0 - sinphi**2)

    coslam = np.sin(phi_np) * np.cos(rlat) * np.cos(rlon - dlam) - np.cos(
        phi_np
    ) * np.sin(rlat)
    sinlam = np.cos(rlat) * np.sin(rlon - dlam)

    if cosphi != 0.0:
        coslam /= cosphi
        sinlam /= cosphi

    olat = DEG_PER_RAD * np.arcsin(sinphi)
    olon = DEG_PER_RAD * (np.arctan2(sinlam, coslam) - dlam - lam_0 + lam_np)

    return olat, olon


def find_closest_city(
    lat: float,
    lon: float,
    pop_size: int = 50_000,
) -> dict:
    """Find the nearest populated place above a population threshold.

    Parameters
    ----------
    lat : float
        Query latitude in degrees North.
    lon : float
        Query longitude in degrees East.
    pop_size : int, optional
        Minimum population for consideration.  Defaults to ``50_000``.

    Returns
    -------
    dict
        ``{"name": str, "lat": float, "lon": float}`` for the closest city.
    """

    def _distance(city):
        return np.hypot(lat - city.geometry.y, lon - city.geometry.x)

    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="populated_places"
    )
    cities = shpreader.Reader(shpfilename).records()

    large_cities = [c for c in cities if c.attributes.get("POP_MAX", 0) > pop_size]
    closest = min(large_cities, key=_distance)
    return {
        "name": closest.attributes["NAME"],
        "lat": closest.geometry.y,
        "lon": closest.geometry.x,
    }
