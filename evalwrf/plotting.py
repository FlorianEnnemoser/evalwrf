import glob
import os
from typing import List, Optional
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import natural_earth
import geopandas as gpd
from matplotlib.colors import ListedColormap
#from .config import CONFIG
from PIL import Image

class createAnimation:
    def __init__(self, input_files,output_file):
        """
        Instantiates IO
        """
        self.input_files = sorted(glob.glob(input_files), key=os.path.getmtime)
        self.output_file = output_file
        
        print("Files used:\n",self.input_files)
            
    def gif(self,loops=0,dur=1):
        """
        create a moving gif out of files located in folder+file given by
        variable "image_files". The final file is saved at the script location 
        with a name given by variable "output_file". file format is added automatically
        
        see PIL documentation:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        
        Parameters
        ----------
        loops : TYPE, optional
            Number of loops. If 0, then loops endlessly. The default is 0.
        
        dur : TYPE, optional
            Duration of singular image displayed in Milliseconds. The default is 0.1
        
        Returns
        -------
        None.
        
        """
        print("############\nCreating GIF\n############")
        from PIL import Image

        img, *imgs = [Image.open(f) for f in self.input_files]
        img.save(fp=f"{self.output_file}.gif", format='GIF', append_images=imgs,
                 save_all=True, duration=dur, loop=loops)
        print("saved gif from images!")

    def video(self,fps=30):
        """
        create a video out of files located in folder+file given by
        variable "image_files". The final file is saved at the script location 
        with a name given by variable "output_file". file format is added automatically
        

        Parameters
        ----------
        fps : TYPE, optional
            Frames Per Second of the video. The default is 30.

        Returns
        -------
        None.

        """
        print("############\nCreating Video\n############")
        import cv2
        
        frame = cv2.imread(self.input_files[0])
        height, width, layers = frame.shape
        video = cv2.VideoWriter(f'{self.output_file}.mp4', 0, fps, (width,height))
        
        for image in self.input_files:
            video.write(cv2.imread(image))
        
        cv2.destroyAllWindows()
        video.release()
        print("saved video from images!")

def create_gif(image_files : str = "img/*.jpg", output_file : str = "animation", loop_gif : int = 0):
    createAnimation(image_files, output_file).gif(loop_gif)

def create_video(image_files : str = "img/*.jpg", output_file : str = "animation", fps : int = 12):
    createAnimation(image_files, output_file).video(fps)


radar_cmap = ListedColormap(np.array([[4,233,231],
                    [1,159,244], [3,0,244],
                    [2,253,2], [1,197,1],
                    [0,142,0], [253,248,2],
                    [229,188,0], [253,149,0],
                    [253,0,0], [212,0,0],
                    [188,0,0],[248,0,253],
                    [152,84,198]], np.float32) / 255.0)


def update_plot_layout(
    fig, 
    image_path=None, 
    opacity=0.5, 
    background_color='white', 
    secondary_background_color='lightgray',
    font_family='Arial',
    font_size=12
):
    """
    Update Plotly Express figure layout with background image and colors
    """
    fig.update_layout(
        paper_bgcolor=background_color,
        plot_bgcolor=secondary_background_color,
        font=dict(
            family=font_family,
            size=font_size
        )        
    )
    
    if image_path:

        fig.update_layout(
            images=[{
                'source': Image.open(image_path),
                'xref': 'paper',
                'yref': 'paper',
                'x': 0,
                'y': 1,
                'sizex': 0.2,
                'sizey': 0.2,
                'opacity': opacity,
                'layer': 'below'
            }]
        )
    
    return fig

def rename_wrfout_files(folder_path : str, testrun : bool = True, keep_files : list[str] = []) -> None:
    if testrun:
        print("-------- THIS IS A TESTRUN ----------")
    
    for filename in os.listdir(folder_path):
        match = ("\uf03a" in filename) and (filename.startswith("wrfout"))
        if match and (filename not in keep_files):
            new_filename = filename.replace("\uf03a","_") + ".nc"
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            if not testrun:
                os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_filename}")

def find_wrfout_nc_files(folder_path : Path | str, dom : str = "d01") -> list:
    folder = Path(folder_path)
    wrfout_files = [file for file in folder.iterdir() if file.name.startswith("wrfout") and file.suffix == ".nc" and dom in file.name]
    return wrfout_files

def load_files(files : list, dim : str = "Time") -> xr.Dataset:
    return xr.concat([xr.load_dataset(f) for f in files],dim=dim)

def plotting_simple(dataarray : xr.DataArray,
                    levels : int | List[int] = 11, 
                    cmap : str = "jet", 
                    title : str = "",
                    cbartitle : str = "",
                    second_dataarray : Optional[xr.DataArray] = None,
                    #custom_config : Optional[dict] = CONFIG,
                    **plt_kwargs):
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": proj})

    natural_earth_path = natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
    gdf = gpd.read_file(natural_earth_path)
    austrian_states = gdf[gdf['adm0_a3'] == 'AUT']
    austrian_states.boundary.plot(ax=ax, edgecolor='k', linewidth=1)

    # bezirk = gpd.read_file(custom_config,columns=["geometry","boundary"])
    # bezirk.boundary.plot(ax=ax,edgecolor='k', lw=1)#,facecolor="none",zorder=99)

    ax.coastlines(resolution='10m', color='black', linestyle='-', lw=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), lw=0.5)    

    gl = ax.gridlines(crs=proj, draw_labels=True, lw=1, color='gray', alpha=0.2, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontfamily': 'serif'}
    gl.ylabel_style = {'fontfamily': 'serif'}
    ax.set_xlabel("Longitude (°E)",fontfamily='serif')
    ax.set_ylabel("Latitude (°N)",fontfamily='serif')

    if isinstance(second_dataarray,xr.DataArray):
        second_dataarray.plot.pcolormesh(ax=ax,
                            x='XLONG', y='XLAT', 
                            cmap=cmap,
                            levels=levels,extend='max',
                            transform=proj,add_colorbar=False,**plt_kwargs)

    im = dataarray.plot.pcolormesh(ax=ax,
                            x='XLONG', y='XLAT', 
                            cmap=cmap,
                            levels=levels,extend='max',
                            transform=proj,add_colorbar=False,**plt_kwargs)




    cb = fig.colorbar(im, ax=ax, orientation="horizontal")
    cb.set_label(label=cbartitle,fontsize=12,fontfamily='serif')
    ax.set_title('') #hack https://github.com/pydata/xarray/issues/2981
    ax.set_title(title, loc="left")
    for ax in fig.get_axes():
        for xlabel in ax.get_xticklabels():
            xlabel.set_fontfamily('serif')
        for ylabel in ax.get_yticklabels():
            ylabel.set_fontfamily('serif')


    return None

def cleandir(directory : str, datadir : str, trialrun : bool = True) -> None:
    """
    Example
    --------
    >>> clean_sim_dir = "002_Test_GFS" #dir under `simulation\\wrf`
    >>> dir_where_data_is_saved = "grib2data"
    >>> cleandir(clean_sim_dir,dir_where_data_is_saved,trialrun=False)
    """
    simdir = Path(directory)

    keep_files = [
        "namelist.wps",
        "namelist.input",
        "run.sh",
        "vars_io.txt",
        "vars_io_d01.txt",
        "vars_io_d02.txt",
        "vars_io_d03.txt",
        datadir
        ]
    
    for f in simdir.iterdir():
        if (f.name not in keep_files):
            if not trialrun:
                os.remove(f.resolve())
            print(f"Removed -> {f.resolve()}")        