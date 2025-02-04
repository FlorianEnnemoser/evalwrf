from dataclasses import dataclass
from datetime import datetime
from typing import Self

@dataclass
class NamelistConfig:
    
    start_date: str 
    end_date: str
    max_dom: int = 1

    i_parent_start: int = 1
    j_parent_start: int = 1
    e_we: int = 91
    e_sn: int = 100
    dx: int = 27000
    dy: int = 27000
    ref_lat           = 48.0
    ref_lon           = 10.0

    parent_id: int = 1
    parent_grid_ratio: int = 1

    map_proj = 'lambert'
    truelat1 = 30.0
    truelat2 = 60.0
    stand_lon = 10.0

    out_format: str = 'WPS'
    prefix: str = 'FILE'
    fg_name: str = 'FILE'

    def __post_init__(self):
        zero_hour = '_00:00:00'

        self.wrf_core = 'ARW'
        self.interval_seconds = 21600
        self.geog_data_res = 'default'

        self.start_date += zero_hour
        self.end_date += zero_hour

        self.start_dt = datetime.strptime(self.start_date, '%Y-%m-%d_%H:%M:%S')
        self.end_dt = datetime.strptime(self.end_date, '%Y-%m-%d_%H:%M:%S')
        
        self._validate_input()

    def _validate_input(self) -> Self:
        if self.max_dom > 3:
            raise NotImplementedError("Only up to max_dom = 3 supported.")
        
        if self.max_dom > 1:
            assert len(self.i_parent_start) == self.max_dom
            assert len(self.j_parent_start) == self.max_dom
            assert len(self.e_we) == self.max_dom
            assert len(self.e_sn) == self.max_dom

            self.start_date = self._join_maxdom_times_str(self.start_date)
            self.end_date = self._join_maxdom_times_str(self.end_date)
            self.geog_data_res = self._join_maxdom_times_str(self.geog_data_res)

            self.start_year =  self._join_maxdom_times_number(self.start_dt.year)
            self.start_month = self._join_maxdom_times_number(self.start_dt.month)
            self.start_day =   self._join_maxdom_times_number(self.start_dt.day)
            self.start_hour =  self._join_maxdom_times_number(str(self.end_dt.hour).zfill(2))
            self.end_year =    self._join_maxdom_times_number(self.end_dt.year)
            self.end_month =   self._join_maxdom_times_number(self.end_dt.month)
            self.end_day =     self._join_maxdom_times_number(self.end_dt.day)            
            self.end_hour =    self._join_maxdom_times_number(str(self.end_dt.hour).zfill(2))

            if len(self.parent_id) == 1:
                self.parent_id = '1, 1, 2'

            if len(self.parent_grid_ratio) == 1:
                self.parent_grid_ratio = '1, 3, 3' 

        return self

    def _join_maxdom_times_str(self, value) -> str:
        return ','.join([f"'{value}'" for _ in range(self.max_dom)])

    def _join_maxdom_times_number(self, value) -> str:
        return ','.join([value for _ in range(self.max_dom)])

    def _generate_namelist_wps(self) -> str:
        namelist = f"""&share
    wrf_core          = '{self.wrf_core}',
    max_dom           = {self.max_dom},
    start_date        = '{self.start_date}',
    end_date          = '{self.end_date}',
    interval_seconds  = {self.interval_seconds}
/

&geogrid
    parent_id         = {self.parent_id},
    parent_grid_ratio = {self.parent_grid_ratio},
    i_parent_start    = {self.i_parent_start},
    j_parent_start    = {self.j_parent_start},
    e_we              = {self.e_we},
    e_sn              = {self.e_sn},
    geog_data_res     = '{self.geog_data_res}',
    dx                = {self.dx},
    dy                = {self.dy},
    map_proj          = '{self.map_proj}',
    ref_lat           = {self.ref_lat},
    ref_lon           = {self.ref_lon},
    truelat1          = {self.truelat1},
    truelat2          = {self.truelat2},
    stand_lon         = {self.stand_lon},
    geog_data_path    = '/home/wrfuser/terrestrial_data/WPS_GEOG'
    /

&ungrib
    out_format        = '{self.out_format}',
    prefix            = '{self.prefix}',
/

&metgrid
    fg_name           = '{self.fg_name}'
/
"""
        return namelist

    def _generate_namelist_input(self) -> str:
        namelist_input = f"""&time_control
    run_days                            = 0,
    run_hours                           = 0,
    run_minutes                         = 0,
    run_seconds                         = 0,
    start_year                          = {self.start_year},
    start_month                         = {self.start_month},
    start_day                           = {self.start_day},
    start_hour                          = {self.start_hour},
    end_year                            = {self.end_year},
    end_month                           = {self.end_month},
    end_day                             = {self.end_day},
    end_hour                            = {self.end_hour},
    interval_seconds                    = 21600
    input_from_file                     = .true., .true., .true.,
    history_interval                    = 360, 360, 30
    frames_per_outfile                  = 1,
    restart                             = .false.,
    restart_interval                    = 1440,
    io_form_history                     = 2
    io_form_restart                     = 2
    io_form_input                       = 2
    io_form_boundary                    = 2
    iofields_filename                   = "vars_io.txt","vars_io.txt","vars_io.txt",
    ignore_iofields_warning             = .true.    
/

&domains
    time_step                           = 150,
    time_step_fract_num                 = 0,
    time_step_fract_den                 = 1,     
    max_dom                             = {self.max_dom},
    e_we                                = {self.e_we},
    e_sn                                = {self.e_sn},
    e_vert                              = 45, 45, 45
    dx                                  = {self.dx},
    dy                                  = {self.dy},
    grid_id                             = {self.parent_id},
    parent_id                           = {self.parent_id},
    i_parent_start                      = {self.i_parent_start},
    j_parent_start                      = {self.j_parent_start},
    parent_grid_ratio                   = {self.parent_grid_ratio},
    parent_time_step_ratio              = 1,
    feedback                            = 1,
    smooth_option                       = 0,
    num_metgrid_levels                  = 34
    num_metgrid_soil_levels             = 4,    
    dzstretch_s                         = 1.1
    p_top_requested                     = 5000,
/

&physics
    physics_suite                       = 'CONUS'
    mp_physics                          = -1,    -1,
    cu_physics                          = -1,    -1,
    ra_lw_physics                       = -1,    -1,
    ra_sw_physics                       = -1,    -1,
    bl_pbl_physics                      = -1,    -1,
    sf_sfclay_physics                   = -1,    -1,
    sf_surface_physics                  = -1,    -1,
    radt                                = 15,    15,
    bldt                                = 0,     0,
    cudt                                = 0,     0,
    icloud                              = 1,
    num_land_cat                        = 21,
    sf_urban_physics                    = 0,     0,
    fractional_seaice                   = 1,
/

&fdda
/

&dynamics
    hybrid_opt                          = 2, 
    w_damping                           = 0,
    diff_opt                            = 2,      2,
    km_opt                              = 4,      4,
    diff_6th_opt                        = 0,      0,
    diff_6th_factor                     = 0.12,   0.12,
    base_temp                           = 290.
    damp_opt                            = 3,
    zdamp                               = 5000.,  5000.,
    dampcoef                            = 0.2,    0.2,
    khdif                               = 0,      0,
    kvdif                               = 0,      0,
    non_hydrostatic                     = .true., .true.,
    moist_adv_opt                       = 1,      1,
    scalar_adv_opt                      = 1,      1,
    gwd_opt                             = 1,      0,
/

&bdy_control
    spec_bdy_width                      = 5,
    specified                           = .true.
/

&grib2
/

&namelist_quilt
    nio_tasks_per_group                 = 0,
    nio_groups                          = 1,
/

&diags
    z_lev_diags                         =  0,
    num_z_levels                        =  0,
    z_levels                            =  0
    p_lev_diags                         =  0,
    num_press_levels                    =  0,
    solar_diagnostics                   =  1,
/
    """
        return namelist_input        

    def write_namelist_wps(self, filename: str = "namelist.wps"):
        namelist = self._generate_namelist_wps()
        self._write(filename,namelist)

    def write_namelist_wps(self, filename: str = "namelist.input"):
        namelist = self._generate_namelist_input()
        self._write(filename,namelist)

    def _write(self,filename : str, data : str) -> None:
        with open(filename, 'w') as file:
            file.write(data)