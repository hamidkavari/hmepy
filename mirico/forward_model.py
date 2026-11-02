import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Literal
from datetime import datetime

# from pyelq.data_access import data_read2
from pyelq.data_access.xarray import load_from_xarray
# from pyelq.data_access.database import load_from_db
from pyelq.data_access.mirico_data import MiricoData
from pyelq.config import builder
from pyelq.source_map import SourceMap
from pyelq.coordinate_system import ENU, LLA
from pyelq.gas_species import CH4
from pyelq.meteorology import meteorology
from pyelq.sensor.sensor import SensorGroup
from pyelq.sensor.beam import Beam
from pyelq.dispersion_model.gaussian_plume import GaussianPlume
from pyelq.dispersion_model.turbulence_model import DraxlerModel

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class ForwardModel():

    def __init__(
            self,
            plume_floor: float = 90.0,
            machine_lla = None,
            data_object: MiricoData = None
        ):
        
        self.gas_species = CH4()
        self.data_object = None

        if machine_lla != None:
            self.machine_lla = machine_lla

        else:     
            self.machine_lla = [50,0,plume_floor]

        if data_object:
            self.data_object = data_object
            self.start = str(data_object.report["timestamp"].min()).split(".", maxsplit=1)[0]
            self.end = str(data_object.report["timestamp"].max()).split(".", maxsplit=1)[0]

            for value in self.data_object.settings.values():
                self.machine_lla = value["orion_lla"]
                break
        
        self.config = {
            'plume_floor' : plume_floor,
            'machine_as_reference' : 'yes',
            'sensor' : {
                'quality_threshold' : 10,
                'n_beam_knots' : 500
            },
            'meteorology':{
                'data_source': 'sensor'
            },
            'horizontal_turbulence' : {
                'model' : 'draxler'
            },
            'vertical_turbulence' : {
                'model' : 'draxler'
            },
            "preprocessor" : {
                "time_bin_edges" : {
                    "freq" : "240s"
                },
                "aggregate_function" : "median",
                "filter_on_met" : {
                    "filter_variable": ['wind_speed'],
                    "lower_limit": [0]
                }
            }
        }

    def load_from_zarr(self,start,end,machine,load_machine_location=True):
        dataset = data_read2.load_zarr(machine,start,end)
        self.data_object = load_from_xarray(dataset, post_calculate_diffusion_params=True)
        
        if load_machine_location == True:
            for value in self.data_object.settings.values():
                self.machine_lla = value["orion_lla"]
                break

    def load_from_database(
        self,
        num_crt: int | None = None,
        mass_emission_rate: Literal["MER1", "MER2", "MER3", "MER4", "MER5"] | None = None,
        wind_speed: Literal["low", "med", "high"] | None = None,
        wind_direction: Literal["North East", "South East", "South West", "North West"] | None = None,
        random_state: int = 0,
        load_machine_location: bool = True,
        dbname: str = "snoopers_db",
        user: str = "postgres",
        password: str = "snoopers",
        host: str = "localhost",
        port: int = 5432):

        self.data_object, release_start, release_end, actual_emissions, hours = load_from_db(
            num_crt,
            mass_emission_rate,
            wind_speed,
            wind_direction,
            random_state,
            dbname,
            user,
            password,
            host,
            port)
        
        if load_machine_location == True:
            for value in self.data_object.settings.values():
                self.machine_lla = value["orion_lla"]
                break

    def initialise_sources(self,source_locations,source_emissions):
        
        if not(isinstance(source_locations,np.ndarray)):
            source_locations = np.array(source_locations)

        if source_locations.ndim == 1:
            source_locations = np.array([source_locations])

        num_sources = source_locations.shape[0]

        if not(isinstance(source_emissions,np.ndarray)):
            source_emissions = np.array(source_emissions)
        if source_emissions.ndim == 0:
            source_emissions = np.array([source_emissions])

        if len(source_emissions) != num_sources:
            raise ValueError("The number of source locations does not match the number of source emissions")

        coordinate_object = ENU(
            ref_latitude=self.machine_lla[0],
            ref_longitude=self.machine_lla[1],
            ref_altitude= self.config["plume_floor"])
            
        source_map = SourceMap()
        coordinate_object.from_array(array=source_locations)
        source_map.location = coordinate_object

        self.source_map = source_map 

        self.source_emissions = source_emissions

    def set_meteorology(
        self,
        time,
        wind_speed,
        wind_direction,
        pressure = None,
        temperature = None,
        wind_turbulence_horizontal = None,
        wind_turbulence_vertical = None):

        """
        Args:
            time: (list) Labels given to each sample given. This can represent the timestamp, or can be arbitrary.
            wind_speed: (list) The wind speed (m/s)
            wind_direction: (list) The direction the wind is coming from, in degrees clockwise from north
            pressure: (list,optional) The air pressure, used when calculating the gas density (Kg/s). Defaults to 101.325
            temperature: (list,optional) The temperature, used when calculating the gas density (Kelvin). Defaults to 273.15
            wind_turbulence_horizontal: (list,optional) The horizontal wind turbulence. Defaults to 0.5
            wind_turbulence_vertical: (list,optional) The vertical wind turbulence. Defaults to 0.5
        """

        met_dict = {
            "time":time,
            "wind_speed":wind_speed,
            "wind_direction":wind_direction            
        }

        if pressure == None:
            pressure = np.ones_like(time)*101.325
        met_dict["pressure"] = pressure

        if temperature == None:
            temperature = np.ones_like(time)*273.15
        met_dict["temperature"] = temperature

        if wind_turbulence_horizontal == None:
            wind_turbulence_horizontal = np.ones_like(time)*0.5
        met_dict["wind_turbulence_horizontal"] = wind_turbulence_horizontal

        if wind_turbulence_vertical == None:
            wind_turbulence_vertical = np.ones_like(time)*0.5
        met_dict["wind_turbulence_vertical"] = wind_turbulence_vertical

        for key, value in met_dict.items():
            if not(isinstance(value,np.ndarray)):
                met_dict[key] = np.array(value)
            if met_dict[key].ndim == 0:
                met_dict[key] = np.array([met_dict[key]])
            if len(met_dict[key]) != len(met_dict["time"]):
                raise ValueError("Not all of the meteorology inputs were the same length.")
        
        self.met_object = Meteorology()
        self.met_object.__dict__ = met_dict
        self.met_object.calculate_uv_from_wind_speed_direction()

    def set_meteorology_from_data(self, preprocess=False, bin_size=240):

        if preprocess == False:

            prep_config = deepcopy(self.config)

            prep_config["meteorology"]["horizontal_turbulence"] = prep_config["horizontal_turbulence"]
            prep_config["meteorology"]["vertical_turbulence"] = prep_config["vertical_turbulence"]
            self.met_object = self.data_object.to_meteorology(**prep_config["meteorology"])

        else:
            prep_config = deepcopy(self.config)

            if isinstance(self.start, str):
                self.start = datetime.strptime(self.start, "%Y-%m-%d %H:%M:%S")
            if isinstance(self.end, str):
                self.end = datetime.strptime(self.end, "%Y-%m-%d %H:%M:%S")

            prep_config["preprocessor"]["time_bin_edges"]["start"] = self.start
            prep_config["preprocessor"]["time_bin_edges"]["end"] = self.end
            
            prep_config["preprocessor"]["time_bin_edges"]["freq"] = f"{bin_size}s"

            preprocessor = builder._get_preprocessor(prep_config, self.data_object)
            meteorology_group=preprocessor.met_object

            met_df = pd.DataFrame()
            for key, model in meteorology_group.items():
                model_df = pd.DataFrame(model.__dict__, index=model.time)
                met_df = pd.concat([met_df,model_df],axis=0,ignore_index=True)
            met_df = met_df.drop_duplicates().sort_values(by='time')
            met_dict = met_df.to_dict(orient='list')

            met_dict = {key:np.array(value) for key, value in met_dict.items()}
            
            for key in meteorology_group.keys():
                meteorology_group[key].__dict__ = met_dict
            
            self.meteorology_group = meteorology_group
            self.met_object = Meteorology()
            self.met_object.__dict__ = met_dict
        
    def set_sensor(self,retros):

        self.sensor_object = SensorGroup()

        for name, loc in retros.items():
            sensor = Beam(label=name, n_beam_knots=self.config["sensor"]["n_beam_knots"])

            retro_enu = ENU(
                ref_latitude=self.machine_lla[0],
                ref_longitude=self.machine_lla[1],
                ref_altitude=self.config["plume_floor"],
                east = loc[0],
                north = loc[1],
                up = loc[2]
                )
            
            retro_lla = retro_enu.to_lla().to_array()[0]
            lat_pair, lng_pair, alt_pair = zip(self.machine_lla, retro_lla)

            sensor.location = LLA(
                latitude=np.array(lat_pair), longitude=np.array(lng_pair), altitude=np.array(alt_pair)
            )

            self.sensor_object.add_sensor(sensor)

    def set_sensor_from_data(self):

        self.sensor_object = self.data_object.to_sensor(**self.config["sensor"])

        
    def _build_dispersion_model(self):
        """Construct a Gaussian plume dispersion model directly.

        The pyelq ``builder`` no longer exposes a standalone ``_get_dispersion_model``
        helper: dispersion models are now assembled inside
        ``_get_source_regions_components``. We replicate the relevant part here so the
        forward model can be evaluated directly, without going through the full
        config-driven builder pipeline.
        """

        dispersion_model = GaussianPlume(self.source_map)
        dispersion_model.turbulence_model_horizontal = DraxlerModel.default_horizontal()
        dispersion_model.turbulence_model_vertical = DraxlerModel.default_vertical()
        return dispersion_model

    def forward_model_from_sensor(self):

        dispersion_model = self._build_dispersion_model()

        plume_coupling = dispersion_model.compute_coupling(
            self.sensor_object,
            self.met_object,
            self.gas_species,
            run_interpolation=False
        )
        concs_df = pd.DataFrame()

        for key in plume_coupling.keys():
            concs = np.matmul(plume_coupling[key],self.source_emissions)
            temp_df = pd.DataFrame(np.reshape(concs,(1,-1)),index=[key],columns=self.met_object.time)
            concs_df = pd.concat([concs_df,temp_df],axis=0)

        self.conc_df = concs_df

        return concs_df
    
    def create_simulated_dataframe(self, background=2):
        df = self.forward_model_from_sensor().T
        df += background

        for var, array in self.met_object.__dict__.items():
            df[var] = array
        
        df = df.drop(['time','location','label'],axis=1)
        
        return df

    
    def forward_model_from_points(self,points):

        dispersion_model = self._build_dispersion_model()

        theta = (180 + 90-self.met_object.wind_direction)%360

        theta[theta>180] -= 360
        # if theta > 180:
        #     theta = theta-360
        theta = theta*np.pi/180

        gas_density = self.gas_species.gas_density(self.met_object.temperature,self.met_object.pressure)

        if not(isinstance(points,np.ndarray)):
            points = np.array(points)
        if points.ndim == 1:
            points = np.reshape(points,(1,-1))

        enu_source_array = self.source_map.location.to_array()

        concs = np.zeros([len(self.met_object.time),points.shape[0]])

        for i in range(enu_source_array.shape[0]):
            relative_points = points - enu_source_array[i,:]

            sensor_z = points[:,2]
            source_z = enu_source_array[i,2]

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            couplings = []

            for j in range(len(self.met_object.time)):

                sensor_x = relative_points[:,0]
                sensor_y = relative_points[:,1]

                sensor_x = sensor_x + 0.01*cos_theta[j]
                sensor_y = sensor_y + 0.01*sin_theta[j]

                temp_couplings = dispersion_model.compute_coupling_array(
                    sensor_x,
                    sensor_y,
                    sensor_z,
                    source_z,
                    self.met_object.wind_speed[j],
                    theta[j],
                    self.met_object.wind_turbulence_horizontal[j],
                    self.met_object.wind_turbulence_vertical[j],
                    gas_density[j]
                )

                couplings.append(temp_couplings)

            couplings = np.array(couplings)

            concs += couplings*self.source_emissions[i]

        return concs
    
    def generate_grid_points(self, x_limits=[-100,100],y_limits=[-100,100],z_limits=[0,10]):

        # x_points = np.linspace(*x_limits,int(round(x_limits[1]-x_limits[0])+1))
        # y_points = np.linspace(*y_limits,int(round(y_limits[1]-y_limits[0])+1))

        x_points = np.linspace(*x_limits,200)
        y_points = np.linspace(*y_limits,200)
        z_points = np.linspace(*z_limits,int(round(z_limits[1]-z_limits[0])+1))

        grid_points = np.array([(x,y,z) for z in z_points for y in y_points for x in x_points])
        # dim_1, dim_0, dim_2 = np.meshgrid(y_points, x_points, z_points)
        # grid_points = np.stack([dim_2.flatten(), dim_1.flatten(), dim_0.flatten()], axis=1)
        
        return grid_points, x_points, y_points, z_points

    def plot_sensor_concs(self):
        # pd.DataFrame().stack().reset_index().rename()
        concs = self.conc_df.stack().reset_index().rename({'level_0':'Retros',0:'Concentration (ppm)','level_1':'Time'},axis=1)
        
        fig = px.scatter(concs, x="Time",y="Concentration (ppm)",color="Retros",title="Concentration plot by retro", )
        fig.show()

    def plot_wind(self):

        fig = make_subplots(rows=3, cols=1)

        fig.add_trace(
            go.Scatter(x=self.met_object['time'], y=self.met_object['wind_direction'], mode="markers", name="Wind direction"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=self.met_object['time'], y=self.met_object['wind_speed'], mode="markers", name="Wind speed"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=self.met_object['time'], y=self.met_object['wind_turbulence_horizontal'], mode="markers", name="Horizontal Turbulence"),
            row=3,
            col=1,
        )

        fig.show()
