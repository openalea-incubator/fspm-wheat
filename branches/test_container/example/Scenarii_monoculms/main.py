# -*- coding: latin-1 -*-

import datetime
import logging
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fspmwheat import caribu_facade
from fspmwheat import cnwheat_facade
from fspmwheat import elongwheat_facade
from fspmwheat import farquharwheat_facade
from fspmwheat import growthwheat_facade
from fspmwheat import senescwheat_facade

from alinea.adel.adel_dynamic import AdelDyn
from alinea.adel.echap_leaf import echap_leaves
from alinea.adel.Stand import AgronomicStand

"""
    main
    ~~~~

    An example to show how to couple models CN-Wheat, Farquhar-Wheat, Senesc-Wheat, Elong-Wheat, Growth-Wheat, Adel-Wheat and Caribu.
    This example uses the format MTG to exchange data between the models.

    You must first install :mod:`alinea.adel`, :mod:`cnwheat`, :mod:`farquharwheat`, :mod:`elongwheat`, :mod:`growthwheat` and :mod:`senescwheat` (and add them to your PYTHONPATH)
    before running this script with the command `python`.

    :copyright: Copyright 2014-2016 INRA-ECOSYS, see AUTHORS.
    :license: see LICENSE for details.

"""

'''
    Information about this versioned file:
        $LastChangedBy: mngauthier $
        $LastChangedDate: 2019-01-09 14:10:09 +0100 (mer., 09 janv. 2019) $
        $LastChangedRevision: 115 $
        $URL: https://subversion.renater.fr/authscm/rbarillot/svn/fspm-wheat/trunk/fspmwheat/main.py $
        $Id: main.py 115 2019-01-09 13:10:09Z mngauthier $
'''

random.seed(1234)
np.random.seed(1234)

# number of seconds in 1 hour
HOUR_TO_SECOND_CONVERSION_FACTOR = 3600

INPUTS_OUTPUTS_PRECISION = 8

AXES_INDEX_COLUMNS = ['t', 'plant', 'axis']
ELEMENTS_INDEX_COLUMNS = ['t', 'plant', 'axis', 'metamer', 'organ', 'element']
HIDDENZONES_INDEX_COLUMNS = ['t', 'plant', 'axis', 'metamer']
ORGANS_INDEX_COLUMNS = ['t', 'plant', 'axis', 'organ']
SAM_INDEX_COLUMNS = ['t', 'plant', 'axis']
SOILS_INDEX_COLUMNS = ['t', 'plant', 'axis']

def save_df_to_csv(df, states_filepath):
    try:
        df.to_csv(states_filepath, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))
    except IOError as err:
        path, filename = os.path.split(states_filepath)
        filename = os.path.splitext(filename)[0]
        newfilename = 'ACTUAL_{}.csv'.format(filename)
        newpath = os.path.join(path, newfilename)
        df.to_csv(newpath, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))
        warnings.warn('[{}] {}'.format(err.errno, err.strerror))
        warnings.warn('File will be saved at {}'.format(newpath))


LOGGING_CONFIG_FILEPATH = os.path.join('..', '..', 'logging.json')

LOGGING_LEVEL = logging.INFO  # can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL


def main(stop_time, forced_start_time=0, run_simu=True, run_postprocessing=True, generate_graphs=True, run_from_outputs=False, option_static=False, show_3Dplant=True, tillers_replications=None,
         heterogeneous_canopy=True, N_fertilizations=None, PLANT_DENSITY=None, update_parameters_all_models=None,
         INPUTS_DIRPATH='inputs', METEO_FILENAME='meteo.csv', OUTPUTS_DIRPATH = 'outputs', POSTPROCESSING_DIRPATH = 'postprocessing', GRAPHS_DIRPATH = 'graphs'):
    """
    update_parameters_all_models = {'cnwheat': {'organ1': {'param1': 'val1', 'param2': 'val2'},
                                                'organ2': {'param1': 'val1', 'param2': 'val2'}
                                                },
                                    'elongwheat': {'param1': 'val1', 'param2': 'val2'}
                                    }

    N_fertilizations = {time_step1: n_fertilization_qty, time_step2: n_fertilization_qty} # nitrates concentrations fluctuactes and we add n to the soil
    or N_fertilizations = {'constant_Conc_Nitrates': val} # nitrates concentrations is set to a constant value

    INPUTS_DIRPATH = 'str'
    or INPUTS_DIRPATH = {'adel':str, 'plants':str, 'meteo':str, 'soils':str} #  The directory at path 'adel' must contain files 'adel_pars.RData', 'adel0000.pckl' and 'scene0000.bgeom' for ADELWHEAT
    """

# Define default plant density (culm m-2)
    if PLANT_DENSITY is None:
        PLANT_DENSITY = {1: 250.}

    # inputs
    INPUTS_DIRPATH_DICT = {}
    INPUTS_DIRPATH_DEFAULT = 'inputs'
    if type(INPUTS_DIRPATH) is dict:
        INPUTS_DIRPATH_DICT = INPUTS_DIRPATH
    else:
        INPUTS_DIRPATH_DEFAULT = INPUTS_DIRPATH
    SAM_INPUTS_FILEPATH = os.path.join( INPUTS_DIRPATH_DICT.get('plants', INPUTS_DIRPATH_DEFAULT), 'SAM_inputs.csv')
    ORGANS_INPUTS_FILEPATH = os.path.join( INPUTS_DIRPATH_DICT.get('plants', INPUTS_DIRPATH_DEFAULT), 'organs_inputs.csv')
    HIDDENZONE_INPUTS_FILEPATH = os.path.join( INPUTS_DIRPATH_DICT.get('plants', INPUTS_DIRPATH_DEFAULT), 'hiddenzones_inputs.csv')
    ELEMENTS_INPUTS_FILEPATH = os.path.join( INPUTS_DIRPATH_DICT.get('plants', INPUTS_DIRPATH_DEFAULT), 'elements_inputs.csv')
    SOILS_INPUTS_FILEPATH = os.path.join( INPUTS_DIRPATH_DICT.get('soils', INPUTS_DIRPATH_DEFAULT), 'soils_inputs.csv')
    METEO_FILEPATH = os.path.join( INPUTS_DIRPATH_DICT.get('meteo', INPUTS_DIRPATH_DEFAULT),METEO_FILENAME )

    # the path of the CSV files where to save the states of the modeled system at each step
    AXES_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'axes_states.csv')
    SAM_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'SAM_states.csv')
    ORGANS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'organs_states.csv')
    HIDDENZONES_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'hiddenzones_states.csv')
    ELEMENTS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'elements_states.csv')
    SOILS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'soils_states.csv')

    # post-processing directory path
    AXES_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'axes_postprocessing.csv')
    ORGANS_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'organs_postprocessing.csv')
    HIDDENZONES_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'hiddenzones_postprocessing.csv')
    ELEMENTS_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'elements_postprocessing.csv')
    SOILS_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'soils_postprocessing.csv')

    if run_simu:
        meteo = pd.read_csv(METEO_FILEPATH, index_col='t')

        # define the time step in hours for each simulator
        caribu_ts = 4
        senescwheat_ts = 2
        farquharwheat_ts = 2
        elongwheat_ts = 1
        growthwheat_ts = 1
        cnwheat_ts = 1

        # Create the stand using density pattern
        stand = AgronomicStand(sowing_density=PLANT_DENSITY[1], plant_density=PLANT_DENSITY[1], inter_row=0.15, noise=0.)
        # Create adel_wheat obj and empty mtg
        adel_wheat = AdelDyn(seed=1, scene_unit='m', leaves=echap_leaves(xy_model='Soissons_byleafclass', top_leaves=0), stand=stand)
        axeT_df = pd.read_csv(os.path.join(INPUTS_DIRPATH,'axeTable.csv'))
        nff = 50
        if update_parameters_all_models and 'elongwheat' in update_parameters_all_models:
            nff = update_parameters_all_models['elongwheat'].get('max_nb_leaves', nff)
        axeT_df.HS_final = nff
        axeT_df.nf = nff
        axeT_df.nf_end = nff
        g = adel_wheat.build_stand(axeT_df)

        # create empty dataframes to shared data between the models
        shared_axes_inputs_outputs_df = pd.DataFrame()
        shared_SAM_inputs_outputs_df = pd.DataFrame()
        shared_organs_inputs_outputs_df = pd.DataFrame()
        shared_hiddenzones_inputs_outputs_df = pd.DataFrame()
        shared_elements_inputs_outputs_df = pd.DataFrame()
        shared_soils_inputs_outputs_df = pd.DataFrame()

        # read the inputs
        if run_from_outputs:
            axes_previous_outputs = pd.read_csv(AXES_STATES_FILEPATH)
            organs_previous_outputs = pd.read_csv(ORGANS_STATES_FILEPATH)
            hiddenzones_previous_outputs = pd.read_csv(HIDDENZONES_STATES_FILEPATH)
            elements_previous_outputs = pd.read_csv(ELEMENTS_STATES_FILEPATH)
            SAM_previous_outputs = pd.read_csv(SAM_STATES_FILEPATH)
            soils_previous_outputs = pd.read_csv(SOILS_STATES_FILEPATH)

            assert 't' in hiddenzones_previous_outputs.columns
            if forced_start_time > 0:
                new_start_time = forced_start_time + 1

                axes_previous_outputs = axes_previous_outputs[axes_previous_outputs.t <= forced_start_time]
                organs_previous_outputs = organs_previous_outputs[organs_previous_outputs.t <= forced_start_time]
                hiddenzones_previous_outputs = hiddenzones_previous_outputs[hiddenzones_previous_outputs.t <= forced_start_time]
                elements_previous_outputs = elements_previous_outputs[elements_previous_outputs.t <= forced_start_time]
                SAM_previous_outputs = SAM_previous_outputs[SAM_previous_outputs.t <= forced_start_time]
                soils_previous_outputs = soils_previous_outputs[soils_previous_outputs.t <= forced_start_time]

            else:
                last_t_step = max(hiddenzones_previous_outputs['t'])
                new_start_time = last_t_step + 1

            idx = organs_previous_outputs.groupby([organ_index for organ_index in ORGANS_INDEX_COLUMNS if organ_index != 't'])['t'].transform(max) == organs_previous_outputs['t']
            organs_inputs_t0 = organs_previous_outputs[idx].drop(['t'], axis=1)

            idx = hiddenzones_previous_outputs.groupby([hiddenzone_index for hiddenzone_index in HIDDENZONES_INDEX_COLUMNS if hiddenzone_index != 't']
                                                       )['t'].transform(max) == hiddenzones_previous_outputs['t']
            hiddenzones_inputs_t0 = hiddenzones_previous_outputs[idx].drop(['t'], axis=1)

            elements_previous_outputs_filtered = elements_previous_outputs[~elements_previous_outputs.is_over.isna()]
            idx = elements_previous_outputs_filtered.groupby([element_index for element_index in ELEMENTS_INDEX_COLUMNS if element_index != 't']
                                                             )['t'].transform(max) == elements_previous_outputs_filtered['t']
            elements_inputs_t0 = elements_previous_outputs_filtered[idx].drop(['t'], axis=1)

            idx = SAM_previous_outputs.groupby([SAM_index for SAM_index in SAM_INDEX_COLUMNS if SAM_index != 't'])['t'].transform(max) == SAM_previous_outputs['t']
            SAM_inputs_t0 = SAM_previous_outputs[idx].drop(['t'], axis=1)

            idx = soils_previous_outputs.groupby([soil_index for soil_index in SOILS_INDEX_COLUMNS if soil_index != 't'])['t'].transform(max) == soils_previous_outputs['t']
            soils_inputs_t0 = soils_previous_outputs[idx].drop(['t'], axis=1)

            # Make sure boolean columns have either type bool or float
            bool_columns = ['is_over', 'is_growing', 'leaf_is_emerged', 'internode_is_visible', 'leaf_is_growing', 'internode_is_growing', 'leaf_is_remobilizing', 'internode_is_remobilizing']
            for df in [elements_inputs_t0, hiddenzones_inputs_t0]:
                for cln in bool_columns:
                    if cln in df.keys():
                        df[cln].replace(to_replace='False', value=0.0, inplace=True)
                        df[cln].replace(to_replace='True', value=1.0, inplace=True)
                        df[cln] = pd.to_numeric(df[cln])

        else:
            new_start_time = -1

            organs_inputs_t0 = pd.read_csv(ORGANS_INPUTS_FILEPATH)
            hiddenzones_inputs_t0 = pd.read_csv(HIDDENZONE_INPUTS_FILEPATH)
            elements_inputs_t0 = pd.read_csv(ELEMENTS_INPUTS_FILEPATH)
            SAM_inputs_t0 = pd.read_csv(SAM_INPUTS_FILEPATH)
            soils_inputs_t0 = pd.read_csv(SOILS_INPUTS_FILEPATH)

        # create the facades

        # elongwheat (created first because it is the only facade to add new metamers)
        elongwheat_hiddenzones_inputs_t0 = hiddenzones_inputs_t0[
            elongwheat_facade.converter.HIDDENZONE_TOPOLOGY_COLUMNS + [i for i in elongwheat_facade.simulation.HIDDENZONE_INPUTS if i in
                                                                       hiddenzones_inputs_t0.columns]].copy()
        elongwheat_elements_inputs_t0 = elements_inputs_t0[
            elongwheat_facade.converter.ELEMENT_TOPOLOGY_COLUMNS + [i for i in elongwheat_facade.simulation.ELEMENT_INPUTS if i in
                                                                    elements_inputs_t0.columns]].copy()
        elongwheat_SAM_inputs_t0 = SAM_inputs_t0[
            elongwheat_facade.converter.SAM_TOPOLOGY_COLUMNS + [i for i in elongwheat_facade.simulation.SAM_INPUTS if i in SAM_inputs_t0.columns]].copy()

        # Update parameters if specified
        if update_parameters_all_models and 'elongwheat' in update_parameters_all_models:
            update_parameters_elongwheat = update_parameters_all_models['elongwheat']
        else:
            update_parameters_elongwheat = None
        elongwheat_facade_ = elongwheat_facade.ElongWheatFacade(g,
                                                                elongwheat_ts * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                                elongwheat_SAM_inputs_t0,
                                                                elongwheat_hiddenzones_inputs_t0,
                                                                elongwheat_elements_inputs_t0,
                                                                shared_SAM_inputs_outputs_df,
                                                                shared_hiddenzones_inputs_outputs_df,
                                                                shared_elements_inputs_outputs_df,
                                                                adel_wheat,
                                                                update_parameters_elongwheat)

        # caribu
        caribu_facade_ = caribu_facade.CaribuFacade(g,
                                                    shared_elements_inputs_outputs_df,
                                                    adel_wheat)

        # TODO : est-ce necessaire de selectionner les colonnes en entrees ?

        # senescwheat
        senescwheat_roots_inputs_t0 = organs_inputs_t0.loc[organs_inputs_t0['organ'] == 'roots'][senescwheat_facade.converter.ROOTS_TOPOLOGY_COLUMNS +
                                                                                                 [i for i in senescwheat_facade.converter.SENESCWHEAT_ROOTS_INPUTS if i in
                                                                                                  organs_inputs_t0.columns]].copy()
        senescwheat_elements_inputs_t0 = elements_inputs_t0[senescwheat_facade.converter.ELEMENTS_TOPOLOGY_COLUMNS + [i for i in senescwheat_facade.converter.SENESCWHEAT_ELEMENTS_INPUTS if i in
                                                                                                                      elements_inputs_t0.columns]].copy()
        senescwheat_SAM_inputs_t0 = SAM_inputs_t0[senescwheat_facade.converter.SAM_TOPOLOGY_COLUMNS + [i for i in senescwheat_facade.converter.SENESCWHEAT_SAM_INPUTS if i in
                                                                                                       SAM_inputs_t0.columns]].copy()
        # Update parameters if specified
        if update_parameters_all_models and 'senescwheat' in update_parameters_all_models:
            update_parameters_senescwheat = update_parameters_all_models['senescwheat']
        else:
            update_parameters_senescwheat = None

        senescwheat_facade_ = senescwheat_facade.SenescWheatFacade(g,
                                                                   senescwheat_ts * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                                   senescwheat_roots_inputs_t0,
                                                                   senescwheat_SAM_inputs_t0,
                                                                   senescwheat_elements_inputs_t0,
                                                                   shared_organs_inputs_outputs_df,
                                                                   shared_SAM_inputs_outputs_df,
                                                                   shared_elements_inputs_outputs_df,
                                                                   update_parameters_senescwheat)

        # farquharwheat
        farquharwheat_elements_inputs_t0 = elements_inputs_t0[farquharwheat_facade.converter.ELEMENT_TOPOLOGY_COLUMNS + [i for i in farquharwheat_facade.converter.FARQUHARWHEAT_ELEMENTS_INPUTS if i in
                                                                                                                         elements_inputs_t0.columns]].copy()
        farquharwheat_SAM_inputs_t0 = SAM_inputs_t0[
            farquharwheat_facade.converter.SAM_TOPOLOGY_COLUMNS + [i for i in farquharwheat_facade.converter.FARQUHARWHEAT_SAMS_INPUTS if i in SAM_inputs_t0.columns]].copy()

        farquharwheat_facade_ = farquharwheat_facade.FarquharWheatFacade(g,
                                                                         farquharwheat_elements_inputs_t0,
                                                                         farquharwheat_SAM_inputs_t0,
                                                                         shared_elements_inputs_outputs_df)

        # growthwheat
        growthwheat_hiddenzones_inputs_t0 = hiddenzones_inputs_t0[
            growthwheat_facade.converter.HIDDENZONE_TOPOLOGY_COLUMNS + [i for i in growthwheat_facade.simulation.HIDDENZONE_INPUTS if i in hiddenzones_inputs_t0.columns]].copy()
        growthwheat_elements_inputs_t0 = elements_inputs_t0[
            growthwheat_facade.converter.ELEMENT_TOPOLOGY_COLUMNS + [i for i in growthwheat_facade.simulation.ELEMENT_INPUTS if i in elements_inputs_t0.columns]].copy()
        growthwheat_root_inputs_t0 = organs_inputs_t0.loc[organs_inputs_t0['organ'] == 'roots'][
            growthwheat_facade.converter.ROOT_TOPOLOGY_COLUMNS + [i for i in growthwheat_facade.simulation.ROOT_INPUTS if i in organs_inputs_t0.columns]].copy()
        growthwheat_SAM_inputs_t0 = SAM_inputs_t0[
            growthwheat_facade.converter.SAM_TOPOLOGY_COLUMNS + [i for i in growthwheat_facade.simulation.SAM_INPUTS if i in SAM_inputs_t0.columns]].copy()

        # Update parameters if specified
        if update_parameters_all_models and 'growthwheat' in update_parameters_all_models:
            update_parameters_growthwheat = update_parameters_all_models['growthwheat']
        else:
            update_parameters_growthwheat = None

        growthwheat_facade_ = growthwheat_facade.GrowthWheatFacade(g,
                                                                   growthwheat_ts * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                                   growthwheat_hiddenzones_inputs_t0,
                                                                   growthwheat_elements_inputs_t0,
                                                                   growthwheat_root_inputs_t0,
                                                                   growthwheat_SAM_inputs_t0,
                                                                   shared_organs_inputs_outputs_df,
                                                                   shared_hiddenzones_inputs_outputs_df,
                                                                   shared_elements_inputs_outputs_df,
                                                                   update_parameters_growthwheat)

        # cnwheat
        cnwheat_organs_inputs_t0 = organs_inputs_t0[[i for i in cnwheat_facade.cnwheat_converter.ORGANS_VARIABLES if i in organs_inputs_t0.columns]].copy()
        cnwheat_hiddenzones_inputs_t0 = hiddenzones_inputs_t0[[i for i in cnwheat_facade.cnwheat_converter.HIDDENZONE_VARIABLES if i in hiddenzones_inputs_t0.columns]].copy()
        cnwheat_elements_inputs_t0 = elements_inputs_t0[[i for i in cnwheat_facade.cnwheat_converter.ELEMENTS_VARIABLES if i in elements_inputs_t0.columns]].copy()
        cnwheat_soils_inputs_t0 = soils_inputs_t0[[i for i in cnwheat_facade.cnwheat_converter.SOILS_VARIABLES if i in soils_inputs_t0.columns]].copy()

        # Update parameters if specified
        if update_parameters_all_models and 'cnwheat' in update_parameters_all_models:
            update_parameters_cnwheat = update_parameters_all_models['cnwheat']
        else:
            update_parameters_cnwheat = {}

        cnwheat_facade_ = cnwheat_facade.CNWheatFacade(g,
                                                       cnwheat_ts * HOUR_TO_SECOND_CONVERSION_FACTOR,
                                                       PLANT_DENSITY,
                                                       update_parameters_cnwheat,
                                                       cnwheat_organs_inputs_t0,
                                                       cnwheat_hiddenzones_inputs_t0,
                                                       cnwheat_elements_inputs_t0,
                                                       cnwheat_soils_inputs_t0,
                                                       shared_axes_inputs_outputs_df,
                                                       shared_organs_inputs_outputs_df,
                                                       shared_hiddenzones_inputs_outputs_df,
                                                       shared_elements_inputs_outputs_df,
                                                       shared_soils_inputs_outputs_df)

        # Run cnwheat with constant nitrates concentration in the soil if specified
        if N_fertilizations is not None and len(N_fertilizations) > 0:
            if 'constant_Conc_Nitrates' in N_fertilizations.keys():
                cnwheat_facade_.soils[(1, 'MS')].constant_Conc_Nitrates = True
                cnwheat_facade_.soils[(1, 'MS')].nitrates = N_fertilizations['constant_Conc_Nitrates'] * cnwheat_facade_.soils[(1, 'MS')].volume

        # Update geometry
        adel_wheat.update_geometry(g)
        if show_3Dplant:
            adel_wheat.plot(g)

        # define the start and the end of the whole simulation (in hours)
        start_time = max(0, new_start_time)
        stop_time = stop_time

        # define lists of dataframes to store the inputs and the outputs of the models at each step.
        axes_all_data_list = []
        organs_all_data_list = []  # organs which belong to axes: roots, phloem, grains
        hiddenzones_all_data_list = []
        elements_all_data_list = []
        SAM_all_data_list = []
        soils_all_data_list = []

        all_simulation_steps = []  # to store the steps of the simulation

        print('Simulation starts')

        # run the simulators
        current_time_of_the_system = time.time()

        for t_caribu in range(start_time, stop_time, caribu_ts):

            # run Caribu
            PARi = meteo.loc[t_caribu, ['PARi_MA4']].iloc[0]
            DOY = meteo.loc[t_caribu, ['DOY']].iloc[0]
            hour = meteo.loc[t_caribu, ['hour']].iloc[0]
            caribu_facade_.run(energy=PARi, DOY=DOY, hourTU=hour, latitude=48.85, sun_sky_option='sky', heterogeneous_canopy=heterogeneous_canopy, plant_density=PLANT_DENSITY[1] )
            print('t caribu is {}'.format(t_caribu))

            for t_senescwheat in range(t_caribu, t_caribu + caribu_ts, senescwheat_ts):

                # run SenescWheat
                print('t senescwheat is {}'.format(t_senescwheat))
                senescwheat_facade_.run()

                # Test for dead plant # TODO: adapt in case of multiple plants
                if np.nansum( shared_elements_inputs_outputs_df.loc[shared_elements_inputs_outputs_df['element'].isin(['StemElement','LeafElement1']), 'green_area'] ) == 0:

                    print('\n' '! Simulation stopped because all the emerged elements are fully senescent !')

                    # append the inputs and outputs at current step to global lists
                    all_simulation_steps.append(t_senescwheat)
                    axes_all_data_list.append(shared_axes_inputs_outputs_df.copy())
                    organs_all_data_list.append(shared_organs_inputs_outputs_df.copy())
                    hiddenzones_all_data_list.append(shared_hiddenzones_inputs_outputs_df.copy())
                    elements_all_data_list.append(shared_elements_inputs_outputs_df.copy())
                    SAM_all_data_list.append(shared_SAM_inputs_outputs_df.copy())
                    soils_all_data_list.append(shared_soils_inputs_outputs_df.copy())

                    break

                # Run the rest of the model if the plant is alive
                for t_farquharwheat in range(t_senescwheat, t_senescwheat + senescwheat_ts, farquharwheat_ts):
                    # get the meteo of the current step
                    Ta, ambient_CO2, RH, Ur = meteo.loc[t_farquharwheat, ['air_temperature_MA2', 'ambient_CO2_MA2', 'humidity_MA2', 'Wind_MA2']]

                    # run FarquharWheat
                    farquharwheat_facade_.run(Ta, ambient_CO2, RH, Ur)

                    for t_elongwheat in range(t_farquharwheat, t_farquharwheat + farquharwheat_ts, elongwheat_ts):

                        # run ElongWheat
                        print('t elongwheat is {}'.format(t_elongwheat))
                        Tair, Tsoil = meteo.loc[t_elongwheat, ['air_temperature', 'soil_temperature']]
                        elongwheat_facade_.run(Tair, Tsoil, option_static=option_static)

                        # Update geometry
                        adel_wheat.update_geometry(g)
                        if show_3Dplant:
                            adel_wheat.plot(g)

                        for t_growthwheat in range(t_elongwheat, t_elongwheat + elongwheat_ts, growthwheat_ts):

                            # run GrowthWheat
                            print('t growthwheat is {}'.format(t_growthwheat))
                            growthwheat_facade_.run()

                            for t_cnwheat in range(t_growthwheat, t_growthwheat + growthwheat_ts, cnwheat_ts):
                                if t_cnwheat > 0:

                                    # N fertilization if any
                                    if N_fertilizations is not None and len(N_fertilizations) > 0:
                                        if t_cnwheat in N_fertilizations.keys():
                                            cnwheat_facade_.soils[(1, 'MS')].nitrates += N_fertilizations[t_cnwheat]

                                    # run CNWheat
                                    print('t cnwheat is {}'.format(t_cnwheat))
                                    Tair = meteo.loc[t_elongwheat, 'air_temperature']
                                    Tsoil = meteo.loc[t_elongwheat, 'soil_temperature']
                                    cnwheat_facade_.run(Tair, Tsoil, tillers_replications)

                                # append the inputs and outputs at current step to global lists
                                all_simulation_steps.append(t_cnwheat)
                                axes_all_data_list.append(shared_axes_inputs_outputs_df.copy())
                                organs_all_data_list.append(shared_organs_inputs_outputs_df.copy())
                                hiddenzones_all_data_list.append(shared_hiddenzones_inputs_outputs_df.copy())
                                elements_all_data_list.append(shared_elements_inputs_outputs_df.copy())
                                SAM_all_data_list.append(shared_SAM_inputs_outputs_df.copy())
                                soils_all_data_list.append(shared_soils_inputs_outputs_df.copy())

            else:
                # Continue if SenescWheat loop wasn't broken because of dead plant.
                continue
            # SenescWheat loop was broken, break the Caribu loop.
            break

        execution_time = int(time.time() - current_time_of_the_system)
        print ('\n' 'Simulation run in {}'.format(str(datetime.timedelta(seconds=execution_time))))

        # convert list of outputs into dataframs
        all_axes_inputs_outputs = pd.concat(axes_all_data_list, keys=all_simulation_steps, sort=False)
        all_axes_inputs_outputs.reset_index(0, inplace=True)
        all_axes_inputs_outputs.rename({'level_0': 't'}, axis=1, inplace=True)
        all_axes_inputs_outputs = all_axes_inputs_outputs.reindex(AXES_INDEX_COLUMNS + all_axes_inputs_outputs.columns.difference(AXES_INDEX_COLUMNS).tolist(), axis=1, copy=False)

        all_organs_inputs_outputs = pd.concat(organs_all_data_list, keys=all_simulation_steps, sort=False)
        all_organs_inputs_outputs.reset_index(0, inplace=True)
        all_organs_inputs_outputs.rename({'level_0': 't'}, axis=1, inplace=True)
        all_organs_inputs_outputs = all_organs_inputs_outputs.reindex(ORGANS_INDEX_COLUMNS + all_organs_inputs_outputs.columns.difference(ORGANS_INDEX_COLUMNS).tolist(), axis=1, copy=False)

        all_hiddenzones_inputs_outputs = pd.concat(hiddenzones_all_data_list, keys=all_simulation_steps, sort=False)
        all_hiddenzones_inputs_outputs.reset_index(0, inplace=True)
        all_hiddenzones_inputs_outputs.rename({'level_0': 't'}, axis=1, inplace=True)
        all_hiddenzones_inputs_outputs = all_hiddenzones_inputs_outputs.reindex(HIDDENZONES_INDEX_COLUMNS + all_hiddenzones_inputs_outputs.columns.difference(HIDDENZONES_INDEX_COLUMNS).tolist(),
                                                                                axis=1,
                                                                                copy=False)

        all_SAM_inputs_outputs = pd.concat(SAM_all_data_list, keys=all_simulation_steps, sort=False)
        all_SAM_inputs_outputs.reset_index(0, inplace=True)
        all_SAM_inputs_outputs.rename({'level_0': 't'}, axis=1, inplace=True)
        all_SAM_inputs_outputs = all_SAM_inputs_outputs.reindex(SAM_INDEX_COLUMNS + all_SAM_inputs_outputs.columns.difference(SAM_INDEX_COLUMNS).tolist(), axis=1, copy=False)

        all_elements_inputs_outputs = pd.concat(elements_all_data_list, keys=all_simulation_steps, sort=False)
        all_elements_inputs_outputs.reset_index(0, inplace=True)
        all_elements_inputs_outputs.rename({'level_0': 't'}, axis=1, inplace=True)
        all_elements_inputs_outputs = all_elements_inputs_outputs.reindex(ELEMENTS_INDEX_COLUMNS + all_elements_inputs_outputs.columns.difference(ELEMENTS_INDEX_COLUMNS).tolist(), axis=1, copy=False)

        all_soils_inputs_outputs = pd.concat(soils_all_data_list, keys=all_simulation_steps, sort=False)
        all_soils_inputs_outputs.reset_index(0, inplace=True)
        all_soils_inputs_outputs.rename({'level_0': 't'}, axis=1, inplace=True)
        all_soils_inputs_outputs = all_soils_inputs_outputs.reindex(SOILS_INDEX_COLUMNS + all_soils_inputs_outputs.columns.difference(SOILS_INDEX_COLUMNS).tolist(), axis=1, copy=False)

        # write all inputs and outputs to CSV files
        if run_from_outputs:
            all_axes_inputs_outputs = pd.concat([axes_previous_outputs, all_axes_inputs_outputs], sort=False)
            all_organs_inputs_outputs = pd.concat([organs_previous_outputs, all_organs_inputs_outputs], sort=False)
            all_hiddenzones_inputs_outputs = pd.concat([hiddenzones_previous_outputs, all_hiddenzones_inputs_outputs], sort=False)
            all_SAM_inputs_outputs = pd.concat([SAM_previous_outputs, all_SAM_inputs_outputs], sort=False)
            all_elements_inputs_outputs = pd.concat([elements_previous_outputs, all_elements_inputs_outputs], sort=False)
            all_soils_inputs_outputs = pd.concat([soils_previous_outputs, all_soils_inputs_outputs], sort=False)

        save_df_to_csv(all_axes_inputs_outputs, AXES_STATES_FILEPATH)
        save_df_to_csv(all_organs_inputs_outputs, ORGANS_STATES_FILEPATH)
        save_df_to_csv(all_hiddenzones_inputs_outputs, HIDDENZONES_STATES_FILEPATH)
        save_df_to_csv(all_SAM_inputs_outputs, SAM_STATES_FILEPATH)
        save_df_to_csv(all_elements_inputs_outputs, ELEMENTS_STATES_FILEPATH)
        save_df_to_csv(all_soils_inputs_outputs, SOILS_STATES_FILEPATH)

    if run_postprocessing:
        states_df_dict = {}
        for states_filepath in (AXES_STATES_FILEPATH,
                                ORGANS_STATES_FILEPATH,
                                HIDDENZONES_STATES_FILEPATH,
                                ELEMENTS_STATES_FILEPATH,
                                SOILS_STATES_FILEPATH):
            # assert states_filepaths were not opened during simulation run meaning that other filenames were saved
            path, filename = os.path.split(states_filepath)
            filename = os.path.splitext(filename)[0]
            newfilename = 'ACTUAL_{}.csv'.format(filename)
            newpath = os.path.join(path, newfilename)
            assert not os.path.isfile(newpath), \
                "File {} was saved because {} was opened during simulation run. Rename it before running postprocessing".format(newfilename, states_filepath)

            # Retrieve outputs dataframes from precedent simulation run
            states_df = pd.read_csv(states_filepath)
            states_file_basename = os.path.basename(states_filepath).split('.')[0]
            states_df_dict[states_file_basename] = states_df
        time_grid = states_df_dict.values()[0].t
        delta_t = (time_grid.unique()[1] - time_grid.unique()[0]) * HOUR_TO_SECOND_CONVERSION_FACTOR

        # run the postprocessing
        axes_postprocessing_file_basename = os.path.basename(AXES_POSTPROCESSING_FILEPATH).split('.')[0]
        hiddenzones_postprocessing_file_basename = os.path.basename(HIDDENZONES_POSTPROCESSING_FILEPATH).split('.')[0]
        organs_postprocessing_file_basename = os.path.basename(ORGANS_POSTPROCESSING_FILEPATH).split('.')[0]
        elements_postprocessing_file_basename = os.path.basename(ELEMENTS_POSTPROCESSING_FILEPATH).split('.')[0]
        soils_postprocessing_file_basename = os.path.basename(SOILS_POSTPROCESSING_FILEPATH).split('.')[0]
        postprocessing_df_dict = {}
        (postprocessing_df_dict[axes_postprocessing_file_basename],
         postprocessing_df_dict[hiddenzones_postprocessing_file_basename],
         postprocessing_df_dict[organs_postprocessing_file_basename],
         postprocessing_df_dict[elements_postprocessing_file_basename],
         postprocessing_df_dict[soils_postprocessing_file_basename]) \
            = cnwheat_facade.CNWheatFacade.postprocessing(axes_outputs_df=states_df_dict[os.path.basename(AXES_STATES_FILEPATH).split('.')[0]],
                                                          hiddenzone_outputs_df=states_df_dict[os.path.basename(HIDDENZONES_STATES_FILEPATH).split('.')[0]],
                                                          organs_outputs_df=states_df_dict[os.path.basename(ORGANS_STATES_FILEPATH).split('.')[0]],
                                                          elements_outputs_df=states_df_dict[os.path.basename(ELEMENTS_STATES_FILEPATH).split('.')[0]],
                                                          soils_outputs_df=states_df_dict[os.path.basename(SOILS_STATES_FILEPATH).split('.')[0]],
                                                          delta_t=delta_t)

        # save the postprocessing to disk
        for postprocessing_file_basename, postprocessing_filepath in ((axes_postprocessing_file_basename, AXES_POSTPROCESSING_FILEPATH),
                                                                      (hiddenzones_postprocessing_file_basename, HIDDENZONES_POSTPROCESSING_FILEPATH),
                                                                      (organs_postprocessing_file_basename, ORGANS_POSTPROCESSING_FILEPATH),
                                                                      (elements_postprocessing_file_basename, ELEMENTS_POSTPROCESSING_FILEPATH),
                                                                      (soils_postprocessing_file_basename, SOILS_POSTPROCESSING_FILEPATH)):
            postprocessing_df_dict[postprocessing_file_basename].to_csv(postprocessing_filepath, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION + 5))

    if generate_graphs:
        plt.ioff()
        # Retrieve last computed post-processing dataframes
        df_SAM = pd.read_csv(SAM_STATES_FILEPATH)
        axes_postprocessing_file_basename = os.path.basename(AXES_POSTPROCESSING_FILEPATH).split('.')[0]
        organs_postprocessing_file_basename = os.path.basename(ORGANS_POSTPROCESSING_FILEPATH).split('.')[0]
        hiddenzones_postprocessing_file_basename = os.path.basename(HIDDENZONES_POSTPROCESSING_FILEPATH).split('.')[0]
        elements_postprocessing_file_basename = os.path.basename(ELEMENTS_POSTPROCESSING_FILEPATH).split('.')[0]
        soils_postprocessing_file_basename = os.path.basename(SOILS_POSTPROCESSING_FILEPATH).split('.')[0]
        postprocessing_df_dict = {}
        for (postprocessing_filepath, postprocessing_file_basename) in ((AXES_POSTPROCESSING_FILEPATH, axes_postprocessing_file_basename),
                                                                        (ORGANS_POSTPROCESSING_FILEPATH, organs_postprocessing_file_basename),
                                                                        (HIDDENZONES_POSTPROCESSING_FILEPATH, hiddenzones_postprocessing_file_basename),
                                                                        (ELEMENTS_POSTPROCESSING_FILEPATH, elements_postprocessing_file_basename),
                                                                        (SOILS_POSTPROCESSING_FILEPATH, soils_postprocessing_file_basename)):
            postprocessing_df = pd.read_csv(postprocessing_filepath)
            postprocessing_df_dict[postprocessing_file_basename] = postprocessing_df

        ## --- Generate graphs from postprocessing files
        df_hz = postprocessing_df_dict[hiddenzones_postprocessing_file_basename]
        df_elt = postprocessing_df_dict[elements_postprocessing_file_basename]
        cnwheat_facade.CNWheatFacade.graphs(axes_postprocessing_df=postprocessing_df_dict[axes_postprocessing_file_basename],
                                            hiddenzones_postprocessing_df=df_hz[df_hz['axis'] == 'MS'],  # postprocessing_df_dict[hiddenzones_postprocessing_file_basename],
                                            organs_postprocessing_df=postprocessing_df_dict[organs_postprocessing_file_basename],
                                            elements_postprocessing_df=df_elt[df_elt['axis'] == 'MS'],  # postprocessing_df_dict[elements_postprocessing_file_basename],
                                            soils_postprocessing_df=postprocessing_df_dict[soils_postprocessing_file_basename],
                                            graphs_dirpath=GRAPHS_DIRPATH)

        ## --- Additional graphs
        from cnwheat import tools as cnwheat_tools
        colors = ['blue', 'darkorange', 'green', 'red', 'darkviolet', 'gold', 'magenta', 'brown', 'darkcyan', 'grey', 'lime']
        colors = colors + colors

        # 0) Phyllochron
        df_SAM = df_SAM[df_SAM['axis'] == 'MS']
        df_hz = postprocessing_df_dict[hiddenzones_postprocessing_file_basename]
        grouped_df = df_hz[df_hz['axis'] == 'MS'].groupby(['plant', 'metamer'])[['t', 'leaf_is_emerged']]
        leaf_emergence = {}
        for group_name, data in grouped_df:
            plant, metamer = group_name[0], group_name[1]
            if metamer == 3 or True not in data['leaf_is_emerged'].unique(): continue
            leaf_emergence_t = data[data['leaf_is_emerged'] == True].iloc[0]['t']
            leaf_emergence[(plant, metamer)] = leaf_emergence_t

        phyllochron = {'plant': [], 'metamer': [], 'phyllochron': []}
        for key, leaf_emergence_t in sorted(leaf_emergence.items()):
            plant, metamer = key[0], key[1]
            if metamer == 4: continue
            phyllochron['plant'].append(plant)
            phyllochron['metamer'].append(metamer)
            prev_leaf_emergence_t = leaf_emergence[(plant, metamer - 1)]
            if df_SAM[(df_SAM['t'] == leaf_emergence_t) | (df_SAM['t'] == prev_leaf_emergence_t)].sum_TT.count() == 2:
                phyllo_DD = df_SAM[(df_SAM['t'] == leaf_emergence_t)].sum_TT.values[0] - df_SAM[(df_SAM['t'] == prev_leaf_emergence_t)].sum_TT.values[0]
            else:
                phyllo_DD = np.nan
            phyllochron['phyllochron'].append(phyllo_DD)

        if len(phyllochron['metamer']) > 0:
            plt.figure()
            plt.xlim((int(min(phyllochron['metamer']) - 1), int(max(phyllochron['metamer']) + 1)))
            plt.ylim(ymin=0, ymax=150)
            ax = plt.subplot(111)
            ax.plot(phyllochron['metamer'], phyllochron['phyllochron'], color='b', marker='o')
            for i, j in zip(phyllochron['metamer'], phyllochron['phyllochron']):
                ax.annotate(str(int(round(j, 0))), xy=(i, j + 2), ha='center')
            ax.set_xlabel('Leaf number')
            ax.set_ylabel('Phyllochron (Degree Day)')
            ax.set_title('phyllochron')
            plt.savefig(os.path.join(GRAPHS_DIRPATH, 'phyllochron' + '.PNG'))
            plt.close()

        # 1) Comparison Dimensions with Ljutovac 2002
        bchmk = pd.read_csv(r'inputs\Ljutovac2002.csv')
        res = pd.read_csv(HIDDENZONES_STATES_FILEPATH)
        res = res[(res['axis'] == 'MS') & (res['plant'] == 1) & ~np.isnan(res.leaf_Lmax)].copy()
        res_IN = res[~ np.isnan(res.internode_Lmax)]
        last_value_idx = res.groupby(['metamer'])['t'].transform(max) == res['t']
        res = res[last_value_idx].copy()
        res['lamina_Wmax'] = res.leaf_Wmax
        res['lamina_W_Lg'] = res.leaf_Wmax / res.lamina_Lmax
        bchmk = bchmk[bchmk.metamer >= min(res.metamer)]
        bchmk['lamina_W_Lg'] = bchmk.lamina_Wmax / bchmk.lamina_Lmax
        last_value_idx = res_IN.groupby(['metamer'])['t'].transform(max) == res_IN['t']
        res_IN = res_IN[last_value_idx].copy()
        res = res[['metamer', 'leaf_Lmax', 'lamina_Lmax', 'sheath_Lmax', 'lamina_Wmax', 'lamina_W_Lg', 'SSLW', 'LSSW']].merge(res_IN[['metamer', 'internode_Lmax']], left_on='metamer',
                                                                                                                              right_on='metamer', how='outer').copy()

        var_list = ['leaf_Lmax', 'lamina_Lmax', 'sheath_Lmax', 'lamina_Wmax', 'internode_Lmax']
        for var in list(var_list):
            plt.figure()
            plt.xlim((int(min(res.metamer) - 1), int(max(res.metamer) + 1)))
            plt.ylim(ymin=0, ymax=np.nanmax(list(res[var] * 100 * 1.05) + list(bchmk[var] * 1.05)))

            ax = plt.subplot(111)

            tmp = res[['metamer', var]].drop_duplicates()

            line1 = ax.plot(tmp.metamer, tmp[var] * 100, color='c', marker='o')
            line2 = ax.plot(bchmk.metamer, bchmk[var], color='orange', marker='o')

            ax.set_ylabel(var + ' (cm)')
            ax.set_title(var)
            ax.legend((line1[0], line2[0]), ('Simulation', 'Ljutovac 2002'), loc=2)
            plt.savefig(os.path.join(GRAPHS_DIRPATH, var + '.PNG'))
            plt.close()

        var = 'lamina_W_Lg'
        plt.figure()
        plt.xlim((int(min(res.metamer) - 1), int(max(res.metamer) + 1)))
        plt.ylim(ymin=0, ymax=np.nanmax(list(res[var] * 1.05) + list(bchmk[var] * 1.05)))
        ax = plt.subplot(111)
        tmp = res[['metamer', var]].drop_duplicates()
        line1 = ax.plot(tmp.metamer, tmp[var], color='c', marker='o')
        line2 = ax.plot(bchmk.metamer, bchmk[var], color='orange', marker='o')
        ax.set_ylabel(var)
        ax.set_title(var)
        ax.legend((line1[0], line2[0]), ('Simulation', 'Ljutovac 2002'), loc=2)
        plt.savefig(os.path.join(GRAPHS_DIRPATH, var + '.PNG'))
        plt.close()

        # 1bis) Comparison Structural Masses vs. adaptation from Bertheloot 2008

        # SSLW Laminae
        bchmk = pd.DataFrame.from_dict({1: 15, 2: 23, 3: 25, 4: 18, 5: 22, 6: 25, 7: 20, 8: 23, 9: 26, 10: 28, 11: 31}, orient='index').rename(columns={0: 'SSLW'})
        bchmk.index.name = 'metamer'
        bchmk = bchmk.reset_index()
        bchmk = bchmk[bchmk.metamer >= min(res.metamer)]

        plt.figure()
        plt.xlim((int(min(res.metamer) - 1), int(max(res.metamer) + 1)))
        plt.ylim(ymin=0, ymax=50)
        ax = plt.subplot(111)

        tmp = res[['metamer', 'SSLW']].drop_duplicates()

        line1 = ax.plot(tmp.metamer, tmp.SSLW, color='c', marker='o')
        line2 = ax.plot(bchmk.metamer, bchmk.SSLW, color='orange', marker='o')

        ax.set_ylabel('Structural Specific Lamina Weight (g.m-2)')
        ax.set_title('Structural Specific Lamina Weight')
        ax.legend((line1[0], line2[0]), ('Simulation', 'adapated from Bertheloot 2008'), loc=3)
        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'SSLW.PNG'))
        plt.close()

        # LWS Sheaths
        bchmk = pd.DataFrame.from_dict({1: 0.08, 2: 0.09, 3: 0.11, 4: 0.18, 5: 0.17, 6: 0.21, 7: 0.24, 8: 0.4, 9: 0.5, 10: 0.55, 11: 0.65}, orient='index').rename(columns={0: 'LSSW'})
        bchmk.index.name = 'metamer'
        bchmk = bchmk.reset_index()
        bchmk = bchmk[bchmk.metamer >= min(res.metamer)]

        plt.figure()
        plt.xlim((int(min(res.metamer) - 1), int(max(res.metamer) + 1)))
        plt.ylim(ymin=0, ymax=0.8)
        ax = plt.subplot(111)

        tmp = res[['metamer', 'LSSW']].drop_duplicates()

        line1 = ax.plot(tmp.metamer, tmp.LSSW, color='c', marker='o')
        line2 = ax.plot(bchmk.metamer, bchmk.LSSW, color='orange', marker='o')

        ax.set_ylabel('Lineic Structural Sheath Weight (g.m-1)')
        ax.set_title('Lineic Structural Sheath Weight')
        ax.legend((line1[0], line2[0]), ('Simulation', 'adapated from Bertheloot 2008'), loc=2)
        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'LSSW.PNG'))
        plt.close()

        # 2) LAI
        df_elt['green_area_rep'] = df_elt.green_area * df_elt.nb_replications
        grouped_df = df_elt[(df_elt.axis == 'MS') & (df_elt.element == 'LeafElement1')].groupby(['t', 'plant'])
        LAI_dict = {'t': [], 'plant': [], 'LAI': []}
        for name, data in grouped_df:
            t, plant = name[0], name[1]
            LAI_dict['t'].append(t)
            LAI_dict['plant'].append(plant)
            LAI_dict['LAI'].append(data['green_area_rep'].sum() * PLANT_DENSITY[plant])

        cnwheat_tools.plot_cnwheat_ouputs(pd.DataFrame(LAI_dict), 't', 'LAI', x_label='Time (Hour)', y_label='LAI', plot_filepath=os.path.join(GRAPHS_DIRPATH, 'LAI.PNG'), explicit_label=False)

        # 3) RUE
        df_elt['day'] = df_elt['t'] // 24 + 1
        days = df_elt['day'].unique()
        df_elt['PARa_MJ'] = df_elt['PARa'] * df_elt['green_area'] * df_elt['nb_replications'] * 3600 / 4.6 * 10 ** -6  # Il faudrait idealement utiliser les calculcs green_area et PARa des talles
        df_elt['RGa_MJ'] = df_elt['PARa'] * df_elt['green_area'] * df_elt['nb_replications'] * 3600 / 2.02 * 10 ** -6  # Il faudrait idealement utiliser les calculcs green_area et PARa des talles
        PARa = df_elt.groupby(['day'])['PARa_MJ'].agg('sum')
        PARa_cum = np.cumsum(PARa)

        df_axe = postprocessing_df_dict[axes_postprocessing_file_basename]
        df_axe['day'] = df_axe['t'] // 24 + 1
        Total_Photosynthesis = df_axe.groupby(['day'])['Tillers_Photosynthesis'].agg('sum')
        sum_dry_mass_shoot = df_axe.groupby(['day'])['sum_dry_mass_shoot'].agg('max')
        sum_dry_mass = df_axe.groupby(['day'])['sum_dry_mass'].agg('max')

        RUE_shoot = np.polyfit(PARa_cum, sum_dry_mass_shoot, 1)[0]
        RUE_plant = np.polyfit(PARa_cum, sum_dry_mass, 1)[0]

        fig, ax = plt.subplots()
        ax.plot(PARa_cum, sum_dry_mass_shoot, label='Shoot dry mass (g)')
        ax.plot(PARa_cum, sum_dry_mass, label='Plant dry mass (g)')
        ax.legend(prop={'size': 10}, framealpha=0.5, loc='center left', bbox_to_anchor=(1, 0.815), borderaxespad=0.)
        ax.set_xlabel('Cumulative absorbed PAR (MJ)')
        ax.set_ylabel('Dry mass (g)')
        ax.set_title('RUE')
        plt.text(max(PARa_cum) * 0.02, max(sum_dry_mass) * 0.95, 'RUE shoot : {0:.2f} , RUE plant : {1:.2f}'.format(round(RUE_shoot, 2), round(RUE_plant, 2)))
        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'RUE.PNG'), dpi=200, format='PNG', bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(days, sum_dry_mass_shoot, label='Shoot dry mass (g)')
        ax.plot(days, sum_dry_mass, label='Plant dry mass (g)')
        ax.plot(days, PARa_cum, label='Cumulative absorbed PAR (MJ)')
        ax.legend(prop={'size': 10}, framealpha=0.5, loc='center left', bbox_to_anchor=(1, 0.815), borderaxespad=0.)
        ax.set_xlabel('Days')
        ax.set_title('RUE investigations')
        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'RUE2.PNG'), dpi=200, format='PNG', bbox_inches='tight')
        plt.close()

        # 3bis) Photosynthetic efficiency of the plant
        df_elt['Photosynthesis_tillers'] = df_elt.Ag * df_elt.green_area * df_elt.nb_replications.fillna(1.)
        df_elt['PARa_tot_tillers'] = df_elt.PARa * df_elt.green_area * df_elt.nb_replications.fillna(1.)
        df_elt['green_area_tillers'] = df_elt.green_area * df_elt.nb_replications.fillna(1.)
        photo_y = df_elt.groupby(['t'],as_index=False).agg({'Photosynthesis_tillers':'sum', 'PARa_tot_tillers':'sum', 'green_area_tillers':'sum'})
        photo_y['Photosynthetic_efficiency_plant'] = photo_y.Photosynthesis_tillers / photo_y.PARa_tot_tillers

        fig, ax = plt.subplots()
        ax.plot(photo_y.t, photo_y.Photosynthetic_efficiency_plant)
        ax.set_ylim(bottom=0.)
        ax.set_xlabel('t')
        ax.set_ylabel(u'Photosynthetic efficiency (�mol C/�mol PARa)')
        ax.set_title('Photosynthetic efficiency of the plant')
        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'Photosynthetic_efficiency_plant.PNG'), dpi=200, format='PNG', bbox_inches='tight')
        plt.close()

        PARa2 = df_elt.groupby(['day'])['PARa_tot_tillers'].agg('sum')
        PARa2_cum = np.cumsum(PARa2)
        Photosynthesis = df_elt.groupby(['day'])['Photosynthesis_tillers'].agg('sum')
        Photosynthesis_cum = np.cumsum(Photosynthesis)

        avg_photo_y = np.polyfit(PARa2_cum, Photosynthesis_cum, 1)[0]

        fig, ax = plt.subplots()
        ax.plot(PARa2_cum, Photosynthesis_cum)
        ax.set_xlabel(u'Cumulative absorbed PAR (�mol.s$^{-1}$)')
        ax.set_ylabel(u'Cumulative photosynthesis (�mol C.s$^{-1}$)')
        ax.set_title('Average photosynthetic efficiency of the plant: {:.2%}'.format(round(avg_photo_y, 3)))
        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'Photosynthetic_efficiency_plant2.PNG'), dpi=200, format='PNG', bbox_inches='tight')
        plt.close()

        # 4) Sum thermal time
        df_SAM = df_SAM[df_SAM['axis'] == 'MS']
        fig, ax = plt.subplots()
        ax.plot(df_SAM['t'], df_SAM['sum_TT'])
        ax.set_xlabel('Hours')
        ax.set_ylabel('Thermal Time')
        ax.set_title('Thermal Time')
        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'SumTT.PNG'), dpi=200, format='PNG', bbox_inches='tight')
        plt.close()

        # 5) Residual N : ratio_N_mstruct_max
        df_elt_outputs = pd.read_csv(ELEMENTS_STATES_FILEPATH)
        df_elt_outputs = df_elt_outputs.loc[df_elt_outputs.axis == 'MS']
        df_elt_outputs = df_elt_outputs.loc[df_elt_outputs.mstruct != 0]
        df_elt_outputs['N_content_total'] = df_elt_outputs['N_content_total'] * 100
        x_name = 't'
        x_label = 'Time (Hour)'
        graph_variables_ph_elements = {'N_content_total': u'N content in green + senesced tissues (% mstruct)'}
        for org_ph in (['blade'], ['sheath'], ['internode'], ['peduncle', 'ear']):
            for variable_name, variable_label in graph_variables_ph_elements.items():
                graph_name = variable_name + '_' + '_'.join(org_ph) + '.PNG'
                cnwheat_tools.plot_cnwheat_ouputs(df_elt_outputs,
                                                  x_name=x_name,
                                                  y_name=variable_name,
                                                  x_label=x_label,
                                                  y_label=variable_label,
                                                  colors=[colors[i - 1] for i in df_elt_outputs.metamer.unique().tolist()],
                                                  filters={'organ': org_ph},
                                                  plot_filepath=os.path.join(GRAPHS_DIRPATH, graph_name),
                                                  explicit_label=False)

        # 6) C usages relatif to Net Photosynthesis
        df_org = postprocessing_df_dict[organs_postprocessing_file_basename]
        df_roots = df_org[df_org['organ'] == 'roots'].copy()
        df_roots['day'] = df_roots['t'] // 24 + 1
        df_phloem = df_org[df_org['organ'] == 'phloem'].copy()
        df_phloem['day'] = df_phloem['t'] // 24 + 1

        # --- C usages relatif to Net Photosynthesis
        AMINO_ACIDS_C_RATIO = 4.15  #: Mean number of mol of C in 1 mol of the major amino acids of plants (Glu, Gln, Ser, Asp, Ala, Gly)
        AMINO_ACIDS_N_RATIO = 1.25  #: Mean number of mol of N in 1 mol of the major amino acids of plants (Glu, Gln, Ser, Asp, Ala, Gly)

        # Photosynthesis
        df_elt['Photosynthesis_tillers'] = df_elt['Photosynthesis'].fillna(0) * df_elt['nb_replications'].fillna(1.)
        Tillers_Photosynthesis_Ag = df_elt.groupby(['t'], as_index=False).agg({'Photosynthesis_tillers': 'sum'})
        C_usages = pd.DataFrame({'t': Tillers_Photosynthesis_Ag['t']})
        C_usages['C_produced'] = np.cumsum(Tillers_Photosynthesis_Ag.Photosynthesis_tillers)

        # Respiration
        C_usages['Respi_roots'] = np.cumsum(df_axe.C_respired_roots)
        C_usages['Respi_shoot'] = np.cumsum(df_axe.C_respired_shoot)

        # Exudation
        C_usages['exudation'] = np.cumsum(df_axe.C_exudated.fillna(0))

        # Structural growth
        C_consumption_mstruct_roots = df_roots.sucrose_consumption_mstruct.fillna(0) + df_roots.AA_consumption_mstruct.fillna(0) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO
        C_usages['Structure_roots'] = np.cumsum(C_consumption_mstruct_roots.reset_index(drop=True))

        df_hz['C_consumption_mstruct'] = df_hz.sucrose_consumption_mstruct.fillna(0) + df_hz.AA_consumption_mstruct.fillna(0) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO
        df_hz['C_consumption_mstruct_tillers'] = df_hz['C_consumption_mstruct'] * df_hz['nb_replications']
        C_consumption_mstruct_shoot = df_hz.groupby(['t'])['C_consumption_mstruct_tillers'].sum()
        C_usages['Structure_shoot'] = np.cumsum(C_consumption_mstruct_shoot.reset_index(drop=True))

        # Non structural C
        df_phloem['C_NS'] = df_phloem.sucrose.fillna(0) + df_phloem.amino_acids.fillna(0) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO
        C_NS_phloem_init = df_phloem.C_NS - df_phloem.C_NS[0]
        C_usages['NS_phloem'] = C_NS_phloem_init.reset_index(drop=True)

        df_elt['C_NS'] = df_elt.sucrose.fillna(0) + df_elt.fructan.fillna(0) + df_elt.starch.fillna(0) + (
                    df_elt.amino_acids.fillna(0) + df_elt.proteins.fillna(0)) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO
        df_elt['C_NS_tillers'] = df_elt['C_NS'] * df_elt['nb_replications'].fillna(1.)
        C_elt = df_elt.groupby(['t']).agg({'C_NS_tillers': 'sum'})

        df_hz['C_NS'] = df_hz.sucrose.fillna(0) + df_hz.fructan.fillna(0) + (df_hz.amino_acids.fillna(0) + df_hz.proteins.fillna(0)) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO
        df_hz['C_NS_tillers'] = df_hz['C_NS'] * df_hz['nb_replications'].fillna(1.)
        C_hz = df_hz.groupby(['t']).agg({'C_NS_tillers': 'sum'})

        df_roots['C_NS'] = df_roots.sucrose.fillna(0) + df_roots.amino_acids.fillna(0) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO

        C_NS_autre = df_roots.C_NS.reset_index(drop=True) + C_elt.C_NS_tillers + C_hz.C_NS_tillers
        C_NS_autre_init = C_NS_autre - C_NS_autre[0]
        C_usages['NS_other'] = C_NS_autre_init.reset_index(drop=True)

        # Total
        C_usages['C_budget'] = (C_usages.Respi_roots + C_usages.Respi_shoot + C_usages.exudation + C_usages.Structure_roots + C_usages.Structure_shoot + C_usages.NS_phloem + C_usages.NS_other) / \
                               C_usages.C_produced

        # ----- Graph
        fig, ax = plt.subplots()
        ax.plot(C_usages.t, C_usages.Structure_shoot / C_usages.C_produced * 100,
                label=u'Structural mass - Shoot', color='g')
        ax.plot(C_usages.t, C_usages.Structure_roots / C_usages.C_produced * 100,
                label=u'Structural mass - Roots', color='r')
        ax.plot(C_usages.t, (C_usages.NS_phloem + C_usages.NS_other) / C_usages.C_produced * 100, label=u'Non-structural C', color='darkorange')
        ax.plot(C_usages.t, (C_usages.Respi_roots + C_usages.Respi_shoot) / C_usages.C_produced * 100, label=u'C loss by respiration', color='b')
        ax.plot(C_usages.t, C_usages.exudation / C_usages.C_produced * 100, label=u'C loss by exudation', color='c')

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('Time (h)')
        ax.set_ylabel(u'Carbon usages : Photosynthesis (%)')
        ax.set_ylim(bottom=0, top=100.)

        fig.suptitle(u'Total cumulated usages are ' + str(round(C_usages.C_budget.tail(1)*100, 0)) + u' % of Photosynthesis')

        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'C_usages_cumulated.PNG'), format='PNG', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main(2000, forced_start_time=1990, run_simu=False, run_postprocessing=True, generate_graphs=True, run_from_outputs=True,
         option_static=False, tillers_replications={'T1': 0.5, 'T2': 0.5, 'T3': 0.5, 'T4': 0.5},
         heterogeneous_canopy=True,
         PLANT_DENSITY={1:250.}, update_parameters_all_models=None,
         N_fertilizations={'constant_Conc_Nitrates': 5},
         INPUTS_DIRPATH='inputs',METEO_FILENAME = 'meteo_Ljutovac2002.csv', OUTPUTS_DIRPATH='outputs', POSTPROCESSING_DIRPATH='postprocessing', GRAPHS_DIRPATH='graphs'
         )