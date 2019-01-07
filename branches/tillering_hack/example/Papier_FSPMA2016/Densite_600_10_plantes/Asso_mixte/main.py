# -*- coding: latin-1 -*-

'''
    main
    ~~~~

    This example uses the format MTG to exchange data between the models.

    You must first install :mod:`alinea.adel`, :mod:`cnwheat`, :mod:`farquharwheat` and :mod:`senescwheat` (and add them to your PYTHONPATH)
    before running this script with the command `python`.

    :copyright: Copyright 2014-2016 INRA-ECOSYS, see AUTHORS.
    :license: TODO, see LICENSE for details.

    .. seealso:: Barillot et al. 2016.
'''

'''
    Information about this versioned file:
        $LastChangedBy: rbarillot $
        $LastChangedDate: 2017-01-09 15:45:53 +0100 (lun., 09 janv. 2017) $
        $LastChangedRevision: 9 $
        $URL: https://subversion.renater.fr/fspm-wheat/trunk/example/NEMA_H3/main.py $
        $Id: main.py 9 2017-01-09 14:45:53Z rbarillot $
'''

import os
import time, datetime
import profile, pstats

import random

import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from alinea.adel.astk_interface import AdelWheat
from fspmwheat import cnwheat_facade, farquharwheat_facade, senescwheat_facade, growthwheat_facade, caribu_facade

random.seed(1234)
np.random.seed(1234)

INPUTS_DIRPATH = 'inputs'
GRAPHS_DIRPATH = 'graphs'

# adelwheat inputs at t0
ADELWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'adelwheat') #�the directory adelwheat must contain files 'adel0000.pckl' and 'scene0000.bgeom'

# cnwheat inputs at t0
CNWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'cnwheat')
CNWHEAT_PLANTS_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'plants_inputs.csv')
CNWHEAT_AXES_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'axes_inputs.csv')
CNWHEAT_METAMERS_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'metamers_inputs.csv')
CNWHEAT_ORGANS_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'organs_inputs.csv')
CNWHEAT_HIDDENZONE_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'hiddenzones_inputs.csv')
CNWHEAT_ELEMENTS_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'elements_inputs.csv')
CNWHEAT_SOILS_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'soils_inputs.csv')

# farquharwheat inputs at t0
FARQUHARWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'farquharwheat')
FARQUHARWHEAT_INPUTS_FILEPATH = os.path.join(FARQUHARWHEAT_INPUTS_DIRPATH, 'inputs.csv')
METEO_FILEPATH = os.path.join(FARQUHARWHEAT_INPUTS_DIRPATH, 'meteo_Lusignan_2009.csv')
CARIBU_FILEPATH = os.path.join(FARQUHARWHEAT_INPUTS_DIRPATH, 'inputs_eabs.csv')


# senescwheat inputs at t0
SENESCWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'senescwheat')
SENESCWHEAT_ROOTS_INPUTS_FILEPATH = os.path.join(SENESCWHEAT_INPUTS_DIRPATH, 'roots_inputs.csv')
SENESCWHEAT_ELEMENTS_INPUTS_FILEPATH = os.path.join(SENESCWHEAT_INPUTS_DIRPATH, 'elements_inputs.csv')

# growthwheat inputs at t0
GROWTHWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'growthwheat')
GROWTHWHEAT_HIDDENZONE_INPUTS_FILEPATH = os.path.join(GROWTHWHEAT_INPUTS_DIRPATH, 'hiddenzones_inputs.csv')
GROWTHWHEAT_ORGANS_INPUTS_FILEPATH = os.path.join(GROWTHWHEAT_INPUTS_DIRPATH, 'organs_inputs.csv')
GROWTHWHEAT_ROOTS_INPUTS_FILEPATH = os.path.join(GROWTHWHEAT_INPUTS_DIRPATH, 'roots_inputs.csv')

# the path of the CSV files where to save the states of the modeled system at each step
OUTPUTS_DIRPATH = 'outputs'
AXES_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'axes_states.csv')
ORGANS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'organs_states.csv')
ELEMENTS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'elements_states.csv')
SOILS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'soils_states.csv')

AXES_INDEX_COLUMNS = ['t','plant','axis']
ELEMENTS_INDEX_COLUMNS = ['t','plant','axis', 'metamer', 'organ', 'element']
ORGANS_INDEX_COLUMNS = ['t','plant','axis', 'organ']
SOILS_INDEX_COLUMNS = ['t','plant','axis']

# Define culm density (culm m-2)
DENSITY = 600.
NPLANTS = 10
CULM_DENSITY = {i: DENSITY / NPLANTS for i in range(1, NPLANTS + 1)}

INPUTS_OUTPUTS_PRECISION = 5 # 10

LOGGING_CONFIG_FILEPATH = os.path.join('..', '..', 'logging.json')

LOGGING_LEVEL = logging.INFO # can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL

def calculate_PARa_from_df(g, Eabs_df, PARi, multiple_sources=False, ratio_diffus_PAR=None):
    """
    Compute PARa from an input dataframe having Eabs values.
    """


    Eabs_df_grouped = Eabs_df.groupby(['plant', 'metamer', 'organ'])

    #: the name of the elements modeled by FarquharWheat
    CARIBU_ELEMENTS_NAMES = set(['StemElement', 'LeafElement1'])

    PARa_element_data_dict = {}
    # traverse the MTG recursively from top ...
    for mtg_plant_vid in g.components_iter(g.root):
        mtg_plant_index = int(g.index(mtg_plant_vid))
        for mtg_axis_vid in g.components_iter(mtg_plant_vid):
            for mtg_metamer_vid in g.components_iter(mtg_axis_vid):
                mtg_metamer_index = int(g.index(mtg_metamer_vid))
                for mtg_organ_vid in g.components_iter(mtg_metamer_vid):
                    mtg_organ_label = g.label(mtg_organ_vid)
                    for mtg_element_vid in g.components_iter(mtg_organ_vid):
                        mtg_element_label = g.label(mtg_element_vid)
                        if mtg_element_label not in CARIBU_ELEMENTS_NAMES: continue
                        element_id = (mtg_plant_index, mtg_metamer_index, mtg_organ_label)
                        if element_id in Eabs_df_grouped.groups.keys():
                            if PARi == 0:
                                PARa_element_data_dict[mtg_element_vid] = 0
                            elif multiple_sources:
                                PARa_diffuse = Eabs_df_grouped.get_group(element_id)['Eabs_diffuse'].iloc[0] * PARi * ratio_diffus_PAR
                                PARa_direct = Eabs_df_grouped.get_group(element_id)['Eabs_direct'].iloc[0] * PARi * (1 - ratio_diffus_PAR)
                                PARa_element_data_dict[mtg_element_vid] = PARa_diffuse + PARa_direct
                            else:
                                PARa_element_data_dict[mtg_element_vid] = Eabs_df_grouped.get_group(element_id)['Eabs'].iloc[0] * PARi

    return PARa_element_data_dict

def main(stop_time, run_simu=True, make_graphs=True):
    if run_simu:
        meteo = pd.read_csv(METEO_FILEPATH, index_col='t')
        Eabs_df = pd.read_csv(CARIBU_FILEPATH)

        current_time_of_the_system = time.time()

        # define the time step in hours for each simulator
        senescwheat_ts = 2
        growthwheat_ts = 2
        farquharwheat_ts = 2
        cnwheat_ts = 1

        hour_to_second_conversion_factor = 3600

        # read adelwheat inputs at t0
        adel_wheat = AdelWheat(seed=1234, convUnit=1)
        g = adel_wheat.load(dir=ADELWHEAT_INPUTS_DIRPATH)

        # create empty dataframes to shared data between the models
        shared_axes_inputs_outputs_df = pd.DataFrame()
        shared_organs_inputs_outputs_df = pd.DataFrame()
        shared_hiddenzones_inputs_outputs_df = pd.DataFrame()
        shared_elements_inputs_outputs_df = pd.DataFrame()
        shared_soils_inputs_outputs_df = pd.DataFrame()

        # read the inputs at t0 and create the facades

        # caribu
        caribu_facade_ = caribu_facade.CaribuFacade(g,
                                                    shared_elements_inputs_outputs_df,
                                                    adel_wheat)

        # senescwheat
        senescwheat_roots_inputs_t0 = pd.read_csv(SENESCWHEAT_ROOTS_INPUTS_FILEPATH)
        senescwheat_elements_inputs_t0 = pd.read_csv(SENESCWHEAT_ELEMENTS_INPUTS_FILEPATH)
        senescwheat_facade_ = senescwheat_facade.SenescWheatFacade(g,
                                                                   senescwheat_ts * hour_to_second_conversion_factor,
                                                                   senescwheat_roots_inputs_t0,
                                                                   senescwheat_elements_inputs_t0,
                                                                   shared_organs_inputs_outputs_df,
                                                                   shared_elements_inputs_outputs_df)
        # growthwheat
        growthwheat_hiddenzones_inputs_t0 = pd.read_csv(GROWTHWHEAT_HIDDENZONE_INPUTS_FILEPATH)
        growthwheat_organ_inputs_t0 = pd.read_csv(GROWTHWHEAT_ORGANS_INPUTS_FILEPATH)
        growthwheat_root_inputs_t0 = pd.read_csv(GROWTHWHEAT_ROOTS_INPUTS_FILEPATH)
        growthwheat_facade_ = growthwheat_facade.GrowthWheatFacade(g,
                                                                   growthwheat_ts * hour_to_second_conversion_factor,
                                                                   growthwheat_hiddenzones_inputs_t0,
                                                                   growthwheat_organ_inputs_t0,
                                                                   growthwheat_root_inputs_t0,
                                                                   shared_organs_inputs_outputs_df,
                                                                   shared_hiddenzones_inputs_outputs_df,
                                                                   shared_elements_inputs_outputs_df)

        # farquharwheat
        farquharwheat_elements_inputs_t0 = pd.read_csv(FARQUHARWHEAT_INPUTS_FILEPATH)
        farquharwheat_facade_ = farquharwheat_facade.FarquharWheatFacade(g,
                                                                         farquharwheat_elements_inputs_t0,
                                                                         shared_elements_inputs_outputs_df)


        # cnwheat
        cnwheat_organs_inputs_t0 = pd.read_csv(CNWHEAT_ORGANS_INPUTS_FILEPATH)
        cnwheat_hiddenzones_inputs_t0 = pd.read_csv(CNWHEAT_HIDDENZONE_INPUTS_FILEPATH)
        cnwheat_elements_inputs_t0 = pd.read_csv(CNWHEAT_ELEMENTS_INPUTS_FILEPATH)
        cnwheat_soils_inputs_t0 = pd.read_csv(CNWHEAT_SOILS_INPUTS_FILEPATH)
        cnwheat_facade_ = cnwheat_facade.CNWheatFacade(g,
                                                       cnwheat_ts * hour_to_second_conversion_factor,
                                                       CULM_DENSITY,
                                                       cnwheat_organs_inputs_t0,
                                                       cnwheat_hiddenzones_inputs_t0,
                                                       cnwheat_elements_inputs_t0,
                                                       cnwheat_soils_inputs_t0,
                                                       shared_axes_inputs_outputs_df,
                                                       shared_organs_inputs_outputs_df,
                                                       shared_hiddenzones_inputs_outputs_df,
                                                       shared_elements_inputs_outputs_df,
                                                       shared_soils_inputs_outputs_df)


        # define organs for which the variable 'max_proteins' is fixed
        nplants = 10
        forced_max_protein_elements = set([(plant + 1,'MS',metamer + 1,'blade', 'LeafElement1') for metamer in range(0, 3) for plant in range (0, nplants)])

        # define the start and the end of the whole simulation (in hours)
        start_time = 0
        stop_time = stop_time

        #�define lists of dataframes to store the inputs and the outputs of the models at each step.
        axes_all_data_list = []
        organs_all_data_list = [] # organs which belong to axes: roots, phloem, grains
        elements_all_data_list = []
        soils_all_data_list = []

        all_simulation_steps = [] # to store the steps of the simulation

        # run the simulators
        current_time_of_the_system = time.time()

        for t_senescwheat in xrange(start_time, stop_time, senescwheat_ts):
            # run SenescWheat
            senescwheat_facade_.run(forced_max_protein_elements)
            for t_growthwheat in xrange(t_senescwheat, t_senescwheat + senescwheat_ts, growthwheat_ts):
                # run GrowthWheat
                growthwheat_facade_.run()
                for t_farquharwheat in xrange(t_growthwheat, t_growthwheat + growthwheat_ts, farquharwheat_ts):
                    # get the meteo of the current step
                    Ta, ambient_CO2, RH, Ur, PARi, ratio_diffus_PAR = meteo.loc[t_farquharwheat, ['air_temperature', 'ambient_CO2', 'humidity', 'Wind', 'PARi', 'ratio_diffus_PAR']]
                    # get PARa for current step
                    aggregated_PARa = calculate_PARa_from_df(g, Eabs_df[Eabs_df['t']==t_farquharwheat], PARi, multiple_sources=True, ratio_diffus_PAR=ratio_diffus_PAR)
                    caribu_facade_.update_shared_MTG(aggregated_PARa)
                    caribu_facade_.update_shared_dataframes(aggregated_PARa)
                    # run FarquharWheat
                    farquharwheat_facade_.run(Ta, ambient_CO2, RH, Ur)
                    for t_cnwheat in xrange(t_farquharwheat, t_farquharwheat + senescwheat_ts, cnwheat_ts):
                            # run CNWheat
                            cnwheat_facade_.run()

                            # append the inputs and outputs at current step to global lists
                            all_simulation_steps.append(t_cnwheat)
                            axes_all_data_list.append(shared_axes_inputs_outputs_df.copy())
                            organs_all_data_list.append(shared_organs_inputs_outputs_df.copy())
                            elements_all_data_list.append(shared_elements_inputs_outputs_df.copy())
                            soils_all_data_list.append(shared_soils_inputs_outputs_df.copy())

        execution_time = int(time.time() - current_time_of_the_system)
        print '\n', 'Simulation run in ', str(datetime.timedelta(seconds=execution_time))

        # write all inputs and outputs to CSV files
        all_axes_inputs_outputs = pd.concat(axes_all_data_list, keys=all_simulation_steps)
        all_axes_inputs_outputs.reset_index(0, inplace=True)
        all_axes_inputs_outputs.rename_axis({'level_0': 't'}, axis=1, inplace=True)
        all_axes_inputs_outputs.to_csv(AXES_STATES_FILEPATH, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))

        all_organs_inputs_outputs = pd.concat(organs_all_data_list, keys=all_simulation_steps)
        all_organs_inputs_outputs.reset_index(0, inplace=True)
        all_organs_inputs_outputs.rename_axis({'level_0': 't'}, axis=1, inplace=True)
        all_organs_inputs_outputs.to_csv(ORGANS_STATES_FILEPATH, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))

        all_elements_inputs_outputs = pd.concat(elements_all_data_list, keys=all_simulation_steps)
        all_elements_inputs_outputs.reset_index(0, inplace=True)
        all_elements_inputs_outputs.rename_axis({'level_0': 't'}, axis=1, inplace=True)
        all_elements_inputs_outputs.to_csv(ELEMENTS_STATES_FILEPATH, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))

        all_soils_inputs_outputs = pd.concat(soils_all_data_list, keys=all_simulation_steps)
        all_soils_inputs_outputs.reset_index(0, inplace=True)
        all_soils_inputs_outputs.rename_axis({'level_0': 't'}, axis=1, inplace=True)
        all_soils_inputs_outputs.to_csv(SOILS_STATES_FILEPATH, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))

    ########POST-PROCESSING##
    if make_graphs:
        from cnwheat import parameters
        from cnwheat import tools
        x_name = 't'
        x_label='Time (Hour)'

        # 1) Photosynthetic organs
        ph_elements_output_df = pd.read_csv(ELEMENTS_STATES_FILEPATH)

        graph_variables_ph_elements = {'PARa': u'Absorbed PAR (�mol m$^{-2}$ s$^{-1}$)', 'Ag': u'Gross photosynthesis (�mol m$^{-2}$ s$^{-1}$)','An': u'Net photosynthesis (�mol m$^{-2}$ s$^{-1}$)', 'Tr':u'Organ surfacic transpiration rate (mmol H$_{2}$0 m$^{-2}$ s$^{-1}$)', 'Transpiration':u'Organ transpiration rate (mmol H$_{2}$0 s$^{-1}$)', 'Rd': u'Mitochondrial respiration rate of organ in light (�mol C h$^{-1}$)', 'Ts': u'Temperature surface (�C)', 'gs': u'Conductance stomatique (mol m$^{-2}$ s$^{-1}$)',
                           'Conc_TriosesP': u'[TriosesP] (�mol g$^{-1}$ mstruct)', 'Conc_Starch':u'[Starch] (�mol g$^{-1}$ mstruct)', 'Conc_Sucrose':u'[Sucrose] (�mol g$^{-1}$ mstruct)', 'Conc_Fructan':u'[Fructan] (�mol g$^{-1}$ mstruct)',
                           'Conc_Nitrates': u'[Nitrates] (�mol g$^{-1}$ mstruct)', 'Conc_Amino_Acids': u'[Amino_Acids] (�mol g$^{-1}$ mstruct)', 'Conc_Proteins': u'[Proteins] (g g$^{-1}$ mstruct)',
                           'Nitrates_import': u'Total nitrates imported (�mol h$^{-1}$)', 'Amino_Acids_import': u'Total amino acids imported (�mol N h$^{-1}$)',
                           'S_Amino_Acids': u'[Rate of amino acids synthesis] (�mol N g$^{-1}$ mstruct h$^{-1}$)', 'S_Proteins': u'Rate of protein synthesis (�mol N g$^{-1}$ mstruct h$^{-1}$)', 'D_Proteins': u'Rate of protein degradation (�mol N g$^{-1}$ mstruct h$^{-1}$)', 'k_proteins': u'Relative rate of protein degradation (s$^{-1}$)',
                           'Loading_Sucrose': u'Loading Sucrose (�mol C sucrose h$^{-1}$)', 'Loading_Amino_Acids': u'Loading Amino acids (�mol N amino acids h$^{-1}$)',
                           'green_area': u'Green area (m$^{2}$)', 'R_phloem_loading': u'Respiration phloem loading (�mol C h$^{-1}$)', 'R_Nnit_red': u'Respiration nitrate reduction (�mol C h$^{-1}$)', 'R_residual': u'Respiration residual (�mol C h$^{-1}$)', 'R_maintenance': u'Respiration residual (�mol C h$^{-1}$)',
                           'mstruct': u'Structural mass (g)', 'Nstruct': u'Structural N mass (g)',
                           'Conc_cytokinins':u'[cytokinins] (UA g$^{-1}$ mstruct)', 'D_cytokinins':u'Cytokinin degradation (UA g$^{-1}$ mstruct)', 'cytokinins_import':u'Cytokinin import (UA)'}


        for org_ph in (['blade'], ['sheath'], ['internode'], ['peduncle', 'ear']):
            for variable_name, variable_label in graph_variables_ph_elements.iteritems():
                graph_name = variable_name + '_' + '_'.join(org_ph) + '.PNG'
                tools.plot_cnwheat_ouputs(ph_elements_output_df,
                              x_name = x_name,
                              y_name = variable_name,
                              x_label=x_label,
                              y_label=variable_label,
                              filters={'organ': org_ph},
                              plot_filepath=os.path.join(GRAPHS_DIRPATH, graph_name),
                              explicit_label=False)

        # 2) Roots, grains and phloem
        organs_output_df = pd.read_csv(ORGANS_STATES_FILEPATH)

        graph_variables_organs = {'Conc_Sucrose':u'[Sucrose] (�mol g$^{-1}$ mstruct)', 'Dry_Mass':'Dry mass (g)',
                            'Conc_Nitrates': u'[Nitrates] (�mol g$^{-1}$ mstruct)', 'Conc_Amino_Acids':u'[Amino Acids] (�mol g$^{-1}$ mstruct)', 'Proteins_N_Mass': u'[N Proteins] (g)',
                            'Uptake_Nitrates':u'Nitrates uptake (�mol h$^{-1}$)', 'Unloading_Sucrose':u'Unloaded sucrose (�mol C g$^{-1}$ mstruct h$^{-1}$)', 'Unloading_Amino_Acids':u'Unloaded Amino Acids (�mol N AA g$^{-1}$ mstruct h$^{-1}$)',
                            'S_Amino_Acids': u'Rate of amino acids synthesis (�mol N g$^{-1}$ mstruct h$^{-1}$)', 'S_Proteins': u'Rate of protein synthesis (�mol N h$^{-1}$)', 'Export_Nitrates': u'Total export of nitrates (�mol N h$^{-1}$)', 'Export_Amino_Acids': u'Total export of Amino acids (�mol N h$^{-1}$)',
                            'R_Nnit_upt': u'Respiration nitrates uptake (�mol C h$^{-1}$)', 'R_Nnit_red': u'Respiration nitrate reduction (�mol C h$^{-1}$)', 'R_residual': u'Respiration residual (�mol C h$^{-1}$)', 'R_maintenance': u'Respiration residual (�mol C h$^{-1}$)',
                            'R_grain_growth_struct': u'Respiration grain structural growth (�mol C h$^{-1}$)', 'R_grain_growth_starch': u'Respiration grain starch growth (�mol C h$^{-1}$)',
                            'R_growth': u'Growth respiration of roots (�mol C h$^{-1}$)', 'mstruct': u'Structural mass (g)', 'rate_mstruct_death': u'Rate of structural mass death (g)',
                            'C_exudation': u'Carbon lost by root exudation (�mol C g$^{-1}$ mstruct h$^{-1}$', 'N_exudation': u'Nitrogen lost by root exudation (�mol N g$^{-1}$ mstruct h$^{-1}$',
                            'Conc_cytokinins':u'[cytokinins] (UA g$^{-1}$ mstruct)', 'S_cytokinins':u'Rate of cytokinins synthesis (UA g$^{-1}$ mstruct)', 'Export_cytokinins': 'Export of cytokinins from roots (UA h$^{-1}$)',
                            'HATS_LATS': u'Potential uptake (�mol h$^{-1}$)' , 'regul_transpiration':'Regulating transpiration function'}

        for org in (['roots'], ['grains'], ['phloem']):
            for variable_name, variable_label in graph_variables_organs.iteritems():
                graph_name = variable_name + '_' + '_'.join(org) + '.PNG'
                tools.plot_cnwheat_ouputs(organs_output_df,
                              x_name = x_name,
                              y_name = variable_name,
                              x_label=x_label,
                              y_label=variable_label,
                              filters={'organ': org},
                              plot_filepath=os.path.join(GRAPHS_DIRPATH, graph_name),
                              explicit_label=False)

        # 3) Soil
        soil_output_df = pd.read_csv(SOILS_STATES_FILEPATH)

        fig, (ax1) = plt.subplots(1)
        conc_nitrates_soil = soil_output_df['Conc_Nitrates_Soil']*14E-6
        ax1.plot(soil_output_df['t'], conc_nitrates_soil)
        ax1.set_ylabel(u'[Nitrates] (g m$^{-3}$)')
        ax1.set_xlabel('Time from flowering (hour)')
        ax1.set_title = 'Conc Nitrates Soil'
        plt.savefig(os.path.join(GRAPHS_DIRPATH, 'Conc_Nitrates_Soil.PNG'), format='PNG', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main(1200, run_simu=True, make_graphs=False)