# -*- coding: latin-1 -*-

'''
    main
    ~~~~

    An example to show how to couple models CN-Wheat, Farquhar-Wheat and Senesc-Wheat using a static topology from Adel-Wheat.
    This example uses the format MTG to exchange data between the models.

    You must first install :mod:`alinea.adel`, :mod:`cnwheat`, :mod:`farquharwheat` and :mod:`senescwheat` (and add them to your PYTHONPATH)
    before running this script with the command `python`.

    :copyright: Copyright 2014-2016 INRA-ECOSYS, see AUTHORS.
    :license: TODO, see LICENSE for details.

    .. seealso:: Barillot et al. 2016.
'''

'''
    Information about this versioned file:
        $LastChangedBy$
        $LastChangedDate$
        $LastChangedRevision$
        $URL$
        $Id$
'''

import os
import time, datetime
import profile, pstats

import random

import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from alinea.adel.adel_dynamic import AdelWheatDyn
import cnwheat_facade, elongwheat_facade, farquharwheat_facade, growthwheat_facade, senescwheat_facade, caribu_facade

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
METEO_FILEPATH = os.path.join(FARQUHARWHEAT_INPUTS_DIRPATH, 'meteo_Clermont_rebuild.csv')

# senescwheat inputs at t0
SENESCWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'senescwheat')
SENESCWHEAT_ROOTS_INPUTS_FILEPATH = os.path.join(SENESCWHEAT_INPUTS_DIRPATH, 'roots_inputs.csv')
SENESCWHEAT_ELEMENTS_INPUTS_FILEPATH = os.path.join(SENESCWHEAT_INPUTS_DIRPATH, 'elements_inputs.csv')

# elongwheat inputs at t0
ELONGWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'elongwheat')
ELONGWHEAT_HIDDENZONE_INPUTS_FILEPATH = os.path.join(ELONGWHEAT_INPUTS_DIRPATH, 'hiddenzones_inputs.csv')
ELONGWHEAT_ELEMENTS_INPUTS_FILEPATH = os.path.join(ELONGWHEAT_INPUTS_DIRPATH, 'elements_inputs.csv')
ELONGWHEAT_SAM_INPUTS_FILEPATH = os.path.join(ELONGWHEAT_INPUTS_DIRPATH, 'SAM_inputs.csv')

# growthwheat inputs at t0
GROWTHWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'growthwheat')
GROWTHWHEAT_HIDDENZONE_INPUTS_FILEPATH = os.path.join(GROWTHWHEAT_INPUTS_DIRPATH, 'hiddenzones_inputs.csv')
GROWTHWHEAT_ORGANS_INPUTS_FILEPATH = os.path.join(GROWTHWHEAT_INPUTS_DIRPATH, 'organs_inputs.csv')
GROWTHWHEAT_ROOTS_INPUTS_FILEPATH = os.path.join(GROWTHWHEAT_INPUTS_DIRPATH, 'roots_inputs.csv')

# the path of the CSV files where to save the states of the modeled system at each step
OUTPUTS_DIRPATH = 'outputs'
AXES_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'axes_states.csv')
SAM_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'SAM_states.csv')
ORGANS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'organs_states.csv')
HIDDENZONES_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'hiddenzones_states.csv')
ELEMENTS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'elements_states.csv')
SOILS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'soils_states.csv')

AXES_INDEX_COLUMNS = ['t','plant','axis']
ELEMENTS_INDEX_COLUMNS = ['t','plant','axis', 'metamer', 'organ', 'element']
HIDDENZONES_INDEX_COLUMNS = ['t','plant','axis', 'metamer']
ORGANS_INDEX_COLUMNS = ['t','plant','axis', 'organ']
SAM_INDEX_COLUMNS = ['t','plant','axis']
SOILS_INDEX_COLUMNS = ['t','plant','axis']

# Define culm density (culm m-2)
CULM_DENSITY = {1:410}

INPUTS_OUTPUTS_PRECISION = 5 # 10

LOGGING_CONFIG_FILEPATH = os.path.join('..', '..', 'logging.json')

LOGGING_LEVEL = logging.INFO # can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL

def main(stop_time, run_simu=True, make_graphs=True):
    if run_simu:
        meteo = pd.read_csv(METEO_FILEPATH, index_col='t')

        current_time_of_the_system = time.time()

        # define the time step in hours for each simulator
        caribu_ts = 4
        senescwheat_ts = 2
        farquharwheat_ts = 2
        elongwheat_ts = 1
        growthwheat_ts = 1
        cnwheat_ts = 1

        hour_to_second_conversion_factor = 3600

        # read adelwheat inputs at t0
        adel_wheat = AdelWheatDyn(seed=1234, convUnit=1)
        g = adel_wheat.load(dir=ADELWHEAT_INPUTS_DIRPATH)
        adel_wheat.domain = g.get_vertex_property(0)['meta']['domain'] # temp (until Christian's commit)
        adel_wheat.nplants = g.get_vertex_property(0)['meta']['nplants'] # temp (until Christian's commit)

        # create empty dataframes to shared data between the models
        shared_axes_inputs_outputs_df = pd.DataFrame()
        shared_SAM_inputs_outputs_df = pd.DataFrame()
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

        # farquharwheat
        farquharwheat_elements_inputs_t0 = pd.read_csv(FARQUHARWHEAT_INPUTS_FILEPATH)
        farquharwheat_facade_ = farquharwheat_facade.FarquharWheatFacade(g,
                                                                         farquharwheat_elements_inputs_t0,
                                                                         shared_elements_inputs_outputs_df)

        # elongwheat
        elongwheat_hiddenzones_inputs_t0 = pd.read_csv(ELONGWHEAT_HIDDENZONE_INPUTS_FILEPATH)
        elongwheat_element_inputs_t0 = pd.read_csv(ELONGWHEAT_ELEMENTS_INPUTS_FILEPATH)
        elongwheat_SAM_inputs_t0 = pd.read_csv(ELONGWHEAT_SAM_INPUTS_FILEPATH)
        elongwheat_facade_ = elongwheat_facade.ElongWheatFacade(g,
                                                                elongwheat_ts * hour_to_second_conversion_factor,
                                                                elongwheat_SAM_inputs_t0,
                                                                elongwheat_hiddenzones_inputs_t0,
                                                                elongwheat_element_inputs_t0,
                                                                shared_SAM_inputs_outputs_df,
                                                                shared_hiddenzones_inputs_outputs_df,
                                                                shared_elements_inputs_outputs_df,
                                                                adel_wheat)

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

##        # Update geometry
##        adel_wheat.update_geometry(g)#, SI_units=True, properties_to_convert=properties_to_convert) # Returns mtg with non-SI units
####        adel_wheat.convert_to_SI_units(g, properties_to_convert)
##        adel_wheat.plot(g)

        # define the start and the end of the whole simulation (in hours)
        start_time = 0
        stop_time = stop_time

        #�define lists of dataframes to store the inputs and the outputs of the models at each step.
        axes_all_data_list = []
        organs_all_data_list = [] # organs which belong to axes: roots, phloem, grains
        hiddenzones_all_data_list = []
        elements_all_data_list = []
        SAM_all_data_list = []
        soils_all_data_list = []

        all_simulation_steps = [] # to store the steps of the simulation

        print('D�but simulation')

        # run the simulators
        current_time_of_the_system = time.time()

        for t_caribu in xrange(start_time, stop_time, caribu_ts):

          if t_caribu == start_time:

            # run Caribu
            PARi = meteo.loc[t_caribu, ['PARi']].iloc[0]
            caribu_facade_.run(energy=PARi)
            print('t caribu is {}'.format(t_caribu))
            for t_senescwheat in xrange(t_caribu, t_caribu + caribu_ts, senescwheat_ts):
                # run SenescWheat
                print('t senescwheat is {}'.format(t_senescwheat))
                senescwheat_facade_.run()
                for t_farquharwheat in xrange(t_senescwheat, t_senescwheat + senescwheat_ts, farquharwheat_ts):
                    # get the meteo of the current step
                    Ta, ambient_CO2, RH, Ur = meteo.loc[t_farquharwheat, ['air_temperature', 'ambient_CO2', 'humidity', 'Wind']]
                    # run FarquharWheat
                    farquharwheat_facade_.run(Ta, ambient_CO2, RH, Ur)
                    for t_elongwheat in xrange(t_farquharwheat, t_farquharwheat + farquharwheat_ts, elongwheat_ts):
                        print('t elongwheat is {}'.format(t_elongwheat))
                        # run ElongWheat
                        Ta, Tsol = meteo.loc[t_elongwheat, ['air_temperature', 'air_temperature']] # TODO: Add soil temperature in the weather input file
                        elongwheat_facade_.run(Ta, Tsol)
                        # Update geometry
                        adel_wheat.update_geometry(g)#, SI_units=True, properties_to_convert=properties_to_convert) # Return mtg with non-SI units
                        #adel_wheat.plot(g)
##                        adel_wheat.convert_to_SI_units(g, properties_to_convert)

                        for t_growthwheat in xrange(t_elongwheat, t_elongwheat + elongwheat_ts, growthwheat_ts):
                            # run GrowthWheat
                            print('t growthwheat is {}'.format(t_growthwheat))
                            growthwheat_facade_.run()
                            for t_cnwheat in xrange(t_growthwheat, t_growthwheat + growthwheat_ts, cnwheat_ts):
                                # run CNWheat
                                print('t cnwheat is {}'.format(t_cnwheat))
                                cnwheat_facade_.run()

                                # append the inputs and outputs at current step to global lists
                                all_simulation_steps.append(t_cnwheat)
                                axes_all_data_list.append(shared_axes_inputs_outputs_df.copy())
                                organs_all_data_list.append(shared_organs_inputs_outputs_df.copy())
                                hiddenzones_all_data_list.append(shared_hiddenzones_inputs_outputs_df.copy())
                                elements_all_data_list.append(shared_elements_inputs_outputs_df.copy())
                                SAM_all_data_list.append(shared_SAM_inputs_outputs_df.copy())
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

        all_hiddenzones_inputs_outputs = pd.concat(hiddenzones_all_data_list, keys=all_simulation_steps)
        all_hiddenzones_inputs_outputs.reset_index(0, inplace=True)
        all_hiddenzones_inputs_outputs.rename_axis({'level_0': 't'}, axis=1, inplace=True)
        all_hiddenzones_inputs_outputs.to_csv(HIDDENZONES_STATES_FILEPATH, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))

        all_SAM_inputs_outputs = pd.concat(SAM_all_data_list, keys=all_simulation_steps)
        all_SAM_inputs_outputs.reset_index(0, inplace=True)
        all_SAM_inputs_outputs.rename_axis({'level_0': 't'}, axis=1, inplace=True)
        all_SAM_inputs_outputs.to_csv(SAM_STATES_FILEPATH, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))

        all_elements_inputs_outputs = pd.concat(elements_all_data_list, keys=all_simulation_steps)
        all_elements_inputs_outputs = all_elements_inputs_outputs.loc[(all_elements_inputs_outputs.plant == 1) & ### TODO: temporary ; to remove when there will be default input values for each element in the mtg
                                                                      (all_elements_inputs_outputs.axis == 'MS')]
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
                            'R_growth': u'Growth respiration of roots (�mol C h$^{-1}$)', 'mstruct': u'Structural mass (g)',
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

        # 4) Hidden zones
        all_hiddenzones_inputs_outputs_df = pd.read_csv(HIDDENZONES_STATES_FILEPATH)
        graph_variables_hiddenzones = {'leaf_dist_to_emerge': u'Length for leaf emergence (m)','leaf_L': u'Leaf length (m)', 'delta_leaf_L':u'Delta leaf length (m)',
                                       'Conc_Sucrose':u'[Sucrose] (�mol g$^{-1}$ mstruct)', 'Conc_Amino_Acids':u'[Amino Acids] (�mol g$^{-1}$ mstruct)', 'Conc_Proteins': u'[Proteins] (g g$^{-1}$ mstruct)', 'Conc_Fructan':u'[Fructan] (�mol g$^{-1}$ mstruct)',
                                        'Unloading_Sucrose':u'Sucrose unloading (�mol C)', 'Unloading_Amino_Acids':u'Amino_acids unloading (�mol N)', 'mstruct': u'Structural mass (g)', 'Respi_growth': u'Growth respiration (�mol C)', 'sucrose_consumption_mstruct': u'Consumption of sucrose for growth (�mol C)'}

        for variable_name, variable_label in graph_variables_hiddenzones.iteritems():
            graph_name = variable_name + '_hz' + '.PNG'
            tools.plot_cnwheat_ouputs(all_hiddenzones_inputs_outputs_df,
                          x_name = x_name,
                          y_name = variable_name,
                          x_label = x_label,
                          y_label = variable_label,
                          filters={'plant': 1, 'axis': 'MS'},
                          plot_filepath=os.path.join(GRAPHS_DIRPATH, graph_name),
                          explicit_label=False)

        # 5) Organs
        all_organs_inputs_outputs_df = pd.read_csv(ELEMENTS_STATES_FILEPATH)
        graph_variables_organs = {'visible_length': u'Length (m)'}

        for variable_name, variable_label in graph_variables_organs.iteritems():
            graph_name = variable_name + '.PNG'
            tools.plot_cnwheat_ouputs(all_organs_inputs_outputs_df,
                          x_name = x_name,
                          y_name = variable_name,
                          x_label = x_label,
                          y_label = variable_label,
                          filters={'plant': 1, 'axis': 'MS'},
                          plot_filepath=os.path.join(GRAPHS_DIRPATH, graph_name),
                          explicit_label=False)

if __name__ == '__main__':
    main(2, run_simu=True, make_graphs=False)