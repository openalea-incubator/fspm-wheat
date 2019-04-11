# -*- coding: latin-1 -*-

import sys
import os
import inspect
import pandas as pd
import fspmwheat.main

my_dir = os.getcwd()

# Manual cnwheat parameters
scenarii_cnwheat_parameters = pd.read_csv('scenarii_cnwheat_parameters.csv', index_col='Scenario')
scenarii_cnwheat_parameters['Scenario'] = scenarii_cnwheat_parameters.index
scenarii = scenarii_cnwheat_parameters.Scenario

fspm = os.path.join(inspect.getfile(fspmwheat),'..')

def clear_directory(directory_path):
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def run_one_scenario(param):
    if type(param) != list:
        scenario = param
    elif len(param) > 1:
        scenario = int(param[1])
    else:
        sys.exit('Error : No Scenario value as input')

    scenario_name = 'Scenario_{}'.format(scenario)

    # Create the directory of the Scenario where results will be stored
    scenario_dirpath = os.path.join(my_dir, scenario_name)
    if not os.path.exists(scenario_dirpath):
        os.mkdir(scenario_dirpath)

    # Get cnwheat manual parameters for the scenario
    scenario_cnwheat_parameters = scenarii_cnwheat_parameters.to_dict('index')[scenario]

    # Delete outputs, postprocessing and graphs in FSPM directory so we are sure not to store charts from previous scenario
    directories_to_clean = ['postprocessing','outputs','graphs']
    for directory in directories_to_clean:
        directory_path = os.path.join(inspect.getfile(fspmwheat),'..',directory)
        clear_directory(directory_path)

    # Create directory paths for graphs and outputs of this scneario
    scenario_graphs_dirpath = os.path.join(scenario_dirpath,'graphs')
    if os.path.exists(scenario_graphs_dirpath):
        clear_directory(scenario_graphs_dirpath)
    else :
        os.mkdir(scenario_graphs_dirpath)
    scenario_outputs_dirpath = os.path.join(scenario_dirpath,'outputs')
    if os.path.exists(scenario_outputs_dirpath):
        clear_directory(scenario_outputs_dirpath)
    else :
        os.mkdir(scenario_outputs_dirpath)
    scenario_postprocessing_dirpath = os.path.join(scenario_dirpath, 'postprocessing')
    if os.path.exists(scenario_postprocessing_dirpath):
        clear_directory(scenario_postprocessing_dirpath)
    else:
        os.mkdir(scenario_postprocessing_dirpath)

    # Directory path of the inputs
    case_inputs_dirpath = os.path.join(my_dir,'inputs')

    # Run main fspmwheat
    os.chdir(fspm)
    try:
        fspmwheat.main.main(1700, forced_start_time=0, run_simu=True, run_postprocessing=True, generate_graphs=True, run_from_outputs=False, opt_croiss_fix=False,
                            tillers_replications = {'T1':0.5, 'T2':0.5, 'T3':0.5, 'T4':0.5},
                            manual_cyto_init = 200, heterogeneous_canopy = True, N_fertilizations = {2016:357143, 2520:1000000},
                            cnwheat_parameters = scenario_cnwheat_parameters,
                            elongwheat_parameters = scenario_cnwheat_parameters,
                            GRAPHS_DIRPATH = scenario_graphs_dirpath,
                            OUTPUTS_DIRPATH = scenario_outputs_dirpath,
                            POSTPROCESSING_DIRPATH=scenario_postprocessing_dirpath,
                            INPUTS_DIRPATH=case_inputs_dirpath)
    except Exception as e:
        print e
        pass

    os.chdir(my_dir)

if __name__ == '__main__':
    run_one_scenario(sys.argv)