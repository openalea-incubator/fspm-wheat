import subprocess
import multiprocessing as mp
import time
import pandas as pd

import main

scenarii_cnwheat_parameters = pd.read_csv('scenarii_cnwheat_parameters.csv')
scenarii_cnwheat_parameters = scenarii_cnwheat_parameters.reindex(scenarii_cnwheat_parameters.Scenario)
scenarii = scenarii_cnwheat_parameters.Scenario

def launch_cmd(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

def multi_script_launcher():
    commands = []
    for scenario in scenarii:
        commands.append('conda activate fspmwheat && python main.py {}'.format(scenario)) #'conda activate fspmwheat && main.py {}'

    tstart = time.time()
    num_processes = mp.cpu_count()
    p = mp.Pool(num_processes)

    mp_solutions = p.map(launch_cmd, commands)
    p.close()
    p.join()

    tend = time.time()
    tmp = (tend - tstart) / 60.
    print("multiprocessing: %8.3f minutes" % tmp)

if __name__ == '__main__':
    multi_script_launcher()
