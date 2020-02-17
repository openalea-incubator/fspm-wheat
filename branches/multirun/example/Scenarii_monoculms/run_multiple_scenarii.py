import multiprocessing as mp
import time
import pandas as pd
import os

import run_one_scenario
import rearrange_graphs

scenarii_df = pd.read_csv( os.path.join('inputs','scenarii_list.csv'), index_col='Scenario')
scenarii_df['Scenario'] = scenarii_df.index
scenarii = scenarii_df.Scenario

if __name__ == '__main__':
    tstart = time.time()
    num_processes = mp.cpu_count()
    p = mp.Pool(num_processes)

    mp_solutions = p.map( run_one_scenario.run_one_scenario, list(scenarii) )
    p.terminate()
    p.join()

    rearrange_graphs.rearrange_graphs( scenarii = list(scenarii) )

    tend = time.time()
    tmp = (tend - tstart) / 60.
    print("multiprocessing: %8.3f minutes" % tmp)
