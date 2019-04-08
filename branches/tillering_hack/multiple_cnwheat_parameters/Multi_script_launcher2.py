import multiprocessing as mp
import time
import main
import pandas as pd

scenarii_cnwheat_parameters = pd.read_csv('scenarii_cnwheat_parameters.csv')
scenarii_cnwheat_parameters = scenarii_cnwheat_parameters.reindex(scenarii_cnwheat_parameters.Scenario)
scenarii = scenarii_cnwheat_parameters.Scenario

if __name__ == '__main__':
    tstart = time.time()
    num_processes = mp.cpu_count()
    p = mp.Pool(num_processes)

    mp_solutions = p.map( main.run_one_scenario, list(scenarii) )
    p.terminate()
    p.join()

    tend = time.time()
    tmp = (tend - tstart) / 60.
    print("multiprocessing: %8.3f minutes" % tmp)

## Tester 3 versions (2 scenarii, 10 pas de temps) :
# - // en appel de fonction : 2.790 minutes
# - // en appel de script : 2.892 minutes
# - en bouble : 3.843 minutes

## 4 senarii - 1900 pas de temps
# - // fonction : 510.760 minutes i.e. 2h / simulations au lieu de 2h30