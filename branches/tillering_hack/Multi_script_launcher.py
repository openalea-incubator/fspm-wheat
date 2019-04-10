import subprocess
import multiprocessing as mp
import time

def launch_cmd(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

def multi_script_launcher():
    commands = ['conda activate fspmwheat && cd fspmwheat_5U_init && python main.py', 'conda activate fspmwheat && cd fspmwheat_50PAR && python main.py']

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
