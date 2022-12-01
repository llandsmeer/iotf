import multiprocessing as mp
import time
import subprocess
import os

from .. import runners
from .model_configuration import ModelConfiguration

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def measurement_loop(runner, exe, log_file):
    a = time.perf_counter()
    i = 0
    while True:
        print(i)
        b = time.perf_counter()
        res = subprocess.getoutput(exe)
        with open(log_file, 'a') as f:
            for line in res.splitlines():
                print(runner, i, f'{b - a:10.5f}', line, file=f)
        i += 1
        time.sleep(1)


class EnergyBenchmark:
    def __init__(self, log_file):
        self.config = ModelConfiguration.create_new(nneurons=9**3, seed=1234)
        self.runners = [
            (runners.GraphcoreRunner,    'gc-monitor'),
            (runners.GroqchipRunner,     'tsp-ctl monitor'),
            (runners.tf_base_runner,     'nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --id=0'),
            # (runners.TfBaseRunner,       'cat /sys/class/power_supply/BAT0/uevent'), (LAPTOP)
            (runners.OnnxCUDARunner,     'nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --id=0'),
            (runners.OnnxTensorRTRunner, 'nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --id=0'),
            (runners.OnnxTensorRTRunner, 'nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --id=0'),
        ]
        self.log_file = log_file
        self.hostname = subprocess.getoutput('hostname').strip()

    def log(self, *args):
        raw = ' '.join(map(str, args))
        print(bcolors.OKGREEN + raw[:100] + bcolors.ENDC)
        with open(self.log_file, 'a') as f:
            print(raw, file=f)

    def run(self):
        config = self.config
        for Runner, exe in self.runners:
            runner = Runner()
            if not runner.is_supported():
                self.log('unsupported', type(runner).__name__)
                continue
            if os.system(f'which {exe.split()[0]}') != 0:
                self.log('unsupported', type(runner).__name__, 'not-found:', exe)
            self.log('energy', config)
            runner.setup_using_model_config(config, gap_junctions=True)
            # burn jit:
            runner.run_with_gap_junctions(1000, config.state, gj_src=config.gj_src, gj_tgt=config.gj_tgt)
            t = mp.Process(target=measurement_loop, args=(type(runner).__name__, exe, self.log_file))
            t.start()
            runner.run_with_gap_junctions(10000, config.state, gj_src=config.gj_src, gj_tgt=config.gj_tgt)
            t.terminate()
            t.join()

