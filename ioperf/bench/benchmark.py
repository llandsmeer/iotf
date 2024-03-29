import time
import io
import subprocess
import base64
import numpy as np
import matplotlib.pyplot as plt

from ..runners import runners
from .model_configuration import ModelConfiguration

__all__ = ['Benchmark']

EXECUTABLES = (
        'hostname', 'uname -a', 'lspci',
        'lsusb', 'lscpu', 'lsipc', 'nvidia-smi', 'tsp-ctl status'
)

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

class Benchmark:
    def __init__(self, log_file):
        self.base_powers = 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100
        self.seed = 42
        self.run_connected = True
        self.run_unconnected = True
        self.n_ms = 100
        self.n_rep = 5
        self.n_probe = 10000
        self.plot = False
        self.log_file = log_file
        self.spike_probe = True

    def log(self, *args):
        raw = ' '.join(map(str, args))
        print(bcolors.OKGREEN + raw[:100] + bcolors.ENDC)
        with open(self.log_file, 'a') as f:
            print(raw, file=f)

    def _setup(self):
        self.model_configs = [
                ModelConfiguration.create_new(
                    nneurons=bp**3, seed=self.seed)
                for bp in self.base_powers]
        bp = ','.join(map(str, self.base_powers))
        self.log(f'bench b={bp} seed={self.seed} n_ms={self.n_ms} n_rep={self.n_rep} n_probe={self.n_probe} unconnected={self.run_unconnected} connected={self.run_connected}')

    def register(self, key, a, b, config, runner, arg=None):
        if isinstance(arg, np.ndarray):
            arg_p = f'{arg.dtype}{arg.shape}'
            buf = io.BytesIO()
            np.savez_compressed(buf, arg=arg)
            arg = base64.b64encode(buf.getvalue()).decode('latin1')
        else:
            arg_p = arg
            arg = str(arg)
        self.log('row', type(runner).__name__, config.ncells, config.ngj, key, b - a, arg)

    def write_hwinfo(self):
        for e in EXECUTABLES:
            ee = e.replace(' ', '_')
            res = subprocess.getoutput(e)
            for line in res.splitlines():
                self.log('hwinfo', ee, line)


    def run(self):
        self._setup()
        supported_runners = []
        for Runner in runners:
            x = Runner()
            if x.is_supported():
                supported_runners.append(x)
                self.log(f'runner supported {type(x).__name__} {x.tagline()}')
            else:
                self.log(f'runner unsupported {type(x).__name__}')
        self.write_hwinfo()

        if self.spike_probe:
            for runner in supported_runners:
                if(runner == "GroqchipRunnerOpt2NoCopy"):
                    continue
                a = time.perf_counter()
                config = self.model_configs[0]
                runner.setup(
                        ncells=config.ncells,
                        ngj=config.ngj,
                        argconfig=dict(
                            I_app='VARY',
                            g_CaL='VARY',
                            )
                        )
                state = config.state
                np.random.seed(self.seed // 1)
                I_app = np.zeros(config.ncells, dtype='float32')
                g_CaL = np.array(0.5+0.9*np.random.random(config.ncells), dtype='float32')
                traces = []
                # 5 seconds of transients to cool down the sim
                state, trace = runner.run_with_gap_junctions(
                        5000,
                        state,
                        gj_src=config.gj_src,
                        gj_tgt=config.gj_tgt,
                        g_gj=0.05,
                        I_app=I_app,
                        g_CaL=g_CaL,
                        probe=True,
                        )
                traces.append(trace)
                for _ in range(10):
                    # apply a short 50ms pulse current every second
                    idx = np.random.choice(config.ncells, len(I_app)//3)
                    I_app[idx] = np.random.normal(10, 3, len(idx))
                    state, trace = runner.run_with_gap_junctions(
                            50,
                            state,
                            gj_src=config.gj_src,
                            gj_tgt=config.gj_tgt,
                            g_gj=0.05,
                            I_app=I_app,
                            g_CaL=g_CaL,
                            probe=True,
                            )
                    I_app *= 0
                    traces.append(trace[1:])
                    state, trace = runner.run_with_gap_junctions(
                            1000-50,
                            state,
                            gj_src=config.gj_src,
                            gj_tgt=config.gj_tgt,
                            g_gj=0.05,
                            I_app=I_app,
                            g_CaL=g_CaL,
                            probe=True,
                            )
                    traces.append(trace[1:])
                trace = np.vstack(traces)
                b = time.perf_counter()
                self.register('run_spikes_probe', a, b, config, runner, trace)
                if self.plot:
                    plt.title(f'run spikes {type(runner).__name__}')
                    plt.plot(trace)
                    plt.show()

        for config in self.model_configs:
            for runner in supported_runners:
                runner.setup_using_model_config(config, gap_junctions=True)
                try:
                    if self.run_unconnected:
                        a = time.perf_counter()
                        runner.setup_using_model_config(config, gap_junctions=False)
                        b = time.perf_counter()
                        self.register('setup_unconnected', a, b, config, runner)
                        for i in range(self.n_rep):
                            a = time.perf_counter()
                            runner.run_unconnected(self.n_ms, config.state)
                            b = time.perf_counter()
                            self.register('run_unconnected_perf', a, b, config, runner, i)
                        if config.ncells < 1000:
                            a = time.perf_counter()
                            _, trace = runner.run_unconnected(self.n_probe, config.state, probe=True)
                            b = time.perf_counter()
                            self.register('run_unconnected_probe', a, b, config, runner, trace)
                            if self.plot:
                                plt.title(f'run unconnected {type(runner).__name__}')
                                plt.plot(trace)
                                plt.show()
                except KeyboardInterrupt:
                    raise
                except Exception as ex:
                    self.log(f"An occurred {type(x).__name__} ")
                    print(repr(ex))
                try:
                    if self.run_connected:
                        a = time.perf_counter()
                        runner.setup_using_model_config(config, gap_junctions=True)
                        b = time.perf_counter()
                        self.register('setup_connected', a, b, config, runner)
                        for i in range(self.n_rep):
                            a = time.perf_counter()
                            runner.run_with_gap_junctions(self.n_ms, config.state, gj_src=config.gj_src, gj_tgt=config.gj_tgt)
                            b = time.perf_counter()
                            self.register('run_connected_perf', a, b, config, runner, i)
                        if config.ncells < 1000:
                            a = time.perf_counter()
                            _, trace = runner.run_with_gap_junctions(self.n_probe, config.state, gj_src=config.gj_src, gj_tgt=config.gj_tgt, probe=True)
                            b = time.perf_counter()
                            self.register('run_connected_probe', a, b, config, runner, trace)
                            if self.plot:
                                plt.title(f'run connected {type(runner).__name__}')
                                plt.plot(trace)
                                plt.show()
                except KeyboardInterrupt:
                    raise
                except Exception as ex:
                    self.log(f"An exception occurred {type(x).__name__} ")
                    print(repr(ex))
        with open(self.log_file, 'a') as f:
            print('done', file=f)
