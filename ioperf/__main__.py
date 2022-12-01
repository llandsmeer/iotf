import os
import sys
import socket
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
import tempfile
import socket
import subprocess
import datetime
from .bench.benchmark import Benchmark
from .bench.energy_bench import EnergyBenchmark
import os.path
from os import path


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

parser = argparse.ArgumentParser()

def bench_with_ipu_shim():
    from .runners.tests.test_graphcore_runner_with_shim import IPU_Module_Shim
    with IPU_Module_Shim.register():
        main()

def get_benchmark_filename():
    hostname = socket.gethostname()
    gitHash = get_git_revision_short_hash()
    date = datetime.datetime.now().strftime('%m-%d-%Y')    
    prefix = ""
    log_file =  prefix + hostname  + "_" + date + "_" + gitHash
    log_file_o = log_file 
    log_file = log_file_o + ".dat"
    count = 0
    while path.exists(log_file):
        count = count +1
        log_file = log_file_o + "_" + str(count) + ".dat"
    return log_file

def bench():
    log_file = get_benchmark_filename()
    print('logging to', log_file)
    benchmark = Benchmark(log_file=log_file)
    benchmark.plot = False
    benchmark.run_unconnected = True
    benchmark.run_connected = True
    benchmark.run()

def run_energy_bench():
    log_file = get_benchmark_filename() + '.energy'
    print('logging to', log_file)
    benchmark = EnergyBenchmark(log_file=log_file)
    benchmark.run()

def usage():
    print('usage:', sys.argv[0], '<cmd>')
    print('cmd:')
    for k, v in CMDS.items():
        print(' '*10, k)

CMDS = {
        '-h': usage,
        '--help': usage,
        'bench': bench,
        'bench-shim': bench_with_ipu_shim,
        'energy': run_energy_bench
}

def main():
    if len(sys.argv) > 2:
        print('supply cmd or no args')
        exit(1)
    cmd = sys.argv[1] if len(sys.argv) == 2 else 'bench'
    f = CMDS.get(cmd)
    if f is None:
        usage()
        exit(1)
    else:
        f()

if __name__ == '__main__':
    main()
