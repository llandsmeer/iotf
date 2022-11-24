import os
import socket
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
import tempfile
import socket
import subprocess
import datetime
from .bench.benchmark import Benchmark
import os.path
from os import path


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

parser = argparse.ArgumentParser()

def main():
    hostname = socket.gethostname()
    gitHash = get_git_revision_short_hash()
    date = datetime.datetime.now().strftime('%m-%d-%Y')    
    log_file =  hostname  + "_" + date + "_" + gitHash
    log_file_o = log_file 
    log_file = log_file_o + ".dat"
    count = 0
    while path.exists(log_file):
        count = count +1
        log_file = log_file_o + "_" + str(count) + ".dat"
    print('logging to', log_file)
    benchmark = Benchmark(log_file=log_file)
    benchmark.run()

if __name__ == '__main__':
    main()
