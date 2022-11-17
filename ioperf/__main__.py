import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import tempfile
from .bench.benchmark import Benchmark

parser = argparse.ArgumentParser()

def main():
    log_file = 'benchmark.txt'
    print('logging to', log_file)
    benchmark = Benchmark(log_file=log_file)
    benchmark.run()

if __name__ == '__main__':
    main()
