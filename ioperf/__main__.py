import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
from .bench.benchmark import Benchmark

parser = argparse.ArgumentParser()

def main():
    benchmark = Benchmark()
    benchmark.run()

if __name__ == '__main__':
    main()
