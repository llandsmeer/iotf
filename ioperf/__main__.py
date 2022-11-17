import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
from .runners import runners

parser = argparse.ArgumentParser()
parser.add_argument('--runner',
                    choices=list(runners),
                    required=True,
                    help='Which runner to use')
def main():
    args = parser.parse_args()
    raise NotImplementedError('good luck')
    runners[args.runner]()

if __name__ == '__main__':
    main()
