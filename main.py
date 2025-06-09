import subprocess
import argparse

import h5utils

parser = argparse.ArgumentParser(description='Add arguments')

parser.add_argument(
    'action',
    type=str,
    help='e - encode, d - decode'
)

parser.add_argument(
    'inpath',
    type=str,
    help='Provide source file path'
)
parser.add_argument(
    'outpath',
    type=str,
    help='Provide output file path'
)

parser.add_argument(
    'endtime',
    type=str,
    help='0.02'
)

args = parser.parse_args()

def encode(input_path, end_time, output_path):

    h5utils.h5_encode(input_path, end_time)

    #subprocess.call(["waverange/bin/flusi/./wrenc", input_path, output_path, "0", "1e-16"])


def decode(input_path, end_time, output_path):
    subprocess.call(["waverange/bin/flusi/./wrdec", input_path, output_path, "0", "1"])


if args.action == "e":
    encode(args.inpath, args.endtime, args.outpath)

elif args.action == "d":
    decode(args.inpath, args.endtime, args.outpath)