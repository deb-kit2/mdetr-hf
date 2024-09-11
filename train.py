import os
import argparse

import logging


def parse_args() :
    parser = argparse.ArgumentParser()

    parser.add_argument("--")

    args = parser.parse_args()
    return args


if __name__ == "__main__" :
    args = parse_args()

    logging.basicConfig()