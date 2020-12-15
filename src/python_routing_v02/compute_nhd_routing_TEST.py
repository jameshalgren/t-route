#!/usr/bin/env python
# coding: utf-8

"""NHD Network test cases

Test v02 routing on specific test cases

"""
## Parallel execution
import os
import sys
import time
import numpy as np
import argparse
import pathlib
import glob
import pandas as pd
from functools import partial
from joblib import delayed, Parallel
from itertools import chain, islice
from operator import itemgetter