import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import json

# user written modules
import dataproject

# installing API reader, that will allow to load data from DST.
%pip install git+https://github.com/alemartinello/dstapi
%pip install pandas-datareader

import pandas_datareader # install with `pip install pandas-datareader`
from dstapi import DstApi # install with `pip install git+https://github.com/alemartinello/dstapi`

class DstAPI:
    def __init__(self):
        par = self.par = SimpleNamespace()
    
    def importdata(self):

