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

# importing the actual data from DST
employees = DstApi('LBESK03')
lb_short_service = DstApi('KBS2')
lb_short_manu = DstApi('BARO3')
lb_short_cons = DstApi('KBYG33')
with open('International Labor.json', 'r') as f:
    int_data = json.load(f)
int_lb = pd.DataFrame(int_data)

# cleaning the 'LBESK03' dataset
params = employees._define_base_params(language='en')

params = {'table': 'LBESK03',
 'format': 'BULK',
 'lang': 'en',
 'variables': [{'code': 'BRANCHEDB071038', 'values': ['TOT']},
  {'code': 'Tid', 'values': ['>2013M12<=2024M01']}]}

empl = employees.get_data(params=params)
empl.head(5)


