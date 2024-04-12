import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import json
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

'''Importing data from Jobindsats JSON file'''
with open('International Labor.json', 'r') as f:
    int_data = json.load(f)
int_lb = pd.DataFrame(int_data)

def clean_data():
    ''' Defining a callable function to use for cleaning our JSON data file '''
    print(f'Before cleaning, the JSON datafile from JobIndsats contains {int_lb.shape[0]} observations and {int_lb.shape[1]}')

    # Copying the DataFrame, which we will clean, incase we need the original data.
    int_lb_copy = int_lb.copy()

    # As we've only extracted the data from 2014 and after, we do not need to drop any time-dependent variables.
    # First, we don't need the second and last column, so we drop these:
    int_lb_copy.drop(1, axis=1, inplace=True)
    int_lb_copy.drop(4, axis=1, inplace=True)

    # The columns are currently named 0,1,...,4. This doesn't say a lot, so we rename all columns:
    int_lb_copy.rename(columns = {0:'Time'}, inplace=True)
    int_lb_copy.rename(columns= {2:'Industry'}, inplace=True)
    int_lb_copy.rename(columns={3:'int_empl'}, inplace=True)

    print('We have removed two columns and renamed the remaining.')
    print(f'The dataset now contains {int_lb_copy.shape[0]} observations and {int_lb_copy.shape[1]} variables')

    # Our observations for international employment are currently in the 'string' format. We want them to be numbers.
    string_empl = int_lb_copy['int_empl']
    print(f'All our observations are of type: {type(string_empl[0])}. We want them to be integers.')

    # All our observations are written as Danish 1000, e.g. 2.184 which is supposed to be 2184 and not decimals. 
    # The '.' means we can't convert the numbers directly to integers so we convert them to floats first:
    float_empl = string_empl.astype(float)
    print(f'The observations are now of type: {type(float_empl[0])} and the first observation is: {float_empl[0]}')

    # Next we multiply all observations by 1000 and convert to integers:
    inter_empl = float_empl.multiply(1000).astype(int)
    print(f'The observations are now of type: {type(inter_empl[0])} and the first observation is: {inter_empl[0]}')
    
    # Lastly, we replace the string format of the original series and replace it with the new integer series:
    int_lb_copy['int_empl'] = inter_empl

    # We now sort through the data by, first by sorting the data by time.
    int_lb_copy.sort_values(by='Time')
    print('We now convert the DataFrame using the .pivot method, using time as index, industries as columns and international labor as our observations.')
    int_lb_pivot = int_lb_copy.pivot(index='Time', columns='Industry', values='int_empl')

    # The industries are still in Danish, rename to English and in line with our data from DST:
    print('All our industries are in Danish, so we rename them to english')
    int_lb_pivot.rename(columns={'Andre serviceydelser  mv.':'other_services'}, inplace=True)
    int_lb_pivot.rename(columns={'Ejendomshandel og udlejning':'real_estate'}, inplace=True)
    int_lb_pivot.rename(columns={'Finansiering og forsikring':'finance_insurance'}, inplace=True)
    int_lb_pivot.rename(columns={'Hoteller og restauranter':'hotels_restaurents'}, inplace=True)
    int_lb_pivot.rename(columns={'Information og kommunikation':'information_communictaion'}, inplace=True)
    int_lb_pivot.rename(columns={'Kultur og fritid':'culture_leisure'}, inplace=True)
    int_lb_pivot.rename(columns={'Rejsebureau, rengøring o.a. operationel service':'cleaning_etc'}, inplace=True)
    int_lb_pivot.rename(columns={'Transport':'transport'}, inplace=True)
    int_lb_pivot.rename(columns={'Videnservice':'research_consultancy'}, inplace=True)

    # The dataset on the service industry from DST conatins the totalt and 7 sub-industries.
    # Our dataset above contains 9 sub-industries but not the total. 
    # We therefor need to add all observations togteher to create the total:
    print('For our dataset to match the data from DST, we sum over all industries to get the total and combine four of the industires so that they match')
    int_lb_pivot['total'] = int_lb_pivot.sum(axis=1)

    # Now we combine the observations of 'finance and insurance' with 'real estate':
    int_lb_pivot['finance_real_estate'] = int_lb_pivot['finance_insurance'] + int_lb_pivot['real_estate']

    # We combine the observations of 'culture and leisure' with 'other services':
    int_lb_pivot['culture_leisure_other'] = int_lb_pivot['other_services'] + int_lb_pivot['culture_leisure']

    # Make a final copy, incase we need the original data before dropping the last columns
    print('Lastly, we drop the industries, that we have just combined to make new ones.')
    int_lb_cleaned = int_lb_pivot.copy()
    int_lb_cleaned.drop('finance_insurance', axis=1, inplace=True)
    int_lb_cleaned.drop('real_estate', axis=1, inplace=True)
    int_lb_cleaned.drop('other_services', axis=1, inplace=True)
    int_lb_cleaned.drop('culture_leisure', axis=1, inplace=True)

    print(f'The cleaned dataset now contains 8 columns (industries) and {int_lb_cleaned.shape[0]} observations')

    return


