import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib_venn import venn2
import json


'''Importing data from Jobindsats JSON file'''
with open('International Labor.json', 'r') as f:
    int_data = json.load(f)
int_lb = pd.DataFrame(int_data)

def clean_json_data(do_print = True):
    ''' Defining a callable function to use for cleaning our JSON data file '''
        
    # Copying the DataFrame, which we will clean, incase we need the original data.
    int_lb_copy = int_lb.copy()

    # As we've only extracted the data from 2014 and after, we do not need to drop any time-dependent variables.
    # First, we don't need the second and last column, so we drop these:
    int_lb_copy.drop(1, axis=1, inplace=True)
    int_lb_copy.drop(4, axis=1, inplace=True)

    # The columns are currently named 0,1,...,4. This doesn't say a lot, so we rename all columns:
    int_lb_copy.rename(columns = {0:'time'}, inplace=True)
    int_lb_copy.rename(columns= {2:'industry'}, inplace=True)
    int_lb_copy.rename(columns={3:'int_empl'}, inplace=True)

    # Our observations for international employment are currently in the 'string' format. We want them to be numbers.
    string_empl = int_lb_copy['int_empl']

    # All our observations are written as Danish 1000, e.g. 2.184 which is supposed to be 2184 and not decimals. 
    # The '.' means we can't convert the numbers directly to integers so we remove this and then convert to intergers.
    inter_empl = string_empl.str.replace('.', '').astype(int)
    
    # Lastly, we replace the string format of the original series and replace it with the new integer series:
    int_lb_copy['int_empl'] = inter_empl

    # We would like to sort our data by time. To be able to do so, we convert the 'time' variable into datetime variables.
    # All our variables are in the format 'month, year' but in Danish. So we need to translate the 'Time' values from Danish to English
    int_lb_copy['time'] = int_lb_copy['time'].str.replace("Maj", "May")
    int_lb_copy['time'] = int_lb_copy['time'].str.replace("Okt", "Oct")
    # Now we can convert our 'Time' variable into a datetime_variable.
    int_lb_copy['time'] = pd.to_datetime(int_lb_copy['time'], format='%b %Y')

    # The industries are also still in Danish, so we rename to English and in line with our data from DST:
    int_lb_copy['industry'] = int_lb_copy['industry'].str.replace('Andre serviceydelser  mv.', 'other_services')
    int_lb_copy['industry'] = int_lb_copy['industry'].str.replace('Ejendomshandel og udlejning', 'real_estate')
    int_lb_copy['industry'] = int_lb_copy['industry'].str.replace('Finansiering og forsikring', 'finance_insurance')
    int_lb_copy['industry'] = int_lb_copy['industry'].str.replace('Hoteller og restauranter', 'hotels_restaurants')
    int_lb_copy['industry'] = int_lb_copy['industry'].str.replace('Information og kommunikation', 'information_communication')
    int_lb_copy['industry'] = int_lb_copy['industry'].str.replace('Kultur og fritid', 'culture_leisure')
    int_lb_copy['industry'] = int_lb_copy['industry'].str.replace('Rejsebureau, rengøring o.a. operationel service', 'cleaning_etc')
    int_lb_copy['industry'] = int_lb_copy['industry'].str.replace('Transport', 'transport')
    int_lb_copy['industry'] = int_lb_copy['industry'].str.replace('Videnservice', 'research_consultancy')

    # We sort through the data by time.
    int_lb_cleaned = int_lb_copy.sort_values(by='time')

    if do_print:
        print(f'Before cleaning, the JSON datafile from JobIndsats contains {int_lb.shape[0]} observations and {int_lb.shape[1]} variables.')    
        print('We have removed two columns and renamed the remaining.')
        print(f'The dataset now contains {int_lb_copy.shape[0]} observations and {int_lb_copy.shape[1]} variables.')
        print(f'All our observations are of type: {type(string_empl[0])}. We want them to be integers - we use the as.type method.')
        print(f'The observations are now of type: {type(inter_empl[0])} and the first observation is: {inter_empl[0]}')
        print('We would like to sort the data by time, so we convert our time Variable into datetime variables.')
        print('All our industries are in Danish, so we rename them to English.')
     
    return int_lb_cleaned

def clean_json_data2():
    ''' Defining a callable function to use for further cleaning JSON data file '''
    int_lb = clean_json_data()
    print('For the purpose of our analysis, we want to convert the DataFrame into a pivot table, so that the data is easier to work with.')
    print('We do so using the .pivot method, using time as index, industries as columns and international labor as our observations.')
    int_lb_pivot = int_lb.pivot(index='time', columns='industry', values='int_empl')
    
    # The dataset on the service industry from DST conatins the totalt and 7 sub-industries.
    # Our dataset above contains 9 sub-industries but not the total. 
    # We therefor need to add all observations togteher to create the total:
    print('For our dataset to match the data from DST, we sum over all industries to get the total and combine four of the industires so that they match')
    int_lb_pivot['total'] = int_lb_pivot.sum(axis=1)

    # For our data to be in line with the data from DST, we need to combine some of the industries.
    # We combine the observations of 'finance and insurance' with 'real estate':
    int_lb_pivot['finance_real_estate'] = int_lb_pivot['finance_insurance'] + int_lb_pivot['real_estate']
    # We combine the observations of 'culture and leisure' with 'other services':
    int_lb_pivot['culture_leisure_other'] = int_lb_pivot['other_services'] + int_lb_pivot['culture_leisure']

    print('Lastly, we drop the industries, that we have just combined to make new ones.')
    int_lb_pivot.drop('finance_insurance', axis=1, inplace=True)
    int_lb_pivot.drop('real_estate', axis=1, inplace=True)
    int_lb_pivot.drop('other_services', axis=1, inplace=True)
    int_lb_pivot.drop('culture_leisure', axis=1, inplace=True)

    print(f'The cleaned dataset now contains 8 columns (industries) and {int_lb_pivot.shape[0]} observations')

    return int_lb_pivot


def clean_dst_empl(employees):
    ''' Defining a callable function to use for cleaning our data from DST ''' 
    print(f'Since we have extracted all the data from the source on DST, we need to select only the variables that are relevant for our analysis')
    
    params = employees._define_base_params(language='en')

    print(f'For the employment data, we first define our parameters so that we get only data from january 2014 to january 2024 and only for the total of industries.')
    params = {'table': 'LBESK03',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'BRANCHEDB071038', 'values': ['TOT']},
    {'code': 'Tid', 'values': ['>2013M12<=2024M01']}]}

    print(f'Then, we retract the parameters we defined, into our DataFrame, drop the industry since we do not need to split the data on industry, and rename the columns to english, simple titles.')
    empl = employees.get_data(params=params)
    empl.drop(['BRANCHEDB071038'], axis=1, inplace=True)
    empl.rename(columns = {'INDHOLD':'employees', 'TID':'time'}, inplace=True)
    empl['time'] = pd.to_datetime(empl['time'], format='%YM%m')

    print(f'The cleaned dataset contains {empl.shape[1]} columns and {empl.shape[0]} observations.')
    return empl


def clean_dst_shortage1(lb_short_service):
    ''' Defining a callable function to use for cleaning our data from DST ''' 
    print(f'Again, as for all the DST data, we need to select only the variables that are relevant for our analysis')

    params = lb_short_service._define_base_params(language='en')
    
    print(f'For the labor shortage data, we need to sort through the dataset a bit more when defining out variables:')
    print(f'We need to specify which industries we want to get data from, since the dataset contains both broad and narrow categories.')
    print(f'Furhtermore, we want to get data only for the labor shortage and from january 2014 to january 2024.')
    params = {'table': 'KBS2',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'BRANCHE07', 'values': [
        '000',
        '005',
        '015',
        '035',
        '045',
        '060',
        '065',
        '080'
    ]},
    {'code': 'TYPE', 'values': ['MAAK']},
    {'code': 'Tid', 'values': ['>2013M12<=2024M01']}]}

    print(f'We retrieve the parameters and sort the data by time and industry.')
    lab_short_service = lb_short_service.get_data(params=params)
    lab_short_service.sort_values(by = ['TID', 'BRANCHE07'], inplace=True)

    print(f'Then, we drop the column, TYPE, since we only have data for the labor shortage anyways, and this column would otherwise be used to split the data into diffeereeent categories of production limitations.')
    print(f'We also drop the old index and reset it.')
    
    lab_short_service.drop(['TYPE'], axis = 1, inplace = True)
    lab_short_service.rename(columns = {'BRANCHE07':'industry', 'TID':'time', 'INDHOLD':'labor_shortage'}, inplace=True)
    lab_short_service.reset_index(drop=True, inplace=True)

    print(f'We rename the industry codes to the industry names, so that they match the industries in the international labor data.')
    lab_short_service['industry'] = lab_short_service.industry.replace({
        '000':'total','SERVICES TOTAL':'total',
        '005':'transport','TRANSPORT (49-53)':'transport',
        '015':'hotels_restaurants','TOURISME (55-56, 79)': 'hotels_restaurants',
        '035':'information_communication','COMMUNICATION AND INFORMATION (58, 61-63)':'information_communication',
        '045':'finance_real_estate','FINANCE, INSURANCE AND REAL ESTATE (64-65, 68)': 'finance_real_estate',
        '060':'research_consultancy','CONSULTANCY, RESEARCH AND OTHERS (69-74)':'research_consultancy',
        '065':'cleaning_etc', 'CLEANING AND OTHER OPERATIONEL SERVICE (77-78, 81-82)':'cleaning_etc',
        '080':'culture_leisure','ARTS, RECREATION AND OTHER SERVICES (90-95)':'culture_leisure',
        })

    print(f'We convert the time variable into datetime variables.')
    lab_short_service['time'] = pd.to_datetime(lab_short_service['time'], format='%YM%m')

    print(f'The cleaned dataset contains {lab_short_service.shape[1]} columns and {lab_short_service.shape[0]} observations.')

    return lab_short_service


def clean_dst_shortage2(lb_short_manu):
    ''' Defining a callable function to use for cleaning our data from DST ''' 
    print(f'Again, as for all the DST data, we need to select only the variables that are relevant for our analysis')
    params = lb_short_manu._define_base_params(language='en')

    params = {'table': 'BARO3',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'BRANCHE07', 'values': ['C']},
    {'code': 'TYPE', 'values': ['AMA']},
    {'code': 'Tid', 'values': ['>2013K4<=2024K1']}]}

    print(f'We retreieve the parameters we defined into the DataFrame and sort the variables by time.')
    lab_short_manu = lb_short_manu.get_data(params=params)
    lab_short_manu.sort_values(by = ['TID'], inplace=True)
    print(f'We then rename the columns to english, simple titles and reset the index.')
    lab_short_manu.rename(columns={'BRANCHE07': 'industry', 'TID': 'time', 'INDHOLD': 'labor_shortage'}, inplace=True)
    lab_short_manu.reset_index(drop=True, inplace=True)
    print(f'We drop the industry and type columns, since we onle neeed data for the total industry')
    lab_short_manu.drop(['TYPE'], axis=1, inplace=True)
    lab_short_manu.drop(['industry'], axis=1, inplace=True)
    print(f'Finally, we set the time variable to datetime variables.')
    lab_short_manu['time'] = pd.to_datetime(lab_short_manu['time'], format='mixed')

    print(f'The cleaned dataset contains {lab_short_manu.shape[1]} columns and {lab_short_manu.shape[0]} observations.')
    print(f'The reason that the number of observations differ, is that manufacturing labor shortagae data is only publishedc once a quarter.')
    return lab_short_manu


def clean_dst_shortage3(lb_short_cons):
    print(f'The method for the cleaning of this dataset is exactly the same as for the manufacturinng sector.')
    params = lb_short_cons._define_base_params(language='en')

    params = {'table': 'KBYG33',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'BRANCHE07', 'values': ['F']},
    {'code': 'TYPE', 'values': ['AMA']},
    {'code': 'Tid', 'values': ['>2013M12<=2024M01']}]}

    lab_short_cons = lb_short_cons.get_data(params=params)
    lab_short_cons.drop(columns=['TYPE', 'BRANCHE07'], inplace=True)
    lab_short_cons.sort_values(by = ['TID'], inplace=True)
    lab_short_cons.rename(columns={'TID': 'time', 'INDHOLD': 'labor_shortage'}, inplace=True)
    lab_short_cons.reset_index(drop=True, inplace=True)
    lab_short_cons['time'] = pd.to_datetime(lab_short_cons['time'], format='%YM%m')

    print(f'The cleaned dataset contains {lab_short_cons.shape[1]} columns and {lab_short_cons.shape[0]} observations.')
    return lab_short_cons


def dst_empl_merging(employees):
    ''' Defining a callable function to use for cleaning our data from DST ''' 
    # Most of this code is copied from above, but this time we need the actual sub-industries.
    params = employees._define_base_params(language='en')

    params = {'table': 'LBESK03',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'BRANCHEDB071038', 'values': ['I','4','5','6','7','8','M','N','10']},
    {'code': 'Tid', 'values': ['>2013M12<=2024M01']}]}

    emplo = employees.get_data(params=params)
    emplo.rename(columns = {'INDHOLD':'employees', 'TID':'time', 'BRANCHEDB071038':'industry'}, inplace=True)
    emplo['time'] = pd.to_datetime(emplo['time'], format='%YM%m')

    # Like above, we replace all strings of industry observations with the corresponding industry name from above.
    emplo['industry'] = emplo['industry'].str.replace('I Accommodation and food service activities','hotels_restaurents')
    emplo['industry'] = emplo['industry'].str.replace('M Knowledge-based services','research_consultancy')
    emplo['industry'] = emplo['industry'].str.replace('N Travel agent, cleaning, and other operationel services','cleaning_etc')
    emplo['industry'] = emplo['industry'].str.replace('10 Arts, entertainment and recration activities','culture_leisure')
    emplo['industry'] = emplo['industry'].str.replace('4 Trade and transport etc.','transport')
    emplo['industry'] = emplo['industry'].str.replace('5 Information and communication','information_communication')
    emplo['industry'] = emplo['industry'].str.replace('6 Financial and insurance','finance_insurance')
    emplo['industry'] = emplo['industry'].str.replace('7 Real estate','real_estate')
    emplo['industry'] = emplo['industry'].str.replace('8 Other business services','other_services')

    return emplo

def checking_data(int_labor,empl_industry):
    ''' Before we merge the datasets, we check if there are any observations in either time or industry columns, that do not match '''
    print(f'Dates in empl_industry: {empl_industry.time.unique()}')
    print(f'Industries in empl_industry = {empl_industry.industry.unique()}, total = {len(empl_industry.industry.unique())}')
    print(f'Dates in int_lb: {int_labor.time.unique()}')
    print(f'Industries in int_lb = {int_labor.industry.unique()}, total = {len(int_labor.industry.unique())}')

    # Finding the date observations that are in the employee dataset, but not the interntaional workers dataset. 
    diff_d = [d for d in empl_industry.time.unique() if d not in int_labor.time.unique()]
    print(f'dates in employment data, but not in international workers data: {diff_d}')

    diff_i = [i for i in empl_industry.industry.unique() if i not in int_labor.industry.unique()] 
    # doing the same for industries
    print(f'industries in empl_industry data, but not in international workers data: {diff_i}')
    return

def merging_datasets(int_labor,empl_industry):
    ''' Defining a callable function for merging the two datasets '''
    inner = pd.merge(empl_industry,int_labor,how='inner',on=['industry','time'])
    # We use the inner-merge method on both the time and industry variables because: 
    # 1) while both data frames are sorted by the time variables, the industries are not in the same order. We therefore merge on both these variables, to make sure the corresponding values of total and international employment match,
    # 2) Even though both datasets contains the same time variable and industry variable, we do not want to risk ending up with missing variables, in case any of the datasets have observations of 0.
    inner.sort_values(by='time')
    print('Merge succesfull, the dataset now contains data on both total amount of employees and international employees')
    display(inner.head(5))
    print(f'Industries in the merged dataset: {inner.industry.unique()}, total = {len(inner.industry.unique())}')
    return inner
