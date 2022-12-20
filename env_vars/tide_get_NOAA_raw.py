#! python3
# getTides.py - Download bulk tidal data from NOAA CO-OPS

# SET DATE RANGE IF NEEDED

import os
import requests
import json
from datetime import timedelta
from dateutil.parser import parse
import pandas as pd

stations_dict = {
    #'Monterey': '9413450',
    'San Francisco': '9414290',
}

path = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/aux_data/'
path = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/hindcast'

for key in stations_dict:
    station_num = stations_dict[key]  # San Diego
    station_name = key
    begin_date = '20170101'
    end_date = '20220804'

    datum = 'MLLW'  # Mean Lower Low Water (Lowest tide if diurnal)
    units = 'metric'
    time_zone = 'lst'  # Local Standard Time (ignore DLS)
    #product = 'predictions'
    product = 'water_level'
    form = 'json'

    begin_date = parse(begin_date)
    end_date = parse(end_date)
    bd = begin_date
    if (end_date - begin_date).days > 30:
        ed = begin_date + timedelta(days=29)  # NOAA-COOPS allows for up to 3650 days of data per grab
    else:
        ed = end_date # use if you only want a discrete amount of data (i.e. after 2020)
    
    c = 1

    print('Collecting ' + station_name + ' tidal data...')
    while ed <= end_date:
        if c != 1:
            bd = ed + timedelta(days=1)
            ed = ed + timedelta(days=29)  # Account for timestep limit
            if ed > end_date:
                ed = end_date
        print('   Searching for data from ' + bd.strftime('%Y%m%d') + ' to ' + ed.strftime('%Y%m%d'))
    
        url = 'https://tidesandcurrents.noaa.gov/api/datagetter?' + \
            'begin_date=' + bd.strftime('%Y%m%d') + \
            '&end_date=' + ed.strftime('%Y%m%d') + \
            '&station=' + station_num + \
            '&product=' + product + \
            '&datum=' + datum + \
            '&units=' + units + \
            '&time_zone=' + time_zone + \
            '&format=' + form + \
            '&application=web_services'
    
        web = requests.get(url)
        try:
            web.raise_for_status()
        except Exception as exc:
            print('   There was a problem with the URL: %s' % exc)
        data = json.loads(web.text)
        #data = data['data']
    
        try:
            if product == 'predictions':
                data = data['predictions']
            
            else:
                data = data['data']
        except KeyError:
            print('   Could not find data for the following station: ' + station_name)
            continue
        print('   JSON data loaded. Parsing')
    
        if c == 1:
            df = pd.DataFrame.from_dict(data)
        else:
            df = df.append(pd.DataFrame.from_dict(data), ignore_index=True)  # Add to exisiting df
        print('   Data parsed')
        c += 1

    # Save to file
    if product == 'predictions':
        df.columns = ['dt', 'tide']
    else:
        df.columns = ['dt', 'tide', 'std','flag','qa']
    save_file = os.path.join(path, station_name.replace(' ', '_') + '_tide_' + product + '_' +  \
    begin_date.strftime('%Y%m%d') + '_' + end_date.strftime('%Y%m%d') + '.csv')
    df.to_csv(save_file, index=False)
    print(station_name + ' tidal data written to file: ' + save_file)
