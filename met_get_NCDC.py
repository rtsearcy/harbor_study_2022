# getMet_NCDC.py - Download raw met data from National Climatic Data Center (NCDC).
''' RTS - 3/21/2018 (Updated 10/23/2019)
Update Aug 22 - for HMB harbor study

# Grabs hourly METAR data for airport stations along the coast, saves to file
# Parses raw data into met variables
# Note: This is a different data source of METAR data than what is used for 
# implementation (AWS). Data matches up well, though.
'''

import pandas as pd
import numpy as np
import requests
import os
import warnings

warnings.filterwarnings('ignore')

# Inputs
folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/aux_data'
airport_file = os.path.join(folder, 'airports_metadata.csv')  # file with station metadata (see below for necessary columns)

sd = '2022-07-31'  # start date, in YYYY-MM-DD format (account for previous day)
ed = '2022-08-03'  # end date, account for 8hr UTC shift

SF = 10  # scaling factor

air_list = ['Half Moon Bay'] # list(df_air.index)  # or custom list on airport locations

res = '30T'  # 30T, 1H;  find observations closest to these times


### Import Airport Stations
df_air = pd.read_csv(airport_file)
df_air.set_index('NAME', inplace=True)
print('Meterological Data\nDirectory: ' + folder )
for a in air_list:
    print('\nProcessing meteorological data for ' + a + ' (' + df_air.loc[a]['CALL'] + ')')


### Grab data from NCDC
    USAF = str(df_air.loc[a]['USAF'])
    WBAN = str(df_air.loc[a]['WBAN'])
    if len(WBAN) != 5:
        WBAN = '0'*(5-len(WBAN)) + WBAN
    st_id = USAF + WBAN
    url = 'https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-hourly'
    payload = {
        'startDate': sd,
        'endDate': ed,
        'stations': st_id,
        'format': 'json',
        'includeAttributes': 'false'
        }
    print('  Searching for raw data via NCDC')
    r = requests.get(url, params=payload)
    
    try:
        # Organize data request
        r.raise_for_status()
        df_raw = pd.DataFrame(r.json())
        df_raw = df_raw[df_raw['REPORT_TYPE'] == 'FM-15']  # METAR format only
        print('   ' + str(len(df_raw)) + ' METAR records found')
        df_raw['dt'] = pd.to_datetime(df_raw['DATE']) - pd.to_timedelta('8 hours')  # UTC to PST (-8hr)
        df_raw.set_index('dt', inplace=True)
        sd_new = str(df_raw.index[0].date())
        ed_new = str(df_raw.index[-1].date())
        print('Min. Date - ' + sd_new + '\nMax Date - ' + ed_new)
        
        # Select relevant columns
        cols = ['NAME', 'STATION', 'CALL_SIGN', 'REM', 
                'TMP', 'DEW', 'MA1', 'WND', 'OC1',
                'AA1', 'GD1','CIG','VIS']
        df_raw = df_raw[cols]
        
        ''' TMP - air temp; DEW - dew point temp; WND - wind conditions;
            AA1 - precip; GD1 - cloud cover code; VIS - horizontal visibility;
            CIG - ceiling height; MA1- pressure (SLP NAN), OC1 - wind gust '''

        # Save raw METAR data
        raw_file = a.replace(' ', '_') + '_raw_METAR_data_' + sd_new.replace('-', '') + '_' \
            + ed_new.replace('-', '') + '.csv'
        df_raw.to_csv(os.path.join(folder, raw_file))
        print('  METAR data saved to ' + raw_file)

    except Exception as exc:
        print('  There was a problem grabbing met data: %s' % exc)
        continue

    
### Process Raw METAR Data
    
    # time
    time_range = pd.date_range(sd_new, ed_new, freq=res)
    df_raw = df_raw.reindex(time_range, method='nearest')
    
    #df_raw = df_raw.resample('H').last()  # Resample by hour, selecting last value if multiple
    #df_raw.index = df_raw.index.shift(1)  # samples collected on minute 55 go to next hour
    
    # Temperature (degC)
    df_raw['temp'] = df_raw['TMP'][df_raw['TMP'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['temp'] = pd.to_numeric(df_raw['temp'], errors='coerce')/SF
    df_raw['temp'][df_raw['temp'] > 100] = np.nan  # Account for 99999 values
    print('Temperature parsed')

    # Dew Point Temperature (degC)
    df_raw['dtemp'] = df_raw['DEW'][df_raw['DEW'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['dtemp'] = pd.to_numeric(df_raw['dtemp'], errors='coerce') / SF
    df_raw['dtemp'][df_raw['dtemp'] > 100] = np.nan  # Account for 99999 values
    print('Dew point temperature parsed')

    # Sea Level Pressure (mbar)
    df_raw['pres'] = df_raw['MA1'][df_raw['MA1'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['pres'] = pd.to_numeric(df_raw['pres'], errors='coerce') / SF
    df_raw['pres'][df_raw['pres'] > 1500] = np.nan  # Account for 99999 values
    print('Sea level pressure parsed')

    # Wind Direction (deg) and Speed (m/s)
    df_raw['wdir'] = df_raw['WND'][df_raw['WND'].notnull()].apply(lambda x: x.split(',')[0])  # wind direction
    df_raw['wdir'] = pd.to_numeric(df_raw['wdir'], errors='coerce')
    df_raw['wdir'][df_raw['wdir'] > 360] = np.nan  # Account for 999 values
    print('Wind direction parsed')

    df_raw['wspd'] = df_raw['WND'][df_raw['WND'].notnull()].apply(lambda x: x.split(',')[3])  # wind speed
    df_raw['wspd'] = pd.to_numeric(df_raw['wspd'], errors='coerce') / SF
    df_raw['wspd'][df_raw['wspd'] > 90] = np.nan
    print('Wind speed parsed')
    
    # Gust (m/s)
    df_raw['gust'] = df_raw['OC1'][df_raw['OC1'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['gust'] = pd.to_numeric(df_raw['gust'], errors='coerce') / SF
    df_raw['gust'][df_raw['gust'] > 100] = np.nan  # Account for 99999 values
    print('Gust parsed')

    # Precipitation (mm)
    df_raw['rain'] = df_raw['AA1'][df_raw['AA1'].notnull()].apply(lambda x: x.split(',')[1])
    df_raw['rain'] = pd.to_numeric(df_raw['rain'], errors='coerce') / SF
    df_raw['rain'][df_raw['rain'].isnull()] = 0
    df_raw['rain'][df_raw['rain'] > 900] = np.nan
    print('Rain parsed')

    # Ceiling (m)
    df_raw['ceiling'] = df_raw['CIG'][df_raw['CIG'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['ceiling'] = pd.to_numeric(df_raw['ceiling'], errors='coerce')
    ## 22000 = unlimited
    df_raw['ceiling'][df_raw['ceiling'] > 10000] = np.nan  # Account for 99999 values
    
    # Visibility (m)
    df_raw['vis'] = df_raw['VIS'][df_raw['VIS'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['vis'] = pd.to_numeric(df_raw['vis'], errors='coerce')
    df_raw['vis'][df_raw['gust'] > 800000] = np.nan  # Account for 99999 values
    
    # Cloud (category)
    df_raw['cloud'] = df_raw['GD1'][df_raw['GD1'].notnull()].apply(lambda x: x.split(',')[0])
    df_raw['cloud'] = pd.to_numeric(df_raw['cloud'], errors='coerce')
    cloud_map = {0:'clear', 1:'few', 2:'scattered', 3:'broken', 4: 'overcast',
                 5:'obscured', 6:'partially_obs', 9: np.nan}
    df_raw['cloud'] = df_raw['cloud'].map(cloud_map)
    

### Save processed parameters 
    df_met = df_raw[['temp', 'dtemp', 'pres', 'wspd', 'wdir', 'gust',
                      'rain', 'ceiling','vis','cloud']]  

    hourly_file = a.replace(' ', '_') + '_met_data_' + sd_new.replace('-', '') + '_' \
                + ed_new.replace('-', '') + '.csv'
    df_met.to_csv(os.path.join(folder, hourly_file))
