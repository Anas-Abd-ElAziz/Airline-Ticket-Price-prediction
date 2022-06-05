import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess(data):

    # date
    data["date"] = pd.to_numeric(pd.to_datetime(data["date"]))
    data['date'].fillna(data['date'].mean(), inplace=True)

    # airline
    airline_map = {'Trujet':1, 'StarAir':2 ,'SpiceJet':3, 'AirAsia':4, 'GO FIRST':5, 'Indigo':6, 'Air India':7, 'Vistara':8}
    data['airline'] = data['airline'].map(airline_map)
    data['airline'].fillna(data['airline'].mode(), inplace=True)

    # ch_code
    data = data.drop(['ch_code'], axis=1)

    # convert "dep_time" column into datetime
    data["dep_time"] = pd.to_numeric(pd.to_datetime(data['dep_time']))
    data['dep_time'].fillna(data['dep_time'].mean(), inplace=True)

    # convert time_taken into minutes
    h = pd.to_numeric(data['time_taken'].str.split(' ', expand=True)[0].str[:-1], downcast="float")
    m = pd.to_numeric(data['time_taken'].str.split(' ', expand=True)[1].str[:-1], downcast="float")
    data['time_taken'] = (m + h * 60)
    data['time_taken'].fillna(data['time_taken'].mean(), inplace=True)

    # arr_time
    data["arr_time"] = pd.to_numeric(pd.to_datetime(data['arr_time']))
    data['arr_time'].fillna(data['arr_time'].mean() , inplace=True)
    
    # route
    data['route'].fillna(data['route'].mode(), inplace=True)
    data['source'] =data['route'].str.split(',', expand=True)[0].str.split(':', expand=True)[1].str.split("'", expand=True)[1]
    data['destination'] =data['route'].str.split(',',expand=True)[1].str.split(':',expand=True)[1].str.split("'",expand=True)[1]
    data = data.drop(['route'], axis=1)
    
    route_map = {'Chennai':1, 'Hyderabad':2 ,'Kolkata':3, 'Bangalore':4, 'Mumbai':5, 'Delhi':6}
    data['source'] = data['source'].map(route_map)
    data['source'].fillna(data['source'].mode(), inplace=True)
    data['destination'] = data['destination'].map(route_map)
    data['destination'].fillna(data['destination'].mode(), inplace=True)
    

    # stop
    data['stop'] = data['stop'].fillna(data['stop'].mode())
    data['stop'] = data['stop'].str.split('p', expand=True)[0].str.replace('+','',regex=True)+'p'
    stop_map = {'non-stop':1, '1-stop':2 ,'2-stop':3}
    data['stop'] = data['stop'].map(stop_map)
    data['stop'].fillna(data['stop'].mode(), inplace=True)
    
    # type
    data['type'] = data['type'].fillna(data['type'].mode())
    type_map = {'economy':1, 'business':2}
    data['type'] = data['type'].map(type_map)
    data['type'].fillna(data['type'].mode(), inplace=True)
    
    
    
    #data = pd.get_dummies(data, columns = ['type', 'stop','source','destination'], drop_first=True)
    

    return data
