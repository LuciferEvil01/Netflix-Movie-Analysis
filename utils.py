# utils.py
import pandas as pd

def convert_duration(duration):
    if pd.isna(duration):
        return 0
    if 'min' in duration:
        return int(duration.replace(' min', ''))
    elif 'Season' in duration:
        return int(duration.split(' ')[0]) 
    return 0
