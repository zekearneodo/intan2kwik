import datetime
import pandas as pd
import numpy as np
import glob
import os

def datetime_from_filename(filename: str) -> datetime.datetime:
    datetime_str = ''.join(filename.split('.')[0].split('_')[-2:])
    datetime_tuple = tuple([int(datetime_str[i:i+2]) for i in range(0,12,2)])
    t_stamp = datetime.datetime(*datetime_tuple)
    
    # now date is yymmdd, time is hhmmss
    # turn into datetime object
    return t_stamp

def get_rec_breaks(rhd_pd: pd.DataFrame) -> pd.DataFrame:
    rhd_pd['t_diff'] = rhd_pd['t_stamp'].diff()
    rhd_pd['rec_break'] = rhd_pd['t_diff'].apply(lambda x: True if x.seconds%60>0 else False)
    break_indices = np.hstack([rhd_pd[rhd_pd['rec_break']].index.values, [rhd_pd.index.values[-1], 0]])
    break_indices.sort()
    rec_segments = np.vstack([break_indices[:-1], break_indices[1:]]).T

    for i_rec, segment_edges in enumerate(rec_segments):
        rhd_pd.loc[segment_edges[0]:segment_edges[1], 'rec'] = i_rec
    rhd_pd['rec'] = rhd_pd['rec'].astype(np.int)
    return rhd_pd
    
def get_rhd_pd(raw_folder: str, file_extension: str='rhd') -> pd.DataFrame:
    all_rhd_list = glob.glob(os.path.join(raw_folder, '*.{}'.format(file_extension)))
    all_rhd_list.sort()
    all_rhd_pd = pd.DataFrame(all_rhd_list, columns=['path'])
    all_rhd_pd['f_name'] = all_rhd_pd['path'].apply(lambda x: os.path.split(x)[1])
    all_rhd_pd['t_stamp'] = all_rhd_pd['f_name'].apply(lambda x: datetime_from_filename(x))
    #all_rhd_pd.loc[-1:, 'rec_break'] = True
    all_rhd_pd = get_rec_breaks(all_rhd_pd)
    return all_rhd_pd
