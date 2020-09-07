import pickle
import os

def load_data(version='1'):
    dir_name = 'dumped_pkl'
    if version == '1':
        file_name = 'dic_nba_preprocessed.pkl'
    elif isinstance(version, str):
        file_name = 'dic_nba_preprocessed_ver{}.pkl'.format(version)
    dir_path = os.path.join(os.path.dirname(__file__), dir_name)
    file_path = os.path.join(dir_path, file_name)  # path to pickle file

    f = open(file_path, 'rb')
    dic_loaded = pickle.load(f)
    f.close()

    return dic_loaded
