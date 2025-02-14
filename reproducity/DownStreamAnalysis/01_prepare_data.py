
import pickle
import numpy as np
import pandas as pd

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR 

cell_line = sys.argv[1]

all_suggested_plans = []
for step in range(2, 10):
    
    file_path = ROOT_DIR.parent / f"output/AlphaTherapy/ENV_FDA_{cell_line}_STEP{step}.csv"
    plans = pd.read_csv(file_path, index_col=0)
    all_suggested_plans.append(plans)

total_res = pd.concat(all_suggested_plans)  
total_res = total_res.drop_duplicates(['first_ind', 'second_ind'])

outfile_path = DATA_DIR / f"preprocessed_data_2025/down_stream_analysis/plans/{cell_line}.csv"
outfile_path.parent.mkdir(parents=True, exist_ok=True)

total_res.to_csv(outfile_path, index=None)