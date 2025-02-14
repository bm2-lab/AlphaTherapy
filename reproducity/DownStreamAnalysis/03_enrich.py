
import pickle
import numpy as np
import pandas as pd


import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR 

cell_line = sys.argv[1]

total_res_path = DATA_DIR / f"preprocessed_data_2025/down_stream_analysis/plans/{cell_line}.csv"
null_ego_expression_path = ROOT_DIR / f"../data/null_drugA_expression.csv"
drugA_expression_data_path = DATA_DIR / f"preprocessed_data_2025/down_stream_analysis/expression_res/"
output_path =  DATA_DIR / f"preprocessed_data_2025/down_stream_analysis/enrich_res/"

output_path.mkdir(parents=True, exist_ok=True)

import subprocess

command = [
    "Rscript",
    "enrich.R",
    str(cell_line),
    str(total_res_path),
    str(null_ego_expression_path),
    str(drugA_expression_data_path)+"/",
    str(output_path)+"/",
]

# Run the R script
try:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print("R Script Output:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("Error running R script:")
    print(e.stderr)
