import os
import sys
import gym
import argparse
import subprocess
import numpy as np
import pandas as pd

from utils import expression_simulate, simulate_SDME, synergy_data_preprocessing, segments_fit, self_designed_cluster_func, match_key_pathway

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_dir + "/scripts/downstream_analysis/src/")


# 1. The parameters of input
argparser = argparse.ArgumentParser()
argparser.add_argument('--combos_file_name', type=str, help='the data file name (*.csv) of a table of AlphaTherapy-identified sequential drug treatments', default="ENV_FDA_MCF7_STEP2.csv")
argparser.add_argument('--env_name', type=str, help='the cell model environment setting name', default="ENV_FDA_MCF7_STEP2")
argparser.add_argument('--drugA_name', type=str, help='the first drug', default="Camptothecin")
argparser.add_argument('--drugB_name', type=str, help='the second drug', default="Epirubicin HCl")

args = argparser.parse_args()

AlphaTherapy_identified_result_file = project_dir + "/output/AlphaTherapy/" + args.combos_file_name
result_data = pd.read_csv(AlphaTherapy_identified_result_file, index_col=0)
# find corresponding item
item = result_data.loc[(result_data["first_drug"]==args.drugA_name) & (result_data["second_drug"]==args.drugB_name)]
drugA_index = item['first_ind'].values[0]
drugB_index = item['second_ind'].values[0]


# 2. simulate expression profiles during drug A treatment
env = gym.make('gym_cell_model:ccl-env-cpd-v1', env_name=args.env_name)
expression_simulate(args.env_name, drugA_index, env)

# run enrich R script
R_file = project_dir + "/scripts/downstream_analysis/src/enrich.R"
command = ["Rscript", R_file, args.env_name, str(drugA_index)]
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

stdout, stderr = process.communicate()
print("Enrich.R Standard Output:\n", stdout)
print("Enrich.R Standard Error:\n", stderr)

# 3. downstream analysis
# simulate SDE vectors
drugA_synergy_vec, drugB_synergy_vec = simulate_SDME(item.iloc[0,:], env)

# SDE vecter preprocessing
drugA_synergy_vec = synergy_data_preprocessing(drugA_synergy_vec)
max_step = np.argmax(drugA_synergy_vec) + 1

# piecewise linear regression
X = np.arange(10)
Y = drugA_synergy_vec
final_count, slope_arr, px, py = segments_fit(X, Y, 3)
# trend classification
cluster_label = self_designed_cluster_func(slope_arr, thre=0.01)

# save results
enrichment_result_file_name = project_dir + "/scripts/downstream_analysis/working_log/" + args.env_name + "_drugA_" + str(drugA_index) + "_enrichment.csv"
enrich_pathway_df = pd.read_table(enrichment_result_file_name, index_col=0)
key_pathways = match_key_pathway(drugA_synergy_vec, cluster_label, enrich_pathway_df)

cluster_dict = {1:"No trend", 2:"Monotonic increasing", 3:"Monotonic decreasing", 4:"Increasing then decreasing", 5:"Decreasing then increasing", 6:"Increasing then decreasing then increasing", 7:"Decreasing then increasing then decreasing"}
 
title = f"For the sequential drug combination {args.drugA_name} -> {args.drugB_name} in the setting {args.env_name}: \n"

res1 = f"TREND RESULT\n As the duration of {args.drugA_name} administration increases, the efficacy of {args.drugB_name} exhibits ({cluster_dict[cluster_label]}) trend.\n" 
res2 = f"The optimal time step of {args.drugA_name} is observed between [{max_step-1}, {max_step+1}]. \n"
res3 = f"STATE RESULT\n Results of pathway-associated cell state transitions: \n"

output_file_name = project_dir + "/output/downstream_analysis/" + "%s_%s_%s.txt" % (args.env_name, args.drugA_name, args.drugB_name)
with open(output_file_name, 'w') as file:
    file.write(title)
    file.write(res1)
    file.write(res2)
    file.write(res3)

    if len(key_pathways) == 0:
        file.write("NULL")
    else:
        for p in key_pathways.index:
            file.write(p + '\n')

