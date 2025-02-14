cd ./scripts/gym_cell_model
pip install -e .
cd ../tianshou-master
pip install -e .
cd ../..

################# PART1. Prepare Raw Data #################
# First, update the DATA_DIR variable in ./reproducity/path.py to specify the location where your data is stored. 
# You will need to download the raw data: 
#   1. CTRP: The data can be downloaded from CTRPv2.0 and should be placed in DATA_DIR/raw_data/CTRP. 
#   2. LINCS_GSE92742: The data can be downloaded from GEO LINCS_GSE92742 and should be placed in DATA_DIR/raw_data/LINCS_GSE92742.

################# PART2. Run main scripts #################
# 1. Cell viabilility Prediction Model
cd ./reproducity/CellViabilityModel
python 01_data_preprocess.py
python 02_model_training.py

# 2. State Transition Model
cd ../StateTransitionModel/DataPreprocess
python 01_data_preprocess_step1.py
python 01_data_preprocess_step1_5uM.py
python 02_data_preprocess_step2_random_split.py
python 02_data_preprocess_step2_cell_blind.py
python 02_data_preprocess_step2_chemical_blind.py
python 02_data_preprocess_step2_random_split_5uM.py

cd ../Model
python write_config.py 
python train.py
python evaluation.py

cd ../Evaluation
python LinearReconstruction.py 
python TimeEvalution.py 

cd ../../AlphaTherapy
python generate_figure2g_data.py
python generate_figure2i_data.py

# Benchmark 
# Before running the following commands, we recommend setting up a new conda environment according to the instructions in the chemCPA tutorial (https://github.com/theislab/chemCPA/)
# as the environments for which and AlphaTherapy may conflict. 
# To ensure the consistency of the training control data, you need to install the chemcpa package provided by us.
# Note: You need to copy the lincs_01.yaml file from the /Benchmark/chemCPA/chemCPA-main folder into your installed chemCPA-main folder. 
# Additionally, copy the files c824e42f7ce751cf9a8ed26f0d9e0af7.pt and chemCPA_configs.json from the /Benchmark/chemCPA/chemCPA-main/project_folder folder into your installed chemCPA-main/project_folder folder.
cd ../Benchmark/chemCPA/src
python 01_1_Data_preparation.py
python 01_2_Data_preparation.py
python 01_3_Data_preparation.py
python 01_4_Data_embedding.py
python 02_train.py
python 03_evaluation.py

# Before running the following commands, we recommend setting up a new conda environment according to the instructions in the scgen tutorial (https://github.com/theislab/scgen/)
cd ../../scGen
python 01_Data_preparation.py
python 02_train.py
python 03_evaluation.py

cd ../StateTransitionModel
python 01_evaluation.py

# 3. AlphaTherapy Model
# The following programs are very time-consuming. 
# Before running them, please first test the runtime for a single script, specifically for one cell line and one termination step. 

cd ../../..
python ./scripts/utils/build_drugset.py --input FDA_drugset.xlsx --output FDA_drugset.pickle

cell_lines=("A375" "MDAMB231" "SKBR3" "MCF7" "HS578T" "BT20" "HEPG2" "HUH7" "HCC515" "A549")
terminal_steps=(2 3 4 5 6 7 8 9)

for cell_line in "${cell_lines[@]}"; do
  for terminal_step in "${terminal_steps[@]}"; do
    python ./scripts/utils/write_scenario_configs.py --drugset_file_name FDA_drugset.pickle --drugset_short_name FDA --cell_line "$cell_line" --terminal_step "$terminal_step"
    python ./scripts/AlphaTherapy/model/train_RL_agents.py --drugset_short_name FDA --cell_line "$cell_line" --terminal_step "$terminal_step" --max_workers 10
    python ./scripts/AlphaTherapy/model/output_result.py --drugset_short_name FDA --cell_line "$cell_line" --terminal_step "$terminal_step"
  done
done

# 4. Downstream Analysis
cd ./reproducity/DownStreamAnalysis/
for cell_line in "${cell_lines[@]}"; do
    python 01_prepare_data.py "$cell_line"
    python 02_simulate_expression.py "$cell_line"
    python 03_enrich.py "$cell_line"
    python 04_run_analysis.py "$cell_line"
    python 05_MoA_analysis.py "$cell_line"
done

# After all these scripts, you can run the plot scripts in Notebooks folder.
