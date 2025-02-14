import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent.resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from path import DATA_DIR  

sys.path.append(DATA_DIR / "Benchmark/chemCPA/chemCPA-main/chemCPA")
from paths import EMBEDDING_DIR

import numpy as np
import pandas as pd 
import scanpy as sc
from joblib import Parallel, delayed

data_dir =  DATA_DIR / "preprocessed_data_2025/state_transition_model_benchmark/chemCPA/random_split.h5ad"
adata = sc.read(data_dir, backed=True)
unique_smiles = np.unique(adata.obs["canonical_smiles"])

smiles_df = pd.DataFrame(unique_smiles)
smiles_df.columns = ["smiles"]
smiles_list = smiles_df['smiles'].values
print(f'Number of smiles strings: {len(smiles_list)}')

from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
generator = MakeGenerator(("RDKit2D",))
# for name, numpy_type in generator.GetColumns():
#     print(f"{name}({numpy_type.__name__})")
n_jobs = 1
data = Parallel(n_jobs=n_jobs)(delayed(generator.process)(smiles) for smiles in smiles_list )
embedding = np.array(data)
embedding.shape

drug_idx, feature_idx = np.where(np.isnan(embedding))
print(f'drug_idx:\n {drug_idx}')
print(f'feature_idx:\n {feature_idx}')

drug_idx_infs, feature_idx_infs = np.where(np.isinf(embedding))

drug_idx = np.concatenate((drug_idx, drug_idx_infs))
feature_idx = np.concatenate((feature_idx, feature_idx_infs))
embedding[drug_idx, feature_idx] = 0
df = pd.DataFrame(data=embedding,index=smiles_list,columns=[f'latent_{i}' for i in range(embedding.shape[1])]) 

# Drop first feature from generator (RDKit2D_calculated)
df.drop(columns=['latent_0'], inplace=True)

# Drop columns with 0 standard deviation
threshold = 0.01
columns=[f'latent_{idx+1}' for idx in np.where(df.std() <= threshold)[0]]
print(f'Deleting columns with std<={threshold}: {columns}')
df.drop(columns=[f'latent_{idx+1}' for idx in np.where(df.std() <= 0.01)[0]], inplace=True)

normalized_df=(df-df.mean())/df.std()
normalized_df.head()

model_name = 'rdkit2D'
dataset_name = 'lincs_trapnell'
fname = f'{model_name}_embedding_{dataset_name}.parquet'

directory = EMBEDDING_DIR /'rdkit' / 'data' /'embeddings'
directory.mkdir(parents=True, exist_ok=True)

normalized_df.to_parquet(directory / fname)

df = pd.read_parquet(directory/ fname)
df