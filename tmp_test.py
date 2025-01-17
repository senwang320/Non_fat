
#%%
import pandas as pd
import time

fname = './data/fat_list.xlsx'

data  = pd.read_excel(fname)

# Build a dictionary using column 0 as the keys and the rest of the row as values
dict_from_df = data.set_index(data.columns[0]).to_dict(orient='index')
# %%
