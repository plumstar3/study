import numpy as np
import pandas as pd



data_csv=pd.read_csv('./soybean_samples.csv',delimiter=',')

data_npz=np.array(data_csv)

np.savez_compressed('./Soybeans_Data',data=data_npz)



