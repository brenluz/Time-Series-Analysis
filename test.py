import numpy as np
import pandas as pd

random_numbers = np.random.normal(0,1,109)

random_series = pd.Series(random_numbers, name="random")
random_series.to_excel( "teste.xlsx", index=False, sheet_name="ARROZ",startrow=1,header=False)