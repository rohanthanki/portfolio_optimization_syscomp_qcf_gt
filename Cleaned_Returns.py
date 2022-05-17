import pandas as pd

class Cleaned_Returns:
    def __init__(self, f_name = './final_project_data/cleaned_return_data_sc.csv'):
        cleaned_ret = pd.read_csv(f_name).T
        new_header = cleaned_ret.iloc[0]
        cleaned_ret = cleaned_ret[1:]
        cleaned_ret.columns = new_header
        self.df =  pd.melt(cleaned_ret, var_name= 'DATE', value_name='RETURN_CLEAN', ignore_index=False).reset_index().rename(columns={'index': 'SEDOL_6'})
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.df['DATE'] = self.df['DATE'] + pd.tseries.offsets.MonthEnd(0)
        self.df.set_index(['DATE', 'SEDOL_6'])
        self.df['RETURN_CLEAN'] = pd.to_numeric(self.df['RETURN_CLEAN'])