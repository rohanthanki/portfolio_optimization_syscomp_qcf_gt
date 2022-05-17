import pandas as pd
import numpy as np

class Benchmark_Returns:
    
    def __init__(self, f_name = './final_project_data/Benchmark Returns.csv'):
        self.df = pd.read_csv(f_name)
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y%m%d')
    
    def get_max_drawdown(self):
        self.df['cum_max_msci_em'] = self.df['MSCI EM Bench Return'].cummax()
        self.df['msci_em_drawdown'] = self.df['cum_max_msci_em'] - self.df['MSCI EM Bench Return']
        self.df['cum_max_acwixus_em'] = self.df['MSCI ACWIXUS Bench Return'].cummax()
        self.df['msci_acwixus_drawdown'] = self.df['cum_max_acwixus_em'] - self.df['MSCI ACWIXUS Bench Return']
        self.df['cum_max_rus1000'] = self.df['Russell 1000 Bench Return'].cummax()
        self.df['rus1000_drawdown'] = self.df['cum_max_rus1000'] - self.df['Russell 1000 Bench Return']
        return pd.DataFrame(self.df[['msci_em_drawdown', 'msci_acwixus_drawdown', 'rus1000_drawdown']].max(axis=0), columns = ['Maximum value'])