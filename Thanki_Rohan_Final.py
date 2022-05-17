#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Russel_1000_Returns import Russel_1000_Returns
from sklearn import metrics
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import f_regression
import statsmodels.api as sm
from datetime import datetime
from Benchmark_Returns import Benchmark_Returns
from Cleaned_Returns import Cleaned_Returns
import util
import matplotlib.pyplot as plt


# In[2]:


benchmark_returns = Benchmark_Returns()
benchmark_returns.get_max_drawdown()


# In[3]:


#Get the information coefficient and t-stat for benchmark returns

benchmark_returns.df = util.get_scaled_ret(benchmark_returns.df, ['MSCI EM Bench Return', 'Russell 1000 Bench Return', 'MSCI ACWIXUS Bench Return' ])
util.get_ic(benchmark_returns.df, ['MSCI EM Bench Return', 'Russell 1000 Bench Return', 'MSCI ACWIXUS Bench Return' ])


# In[4]:


rus1000_factors = Russel_1000_Returns()
train, test = rus1000_factors.split_train_and_test(train_date = '2002-12-31')


# In[5]:


rus1000_factors.do_regression(train)
rus1000_factors.get_regression_metrics(test)
print('\n The summary results are \n')
rus1000_factors.get_fit_summary(train)


# In[6]:


models, predicted_df = rus1000_factors.get_predicted_rets()
util.filter_predicted_returns(predicted_df, models)


# In[7]:


df_sedols = pd.DataFrame()
for key in models.keys():
    df_sedols  = df_sedols.append( pd.DataFrame(models[key]['sedols'], index= [key]*len(models[key]['sedols']), columns=['SEDOL_6']))

df_sedols.reset_index(inplace = True)
df_sedols.rename(columns={'index': 'DATE'},inplace=True)
df_sedols['DATE'] = pd.to_datetime(df_sedols['DATE'])


# In[8]:


# Since we are running regression using 12 month data, it effectively averages out the coefficients
# coefs = {}

# for key, val in models.items():
#     coefs[key] = pd.read_html(val['model'].summary().tables[1].as_html(), header=0, index_col=0)[0][['coef']].to_dict()['coef']

# coefs_df = pd.DataFrame.from_dict(coefs, orient = 'index').rolling(12, min_periods = 0).mean().add_prefix('rolling_mean_')


# In[9]:


cleaned_returns = Cleaned_Returns()
merged_rets = pd.merge(cleaned_returns.df.reset_index(), df_sedols, how='inner', on = ['DATE', 'SEDOL_6'])


# In[10]:


# merged_rets.groupby('DATE').aggregate(sum).drop(columns = ['index']).describe()


# In[11]:


portfolio_ret = merged_rets.drop(columns = ['index']).groupby('DATE').mean()
cumulative_returns = pd.DataFrame(np.exp(np.log1p(portfolio_ret['RETURN_CLEAN']).cumsum()))


# In[12]:


predicted_portfolio_ret = predicted_df.reset_index()[['DATE','RETURN']].groupby('DATE').mean().pct_change()
predicted_cumulative_returns = pd.DataFrame(np.exp(np.log1p(predicted_portfolio_ret['RETURN']).cumsum())).rename(columns={'RETURN':'PREDICTED_RETURN'})

predicted_cumulative_returns.head()


# In[13]:


# pd.merge(predicted_cumulative_returns,cumulative_returns,left_index = True, right_index = True).plot(ylabel='Returns',title='Returns')
cumulative_returns.plot(ylabel='Returns',title='OLS Cumulative Returns')
plt.savefig('OLS_cumulative_returns.jpg')


# In[14]:


portfolio_ret = util.get_scaled_ret(portfolio_ret, ['RETURN_CLEAN'])
util.get_ic(portfolio_ret, ['RETURN_CLEAN' ])


# ### Read rus1000_stocks_factors.csv, normalize the features, split data into train and test. Use CTEF as the y variable

# In[15]:


rus1000_factors = Russel_1000_Returns(pred_var = 'CTEF')
train, test = rus1000_factors.split_train_and_test()


# ### Get predicted results for CTEF

# In[16]:


models, predicted_df = rus1000_factors.get_predicted_rets()
predicted_df = util.filter_predicted_returns(predicted_df, models)


# In[17]:


a = pd.DataFrame()
for key in models.keys():
    a  = a.append( pd.DataFrame(models[key]['sedols'], index= [key]*len(models[key]['sedols']), columns=['SEDOL_6']))

a.reset_index(inplace = True)
a.rename(columns={'index': 'DATE'},inplace=True)
a['DATE'] = pd.to_datetime(a['DATE'])


# ### Get the clean returns for the SEDOLs in the portfolio for the particular months

# In[18]:


merged_rets = pd.merge(cleaned_returns.df.reset_index(), a, how='inner', on = ['DATE', 'SEDOL_6'])


# ### Get the cumulative return of the portfolio

# In[19]:


portfolio_ret = merged_rets.drop(columns = ['index']).groupby('DATE').mean()
cumulative_returns = pd.DataFrame(np.exp(np.log1p(portfolio_ret['RETURN_CLEAN']).cumsum()))
cumulative_returns.plot(ylabel='Returns',title='CTEF portfolio Cumulative Returns')
plt.savefig('CTEF_cumulative_returns.jpg')


# In[20]:


portfolio_ret = util.get_scaled_ret(portfolio_ret, ['RETURN_CLEAN'])
util.get_ic(portfolio_ret, ['RETURN_CLEAN' ])


# ## Step 7

# ### Random Forest Regressor

# In[ ]:


rus1000_factors = Russel_1000_Returns(model = 'rf')
train, test = rus1000_factors.split_train_and_test()


# In[ ]:


models, predicted_df = rus1000_factors.get_predicted_rets()
predicted_df = util.filter_predicted_returns(predicted_df, models)


# In[ ]:


a = pd.DataFrame()
for key in models.keys():
    a  = a.append( pd.DataFrame(models[key]['sedols'], index= [key]*len(models[key]['sedols']), columns=['SEDOL_6']))

a.reset_index(inplace = True)
a.rename(columns={'index': 'DATE'},inplace=True)
a['DATE'] = pd.to_datetime(a['DATE'])


# ### Get the weights output file

# In[ ]:


t = pd.DataFrame(a.groupby('DATE').apply(lambda x:1/len(x))).reset_index().rename(columns={0:'weights'})
t = pd.merge(t, a, how='inner', on=['DATE'])
t['DATE'] = t['DATE'].dt.to_period('M')
t_final = t.pivot(index='DATE', columns = 'SEDOL_6', values = 'weights')
t_final.to_csv('Random_Forest_regression_weights.csv', index = True, header = True)


# ### Get the predicted returns 

# In[ ]:


p = predicted_df.reset_index()[['SEDOL_6','DATE','RETURN']]
p['DATE'] = p['DATE'].dt.to_period('M')
p = p.drop_duplicates()
t_final = p[['SEDOL_6','DATE','RETURN']].pivot_table(index='DATE', columns = 'SEDOL_6', values = 'RETURN')
t_final.to_csv('Random_Forest_regression_predicted_returns.csv', index = True, header = True)


# In[ ]:


merged_rets = pd.merge(cleaned_returns.df.reset_index(), a, how='inner', on = ['DATE', 'SEDOL_6'])


# In[ ]:


portfolio_ret = merged_rets.drop(columns = ['index']).groupby('DATE').mean()
cumulative_returns = pd.DataFrame(np.exp(np.log1p(portfolio_ret['RETURN_CLEAN']).cumsum()))
cumulative_returns.plot(ylabel='Returns',title='Random Forest Regression Cumulative Returns')
plt.savefig('RF_regressor_cumulative_returns.jpg')


# In[ ]:


portfolio_ret = util.get_scaled_ret(portfolio_ret, ['RETURN_CLEAN'])
util.get_ic(portfolio_ret, ['RETURN_CLEAN' ])


# ### Support Vector Regression

# In[21]:


rus1000_factors = Russel_1000_Returns(model = 'svr')
train, test = rus1000_factors.split_train_and_test()


# In[22]:


models, predicted_df = rus1000_factors.get_predicted_rets()
predicted_df = util.filter_predicted_returns(predicted_df, models)


# In[23]:


a = pd.DataFrame()
for key in models.keys():
    a  = a.append( pd.DataFrame(models[key]['sedols'], index= [key]*len(models[key]['sedols']), columns=['SEDOL_6']))

a.reset_index(inplace = True)
a.rename(columns={'index': 'DATE'},inplace=True)
a['DATE'] = pd.to_datetime(a['DATE'])


# ### Get the weights output file

# In[24]:


t = pd.DataFrame(a.groupby('DATE').apply(lambda x:1/len(x))).reset_index().rename(columns={0:'weights'})
t = pd.merge(t, a, how='inner', on=['DATE'])
t['DATE'] = t['DATE'].dt.to_period('M')
t_final = t.pivot(index='DATE', columns = 'SEDOL_6', values = 'weights')
t_final.to_csv('Support_vector_regression_weights.csv', index = True, header = True)


# ### Get the predicted returns 

# In[25]:


p = predicted_df.reset_index()[['SEDOL_6','DATE','RETURN']]
p['DATE'] = p['DATE'].dt.to_period('M')
p = p.drop_duplicates()
t_final = p[['SEDOL_6','DATE','RETURN']].pivot_table(index='DATE', columns = 'SEDOL_6', values = 'RETURN')
t_final.to_csv('Support_vector_regression_predicted_returns.csv', index = True, header = True)


# In[26]:


merged_rets = pd.merge(cleaned_returns.df.reset_index(), a, how='inner', on = ['DATE', 'SEDOL_6'])


# In[27]:


portfolio_ret = merged_rets.drop(columns = ['index']).groupby('DATE').mean()
cumulative_returns = pd.DataFrame(np.exp(np.log1p(portfolio_ret['RETURN_CLEAN']).cumsum()))
cumulative_returns.plot(ylabel='Returns',title='Support Vector Regression Cumulative Returns')
plt.savefig('SVR_cumulative_returns.jpg')


# In[28]:


portfolio_ret = util.get_scaled_ret(portfolio_ret, ['RETURN_CLEAN'])
util.get_ic(portfolio_ret, ['RETURN_CLEAN' ])


# In[ ]:




