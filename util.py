import pandas as pd
import statsmodels.api as sm
import numpy as np

def filter_predicted_returns(pred_df, models):

    predicted_dates = pd.DataFrame(models.keys(), columns = ['Date'])
    if 'DATE' not in list(pred_df.columns):
        pred_df = pred_df.reset_index() 

    for i in range(1,len(predicted_dates)):
        k = 4
        
        d2 = predicted_dates.loc[i]['Date']
        d1 = predicted_dates.loc[i-1]['Date']
        
        a = models[d2]['sedols']
        b = models[d1]['sedols']
        
        if len(list(a.difference(b))) < k:
            k = len(list(a.difference(b)))
        
        a_b = set(pd.DataFrame(a.difference(b)).sample(k)[0])
        b_a = set(pd.DataFrame(b.difference(a)).sample(k)[0])
        
        b = b.difference(b_a).union(a_b)
#         a = a.difference(a_b).union(b_a)
        
        df_da = pred_df[(pred_df['DATE'] == d2) & (pred_df['SEDOL_6'].isin(list(a_b)))]['DATE'].unique()[0]
        df_db = pred_df[(pred_df['DATE'] == d1) & (pred_df['SEDOL_6'].isin(list(b_a)))]['DATE'].unique()[0]
        
#         pred_df.loc[(pred_df['DATE'] == d2) & (pred_df['SEDOL_6'].isin(list(a_b))), ['DATE']] = df_db
        pred_df.loc[(pred_df['DATE'] == d1) & (pred_df['SEDOL_6'].isin(list(b_a))), ['DATE']] = df_da
    return pred_df

#get scaled returns (convert returns to percentile)
def get_scaled_ret(df, return_cols):
    
    for col in return_cols:
        df['Scaled ' + str(col)] = pd.qcut(df[col], q = 100, labels = False) + 1
    
    return df

def get_coef_table(lin_reg):
    ''' lin_reg is a fitted statsmodels regression model
    Return a dataframe containing coefficients, pvalues, and the confidence intervals
    '''
    err_series = lin_reg.params - lin_reg.conf_int()[0]
    coef_df = pd.DataFrame({'coef': lin_reg.params.values[1:],
                            'ci_err': err_series.values[1:],
                            'pvalue': lin_reg.pvalues.round(4).values[1:],
                            'varname': err_series.index.values[1:]
                           })
    return coef_df


# get information coeff
def get_ic(df, return_cols):

    ic_df = pd.DataFrame(columns=['Portfolio', 'IC', 'T-test'])
    for col in return_cols:
        lr = sm.OLS(df[col] * 100 , np.array(df['Scaled ' + str(col)]).reshape(-1, 1)).fit()
        ic_df = ic_df.append({'Portfolio': col, 'IC': lr.params[0] , 'T-test': lr.pvalues[0]}, ignore_index = True)
    
    print(ic_df)