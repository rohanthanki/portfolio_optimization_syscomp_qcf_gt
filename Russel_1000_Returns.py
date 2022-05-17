import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import Normalizer
import statsmodels.api as sm
from datetime import datetime
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


class Russel_1000_Returns:

    def __init__(self, f_name='./final_project_data/rus1000_stocks_factors.csv', o_f_name = 'processed_input.csv', pred_var='RETURN', model='lr'):
        self.remove_extra_commas()
        cols = pd.read_csv(f_name, nrows=3, skiprows=4).columns
        self.model = model
        self.pred_var = pred_var
        self.df = pd.read_csv(o_f_name, names=cols, header=0,
                              skiprows=5, index_col=False)
        self.df.dropna(axis=1, inplace=True, how='all')
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.df['DATE'] = self.df['DATE'] + pd.tseries.offsets.MonthEnd(0)
        self.df['RETURN'] = pd.to_numeric(self.df['RETURN'], errors='coerce')
        self.df['CTEF'] = pd.to_numeric(self.df['CTEF'], errors='coerce')
        self.df['SEDOL'] = self.df['SEDOL'].astype('str')
        self.df['SEDOL_6'] = self.df['SEDOL'].str[:6]
        self.df.set_index('DATE', inplace=True)
        self.df[self.pred_var] = self.df.groupby(
            'SEDOL').shift(-1)[self.pred_var]
        self.df = self.df[~self.df[self.pred_var].isna()]
        self.df.reset_index(inplace=True)
        self.df.drop(columns=['Symbol', 'FS_ID', 'Company Name'], inplace=True)
        self.df.set_index(['SEDOL', 'SEDOL_6', 'DATE'], inplace=True)
        self.normalize_and_clean_data()
        self.get_set_best_cols(self.df)

    def remove_extra_commas(self, f_name='./final_project_data/rus1000_stocks_factors.csv', o_f_name='processed_input.csv'):
        o = open(o_f_name, 'w')

        with open(f_name, 'r') as f:
            for i in range(0, 4):
                f.readline()
            o.write(f.readline())
            for line in f:
                if len(line) > 2:
                    line_content = line.split(',')
                    if len(line_content) > 29:
                        line_content[1] = f'"{line_content[1]},{line_content[2]}"'
                        del line_content[2]

                    o.write(','.join(line_content))

            o.close()

    def get_sedol_bin_counts(self):
        self.df['bin'] = pd.cut(self.df[self.pred_var], range(0, 101, 10))
        return self.df.groupby('bin').size()

    def normalize_and_clean_data(self):
        self.df.groupby('SEDOL').fillna(
            method='bfill').reset_index(drop=True, inplace=True)
        y_col = self.pred_var

        imputer = KNNImputer(n_neighbors=2, weights="distance")
        cols = self.df.columns.difference([y_col])
        inx = self.df.index
        x = imputer.fit_transform(self.df[cols])

        transformer = Normalizer().fit(x)
        x = transformer.transform(x)
        x = pd.DataFrame(x, columns=cols, index=inx)
        self.df = pd.merge(x, self.df[[y_col]],
                           left_index=True, right_index=True)

    def split_train_and_test(self, train_date='2004-09-30', n_months=2):
        train_start_date = datetime.strptime(train_date, '%Y-%m-%d')
        self.df.reset_index(inplace=True)
        temp_date = train_start_date + pd.tseries.offsets.MonthEnd(n_months)
        train_end_date = temp_date if temp_date <= self.df['DATE'].max(
        ) else self.df['DATE'].max()
        train = self.df[(self.df['DATE'] >= train_start_date) & (
            self.df['DATE'] < train_end_date)].set_index(['SEDOL', 'SEDOL_6', 'DATE'])
        test = self.df[(self.df['DATE'] == train_end_date)
                       ].set_index(['SEDOL', 'SEDOL_6', 'DATE'])
        self.df.set_index(['SEDOL', 'SEDOL_6', 'DATE'], inplace=True)
        return train, test

    def get_set_best_cols(self, train):
        fs = SelectKBest(score_func=f_regression, k=10)
        fs.fit(train.drop(columns=[self.pred_var]), train[self.pred_var])
        self.best_cols = fs.get_support(indices=True)
        return self.best_cols

    def do_regression(self, train):
        train_best = train.iloc[:, self.best_cols]
        if self.model == 'lr':
            self.lr = sm.OLS(train[self.pred_var], train_best).fit()
            return self.lr
        elif self.model == 'rf':
            self.rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            self.rf = self.rf.fit(train_best, train[self.pred_var])
            return self.rf
        elif self.model == 'svr':
            self.svr = SVR()
            self.svr = self.svr.fit(train_best, train[self.pred_var])
            return self.svr

    def get_regression_metrics(self, test):
        try:
            self.best_cols
        except NameError:
            print(
                'you\'ll need to run do_regression before you can get the regression metrics')
        else:
            pred = self.lr.predict(test.iloc[:, self.best_cols])
            print('The mean absolute error is ',
                  metrics.mean_absolute_error(test[self.pred_var], pred))
            print('The mean squared error is ',
                  metrics.mean_squared_error(test[self.pred_var], pred))
            print('The r2 score is ', metrics.r2_score(
                test[self.pred_var], pred))

    def get_fit_summary(self, train):
        try:
            self.best_cols
        except NameError:
            print('you\'ll need to run do_regression before you can get the fit summary')
        else:
            x2 = sm.add_constant(train.iloc[:, self.best_cols])
            est = sm.OLS(train[self.pred_var], x2).fit()
            print(est.summary())

    def predict_model(self, val):
        if self.model == 'lr':
            val['PREDICTED'] = self.lr.predict(val.iloc[:, self.best_cols])
        elif self.model == 'rf':
            val['PREDICTED'] = self.rf.predict(val.iloc[:, self.best_cols])
        elif self.model == 'svr':
            val['PREDICTED'] = self.svr.predict(val.iloc[:, self.best_cols])
        return val

    def get_predicted_rets(self, start_date='2003-10-01', end_date='2019-01-01'):
        models = {}
        h = 70
        k = 4

        dates = self.df.reset_index()[['DATE']]
        start_date = datetime.strptime(start_date, '%Y-%m-%d') 
        end_date = datetime.strptime(end_date, '%Y-%m-%d') 
        max_date = dates.max()[0]
        end_date = max_date if end_date >  max_date else end_date
        dates = dates[(dates['DATE'] > start_date) & ( dates['DATE'] < end_date )]['DATE'].sort_values().unique()
        predicted_df = pd.DataFrame()

        for d in dates:
            d = np.datetime_as_string(d, unit='D')
            train, test = self.split_train_and_test(train_date=d, n_months=12)
            n = len(test.reset_index()['SEDOL_6'].unique())

            test_date = str(np.datetime_as_string(
                test.reset_index()['DATE'].unique()[0], unit='D'))

            mdl = self.do_regression(train)
            models[test_date] = {'model': mdl}

            predicted = self.predict_model(test)
            predicted['PREDICTED_PERCENTILE'] = pd.qcut(
                predicted['PREDICTED'], q=100, labels=False) + 1
            predicted_filtered = predicted.loc[predicted['PREDICTED_PERCENTILE'] > h, [
                'PREDICTED', self.pred_var, 'PREDICTED_PERCENTILE']]

            sedols_with_weights = predicted_filtered.reset_index()[
                'SEDOL_6'].unique()
            models[test_date]['sedols'] = set(sedols_with_weights)

            predicted_df = pd.concat([predicted_df, predicted_filtered])

        return models, predicted_df
