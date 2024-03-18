import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from datetime import date
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost
from xgboost import XGBRegressor

import warnings
from tqdm import tqdm

class Cnod:

    def __init__(self, quotes, n_trees, dim, influ_factors_count, start_date, end_year):

        self.quotes = quotes
        self.start_date = start_date
        self.end_year = end_year
        self.progress_bar = None
        self.n_trees = n_trees
        self.dim = dim
        self.influ_factors_count = influ_factors_count
        self.execution_time = []
        pd.options.display.max_columns = None

    def rfm_segmentation(self, todays_date, data):
        '''
        Derive customer rfm segmentation details

        '''

        data2 = data.copy(deep=True)
        data2.OrderDate = pd.to_datetime(
            data2.OrderDate, format='%Y/%m/%d')
        data2['AccountNo2'] = data2.AccountNo

        # Obtain RFM for each customer
        rfm = data2.groupby('AccountNo').agg(
            {
                'OrderDate': lambda day: (todays_date - day.max()).days,
                'AccountNo2': lambda num: len(num),
                'Margin': lambda price: price.sum()
            }
        ).reset_index()

        rfm_col_list = ['AccountNo',
                        'Recency', 'Frequency', 'Monetary']
        rfm.columns = rfm_col_list
        rfm = rfm.fillna(0)

        try:
            # Calculate RFM scores
            rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(
                method='first'), 5, labels=[5, 4, 3, 2, 1],)
            rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(
                method='first'), 5, labels=[1, 2, 3, 4, 5])
            rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(
                method='first'), 5, labels=[1, 2, 3, 4, 5])

            # label customer segments
            seg_map = {
                r'[1-2][1-2]': 'Hibernating',
                r'[1-2][3-4]': 'At Risk',
                r'[1-2]5': 'Can\'t Lose',
                r'3[1-2]': 'About to Sleep',
                r'33': 'Need Attention',
                r'[3-4][4-5]': 'Loyal Customer',
                r'41': 'Promising',
                r'51': 'New Customer',
                r'[4-5][2-3]': 'Potential Loyalist',
                r'5[4-5]': 'Champion'
            }

            rfm['Segment'] = rfm['R_Score'].astype(
                str) + rfm['F_Score'].astype(str)
            rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)

            rfm[['R_Score', 'F_Score', 'M_Score']] = rfm[[
                'R_Score', 'F_Score', 'M_Score']].astype('int')

            del data2
        except:
            rfm['Segment'] = 'Unknown'
            rfm[['R_Score', 'F_Score', 'M_Score']] = [0, 0, 0]

        return rfm

    def plot_feature_importance(self, model, feature_names):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)
        fig = plt.figure(figsize=(20, 20))
        plt.barh(range(len(importance)), importance[sorted_idx], color='b')
        plt.yticks(range(len(importance)), np.array(feature_names)[sorted_idx])
        plt.title('Feature Importance')
        plt.show()

    def get_top_x(self, row):
        top_x = sorted(zip(row.values, row.index), reverse=True)[
            :self.influ_factors_count]
        return [item[1] for item in top_x], [item[0] for item in top_x]

    def calculate_cv(self, df, groupby_col):
        cv_df = df[df.Days_Next_Order != 0].groupby(groupby_col, dropna=False).agg(
            Mean=('Days_Next_Order', 'mean'), SD=('Days_Next_Order', 'std'),).reset_index()
        cv_df[groupby_col + '_CV'] = cv_df.SD/cv_df.Mean
        cv_df[groupby_col + '_CV'] = cv_df[groupby_col + '_CV']
        return cv_df[[groupby_col, groupby_col + '_CV']].fillna(0)

    def calculate_micv(self, df, groupby_col):
        inter_cycle_summary = df[df.Days_Next_Order > 0].groupby([groupby_col, 'OrderDate', 'Days_Next_Order'], dropna=False)[
            "Days_Next_Order"].count().reset_index(name='Count')
        inter_cycle_summary['InterCycleVariance'] = inter_cycle_summary.groupby(
            [groupby_col, ], dropna=False)["Days_Next_Order"].diff(periods=-1)
        inter_cycle_summary.InterCycleVariance = inter_cycle_summary.InterCycleVariance.abs()
        inter_cycle_summary.InterCycleVariance = inter_cycle_summary.InterCycleVariance.fillna(
            0)
        inter_cycle_summary = inter_cycle_summary[[
            groupby_col, 'OrderDate', 'Days_Next_Order', 'InterCycleVariance']]
        micv = inter_cycle_summary.groupby([groupby_col], dropna=False).agg(Sum_DNO=(
            'InterCycleVariance', 'sum'), Count=('InterCycleVariance', 'count')).reset_index()
        micv = micv[micv.Count >= 3]  # only customers with at least 2 cycles
        micv[groupby_col+'_MICV'] = micv.Sum_DNO/(micv.Count - 1)
        return micv[[groupby_col, groupby_col+'_MICV']]

    def cyclic_trans(self, data, col_names):
        for col_name in col_names:
            data[col_name + '_sin'] = np.sin(2 * np.pi *
                                             data[col_name] / data[col_name].max())
            data[col_name + '_cos'] = np.cos(2 * np.pi *
                                             data[col_name] / data[col_name].max())
            data = data.drop(columns=col_name)

        return data

    def cap_outliers(self, data, col):

        first_quartile = data[col].quantile(0.25)
        third_quartile = data[col].quantile(0.75)

        iqr = third_quartile - first_quartile
        iqr_1_5 = 1.5*iqr

        print(first_quartile, third_quartile, iqr_1_5)

        if first_quartile < iqr_1_5:
            lb = first_quartile  # 0
        else:
            lb = first_quartile - iqr_1_5

        ub = third_quartile + iqr_1_5

        print(lb, ub)

        data[col] = np.where(data[col] < ub, data[col], ub)
        data[col] = np.where((data[col] == 0) | (
            data[col] > lb), data[col], lb)
        return data

    def extract_int_values(self, value):

        try:
            numeric_value = int(value)
            return numeric_value
        except (ValueError, TypeError):
            return -1

    def get_lat_lon(self, postal_code):
        try:
            county_details = self.country.query_postal_code(postal_code)
            return county_details['latitude'], county_details['longitude']

        except Exception as e:
            return 0, 0

    def calculate_reaction_days(self, quotes_df, CampaignFlag_df):

        # Transaction occurred during a CampaignFlag
        quote_CampaignFlag = quotes_df[['AccountNo', 'OrderDate']].merge(CampaignFlag_df, left_on='AccountNo',
                                                                                right_on='Master_Account_No', how='left')[['AccountNo', 'OrderDate', 'Start_Date', 'End_Date']]

        quote_during_CampaignFlag = quote_CampaignFlag[(quote_CampaignFlag['Start_Date'] <= quote_CampaignFlag['OrderDate']) &
                                               (quote_CampaignFlag['OrderDate'] <= quote_CampaignFlag['End_Date'])].drop_duplicates()

        quote_during_CampaignFlag['Reaction_Days_During_CampaignFlag'] = (
            quote_during_CampaignFlag['OrderDate'] - quote_during_CampaignFlag['Start_Date']).dt.days

        quote_during_CampaignFlag = quote_during_CampaignFlag.groupby(['AccountNo', 'OrderDate']).agg({
            'Reaction_Days_During_CampaignFlag': 'min'}).reset_index()

        # Transaction occurred after a CampaignFlag
        quote_after_CampaignFlag = quote_CampaignFlag[quote_CampaignFlag['OrderDate']
                                              > quote_CampaignFlag['End_Date']].drop_duplicates()
        quote_after_CampaignFlag['Reaction_Days_After_CampaignFlag'] = (
            quote_after_CampaignFlag['OrderDate'] - quote_after_CampaignFlag['End_Date']).dt.days
        quote_after_CampaignFlag = quote_after_CampaignFlag.groupby(['AccountNo', 'OrderDate']).agg({
            'Reaction_Days_After_CampaignFlag': 'min'}).reset_index()

        # Merge the two DataFrames to quotes_df
        quotes_df = quotes_df.merge(quote_during_CampaignFlag, on=[
                                    'AccountNo', 'OrderDate'], how='left')
        quotes_df = quotes_df.merge(quote_after_CampaignFlag, on=[
                                    'AccountNo', 'OrderDate'], how='left')
        quotes_df[['Reaction_Days_During_CampaignFlag', 'Reaction_Days_After_CampaignFlag']] = quotes_df[[
            'Reaction_Days_During_CampaignFlag', 'Reaction_Days_After_CampaignFlag']].fillna(0)
        return quotes_df

    def convert_df_to_float(self, df, rf):
        # Convert all numeric columns to float
        for col in df[rf].columns:
            df[col] = df[col].astype('float64')

        return df

    def sim_predict_next_orders(self, data, model, qty_price_assumption, adj_days=7, number_cust=300):
        """
        Iteratively predicts next order dates for each customer within the evaluation period.
        """
        test_data_actual = data.copy()
        evaluation_start = test_data_actual["OrderDate"].min()
        evaluation_end = test_data_actual["OrderDate"].max()
        projections = []
        loop_ctrler =0
        
        with warnings.catch_warnings():
            #warnings.simplefilter("ignore")
            for customer_id in tqdm(test_data_actual["AccountNo"].unique(), desc="Predicting next orders" & customer_id):
                customer_data = test_data_actual.loc[test_data_actual["AccountNo"] == customer_id]
                order_date = evaluation_start #customer_data["OrderDate"].iloc[0]  # Initial order date
                customer_projection = pd.DataFrame()
                print('customer: ',customer_id)
                while order_date < evaluation_end:
                    # Predict next order date
                    features = customer_data[["Quantity", "total_price", 'CreditLimit', 'AccountNo_CV', 'MarketSector_CV', 
                                              'ProductCode_CV', 'CustRegion_CV', 'AccountNo_MICV', 'MarketSector_MICV', 'Recency', 
                                              'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score']].mean(axis=0).to_frame().T
                    predicted_days = int(model.predict(features)[0]) + adj_days
                    predicted_date = order_date + timedelta(days=predicted_days)

                    # Stop if predicted date exceeds evaluation period
                    if (predicted_date > evaluation_end):
                        break

                    # Select average or min for quantity and price from customer history as sim assumption
                    if qty_price_assumption=='Min':
                        sim_qty = customer_data["Quantity"].min()
                        sim_price = customer_data["total_price"].min()
                    else: 
                        sim_qty = customer_data["Quantity"].mean()
                        sim_price = customer_data["total_price"].mean()                        
                        
                    # Update customer data for next iteration
                    new_order = {
                        "AccountNo": customer_id,
                        "Segment": customer_data.Segment.unique()[0],
                        #"CustSubRegion": customer_data.CustSubRegion.unique()[0],
                        "sim_OrderDate": predicted_date.strftime("%Y-%m-%d"),
                        "sim_Quantity": sim_qty, #customer_data["Quantity"].min(), 
                        "sim_Price": sim_price, #customer_data["UnitPrice"].min(),
                      }
                    customer_projection = customer_projection._append(new_order, ignore_index=True)
                    order_date = predicted_date
                
                loop_ctrler = loop_ctrler+1                
                if loop_ctrler > number_cust:
                    print('Limited number of customers to: ', loop_ctrler)
                    break
                projections.extend(customer_projection.to_dict(orient="records"))

            return pd.DataFrame(projections)

    def data_prep(self,):

        start_time = time.time()

        # General transformation
        print('Initial data shape: ', self.quotes.shape)
        print('***Initial number of customers: ',
              self.quotes.AccountNo.nunique())

        # get customer segmentation details
        cust_seg = self.rfm_segmentation(pd.Timestamp.today(), self.quotes)
        
        # Select initial features
        self.quotes = self.quotes[['AccountNo', 'OrderDate', 'ShopCode', 'Depot', 'ProductCode', 'Quantity', 'CostPrice', 'UnitPrice',
                                   'total_price','Margin', 'ReasonForContact', 'Converted', 
                                   'CreditLimit', 'MarketSector', 'CustLatitude', 'CustLongitude', 'CustPostCode', 'CustRegion', 'CustSubRegion',]] #
        print('Data shape after - Selecting initial features: ', self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())
        
        # Format date and scope data by date
        self.quotes.OrderDate = pd.to_datetime(
            self.quotes.OrderDate, format='%Y/%m/%d')
        self.quotes = self.quotes[(self.quotes.OrderDate >= self.start_date) & (
            self.quotes.OrderDate.dt.year <= self.end_year)]
        print('Data shape after - formatting date and scoping data by quote date: ',
              self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # Get unique product(s) bought by each customer on the same day and populate main df
        pdt_cust_date_summary = self.quotes.copy(deep=True)
        pdt_cust_date_summary = pdt_cust_date_summary.groupby(['AccountNo', 'OrderDate'])[
            'ProductCode'].agg(lambda x: ', '.join(set(x))).reset_index(name='Products')
        self.quotes = self.quotes.merge(pdt_cust_date_summary, on=[
                                        'AccountNo', 'OrderDate'], how='left')
        print('Data shape after - Populate main df with same day and customer product quoted: ', self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # Get quantity, cost and price values for only converted quotes
        self.quotes['converted_Quantity'] = np.where(
            self.quotes.Converted == 'Y', self.quotes.Quantity, 0)
        self.quotes['converted_CostPrice'] = np.where(
            self.quotes.Converted == 'Y', self.quotes.CostPrice, 0)
        self.quotes['converted_UnitPrice'] = np.where(
            self.quotes.Converted == 'Y', self.quotes.UnitPrice, 0)
        self.quotes['converted_total_price'] = np.where(
            self.quotes.Converted == 'Y', self.quotes.total_price, 0)
        self.quotes['converted_Margin'] = np.where(
            self.quotes.Converted == 'Y', self.quotes.Margin, 0)

        # Specify categorical features #1
        self.categorical_columns = [
            'ReasonForContact', 'Converted', ]
        self.quotes[self.categorical_columns] = self.quotes[self.categorical_columns].astype(
            'category')
        print('Data shape after - decomposing date and specifying cat features: ',
              self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # Encode categorical columns
        encoded_columns_1 = pd.get_dummies(
            self.quotes[self.categorical_columns], prefix=self.categorical_columns)
        self.quotes = pd.concat(
            [self.quotes.drop(self.categorical_columns, axis=1), encoded_columns_1], axis=1)
        print('Data shape after - one hot encoding of cat features: ',
              self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # summarise by customer details and quote date
        self.quotes['Same_day_activity_count'] = 1
        columns_to_calculate_sum = ['Same_day_activity_count', 'Quantity', 'CostPrice', 'UnitPrice', 'total_price', 'Margin',
                                    'converted_Quantity', 'converted_CostPrice', 'converted_UnitPrice', 'converted_total_price', 'converted_Margin']
        ohe_columns_to_calculate_sum = encoded_columns_1.columns.to_list()
        columns_to_calculate_last = ['CustPostCode', 'CustLatitude','CustLongitude', 'CustSubRegion']

        aggregation_functions = {
            col: np.sum for col in columns_to_calculate_sum}
        aggregation_functions.update(
            {col: np.sum for col in ohe_columns_to_calculate_sum})
        aggregation_functions.update(
            {col: lambda x: x.iloc[-1] for col in columns_to_calculate_last})

        self.quotes = self.quotes.groupby(['AccountNo', 'OrderDate', 'ShopCode', 'Depot', 'CreditLimit',
                                           'MarketSector', 'Products', 'ProductCode', 'CustRegion',], dropna=False).agg(aggregation_functions).reset_index() #

        print('Data shape after - aggregation : ', self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # avoid negative, zero and invalid quote quantity. Advice from CE
        self.quotes = self.quotes[self.quotes.Quantity > 2]
        print('Data shape after - Include only data where Quote_Volume > 2: ',
              self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # sort and remove duplicates
        self.quotes.sort_values(
            by=['AccountNo', 'OrderDate'], ascending=True)
        self.quotes.drop_duplicates(
            subset=['AccountNo', 'OrderDate'], keep='last', inplace=True,)
        print('Data shape after - Drop duplicates based on AccountNo, OrderDate: ', self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # get date difference (in days) by customer and quote date
        date_diff_summary = self.quotes.copy(deep=True)
        date_diff_summary = date_diff_summary.groupby(['AccountNo', 'OrderDate'])[
            "OrderDate"].count().reset_index(name='Count')
        date_diff_summary['Days_Next_Order'] = date_diff_summary.groupby(
            ['AccountNo', ])["OrderDate"].diff(periods=-1)
        date_diff_summary.Days_Next_Order = date_diff_summary.Days_Next_Order.dt.days.abs()
        date_diff_summary.Days_Next_Order = date_diff_summary.Days_Next_Order.fillna(
            0)
        date_diff_summary = date_diff_summary[[
            'AccountNo', 'OrderDate', 'Days_Next_Order']]

        # populate main df with the quote date difference (Actual days before the next order date)
        self.quotes = self.quotes.merge(
            date_diff_summary, on=['AccountNo', 'OrderDate'], how='left')
        print('Data shape after - Populate main df with the quote date difference: ',
              self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # get CV values
        col_list_cv_cal = ['AccountNo', 
                            'MarketSector', 'ProductCode', 'CustRegion',] #

        for col in col_list_cv_cal:
            cv_df = self.calculate_cv(self.quotes, col)
            self.quotes = self.quotes.merge(cv_df, on=col, how='left')

        print('Data shape after - Populate main df with CV: ', self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # get MICV values
        col_list_cv_cal = ['AccountNo', 'MarketSector', ]

        for col in col_list_cv_cal:
            cv_df = self.calculate_micv(self.quotes, col)
            self.quotes = self.quotes.merge(cv_df, on=col, how='left')

        self.quotes = self.quotes.drop(columns=['ProductCode'])
        print('Data shape after - Populate main df with MICV: ', self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # Handle NaNs (#2)
        self.quotes[['AccountNo_CV',  'CustRegion_CV', 'MarketSector_CV', 'ProductCode_CV', 'AccountNo_MICV',
                     'MarketSector_MICV', ]] = self.quotes[['AccountNo_CV', 'MarketSector_CV', 'ProductCode_CV', 'CustRegion_CV',
                                                                   'AccountNo_MICV', 'MarketSector_MICV', ]].fillna(0.0)

        # decompose date
        self.quotes['day_of_week'] = self.quotes.OrderDate.dt.dayofweek
        self.quotes['day'] = self.quotes.OrderDate.dt.day
        self.quotes['week_number'] = self.quotes.OrderDate.dt.isocalendar(
        ).week.astype('int64')
        self.quotes['month'] = self.quotes.OrderDate.dt.month
        self.quotes['qtr'] = self.quotes.OrderDate.dt.quarter
        self.quotes["is_weekend"] = (
            self.quotes.OrderDate.dt.dayofweek > 4).astype('int64')

        # Get season
        seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
        month_to_season = dict(zip(range(1, 13), seasons))
        self.quotes['season'] = self.quotes.OrderDate.dt.month.map(
            month_to_season)
        print('Data shape after - Adding UK seasons: ', self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # Handle Days_Next_Order outliers
        self.quotes = self.cap_outliers(self.quotes, 'Days_Next_Order')
        print(self.quotes[['Days_Next_Order']].boxplot())
        print('Data shape after - Handling outliers: ', self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # Add customer segmentation information to data
        self.quotes = self.quotes.merge(cust_seg, on=['AccountNo'])
        print('Data shape after - Adding RFM information: ', self.quotes.shape)
        print('***Number of customers after data preparation: ',
              self.quotes.AccountNo.nunique())

        # Add customer central tendencies stats
        ct_stats = self.quotes[self.quotes.Days_Next_Order > 0].groupby(['AccountNo', ], dropna=False).agg(Days_Next_Order_Min=('Days_Next_Order', 'min'),
                                                                                                                   Days_Next_Order_Max=(
            'Days_Next_Order', 'max'),
            Days_Next_Order_Median=(
            'Days_Next_Order', 'median'),
            Days_Next_Order_Mean=(
            'Days_Next_Order', 'mean'),
        ).reset_index()
        self.quotes = self.quotes.merge(
            ct_stats, on='AccountNo', how='left')
        self.quotes[['Days_Next_Order_Min', 'Days_Next_Order_Max', 'Days_Next_Order_Median', 'Days_Next_Order_Mean']] = self.quotes[['Days_Next_Order_Min', 'Days_Next_Order_Max',
                                                                                                                                     'Days_Next_Order_Median', 'Days_Next_Order_Mean']].fillna(0.0)
        print('Data shape after - Populate main df with customer central tendencies stats: ', self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # Specify categorical features #2
        self.categorical_columns_2 = ['ShopCode', 
                                      'MarketSector', 'CustRegion',] #
        self.quotes[self.categorical_columns_2] = self.quotes[self.categorical_columns_2].astype(
            'category')
        print('Data shape after - decomposing date and specifying cat features: ',
              self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # Encode categorical columns
        encoded_columns_2 = pd.get_dummies(
            self.quotes[self.categorical_columns_2], prefix=self.categorical_columns_2)
        self.quotes = pd.concat([self.quotes.drop(
            self.categorical_columns_2, axis=1), encoded_columns_2], axis=1)

        # Encode time based features
        self.quotes = self.cyclic_trans(
            self.quotes, ['day_of_week', 'week_number', 'qtr', 'month', 'season', ])
        print('Data shape after - Transformation of some date attributes: ',
              self.quotes.shape)
        print('***Current number of customers: ',
              self.quotes.AccountNo.nunique())

        # Assign feature roles
        self.special_features = ['AccountNo', 'Segment',
                                 'OrderDate', 'Products', 'Depot', 'CustPostCode', 'CustSubRegion'] #'CustLatitude','CustLongitude',
        self.label = ['Days_Next_Order']
        self.regular_features = self.quotes.drop(columns=list(
            set(self.special_features + self.label))).columns.to_list()

        # Seperate Horizon data from training and validation data
        self.horizon = self.quotes[self.quotes.Days_Next_Order == 0]
        self.training_and_validation = self.quotes[self.quotes.Days_Next_Order != 0]

        # Split features and labels
        X = self.training_and_validation[self.regular_features]
        y = self.training_and_validation[self.label]

        # Selected best performing features

            # Create a feature selection transformer
        original_feature_names = X.columns.to_list()

        if len(original_feature_names) > self.dim:
            pass
        else:
            self.dim = len(original_feature_names)

        self.feature_selector = SelectKBest(
            score_func=f_regression, k=self.dim)
        self.feature_selector.fit(X, y)

            # Get the indices of the selected features
        selected_indices = self.feature_selector.get_support(indices=True)

            # Get the names of the selected features
        self.selected_features = [original_feature_names[i]
                                  for i in selected_indices]

        X = X[self.selected_features]

        # Handle NaNs (#3)
        X = X.fillna(0)

        print('Horizon df size: ', self.horizon.shape)
        print('Training and Validation feature size: ', X.shape)
        print('Training and Validation label size: ', y.shape)
        print("Selected Features:", self.selected_features)

        self.quotes = pd.DataFrame()
        self.customers = pd.DataFrame()
        self.postcode_latlong = pd.DataFrame()

        self.execution_time.append(
            'Data Preparation: ' + str((time.time() - start_time)/60))

        return X, y

    def model_prep(self, X, y,):

        start_time = time.time()

        self.model = XGBRegressor(subsample=0.6, n_estimators=100, max_depth=15, # estimators =300, 100
                                  learning_rate=0.05, colsample_bytree=0.5, colsample_bylevel=0.4,  seed=42)

        # Perform cross-validation
        #scores = cross_val_score(
            #self.model, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=3)

        # Convert the negative mean squared error scores to positive RMSE scores
        #rmse_scores = np.sqrt(-scores)
        #mean_rmse = np.mean(rmse_scores)

        # Print the cross-validation scores
        #print("Cross-Validation RMSE Scores:", rmse_scores)
        #print("Mean RMSE Score:", np.mean(mean_rmse))

        # Train production model
        self.model.fit(X, y.values.ravel())
        # self.plot_feature_importance(self.model, self.selected_features)

        self.execution_time.append(
            'Model Preparation: ' + str((time.time() - start_time)/60))

        #return mean_rmse

    def predict(self,):

        start_time = time.time()

        # Split features from labels.
        X = self.horizon[self.selected_features]
        y = self.horizon[self.label]

        # Convert the new data to a DMatrix
        new_data_dmatrix = xgboost.DMatrix(X)

        # Estimator
        pred = self.model.predict(X)
        feature_contributions = self.model.get_booster().predict(
            new_data_dmatrix, pred_contribs=True) #, ntree_limit=self.n_trees)

        # Results
        self.horizon['Predicted_Days_Later'] = pred
        self.horizon['Predicted_Days_Later'] = self.horizon['Predicted_Days_Later'].apply(
            lambda x: round(x, 0))
        self.horizon['Predicted_Days_Later'] = self.horizon['Predicted_Days_Later'].apply(
            lambda x: 1 if x <= 0 else x)

        # Get associated dates
        self.horizon['Predicted_Next_Quote_Date'] = self.horizon.apply(
            lambda x: x.OrderDate + relativedelta(days=x.Predicted_Days_Later), axis=1)
        print('Customers next order dates have been extracted.')

        predictions = self.horizon.loc[:, ['AccountNo', 'Segment',
                                           'Depot', 'Products', 'Quantity', 'total_price', 'OrderDate', 'Predicted_Next_Quote_Date', 'CustPostCode','CustLatitude','CustLongitude', 'CustSubRegion' ]]

        # Add the feature contributions to the estimation results
        #self.progress_bar = tqdm.tqdm(
            #enumerate(self.selected_features), leave=False)
        for i, feature in enumerate(self.selected_features):
            predictions[f"{feature}_Contribution"] = feature_contributions[:, i]
            #self.progress_bar.update()
            #self.progress_bar.refresh()

        # Remove derived columns not required in output
        col_to_delete = ['Days_Next_Order_Min_Contribution', 'Days_Next_Order_Max_Contribution',
                         'Days_Next_Order_Median_Contribution', 'Days_Next_Order_Mean_Contribution', 'Estimate_1']  
        col_found = [
            col for col in col_to_delete if col in predictions.columns]
        predictions.drop(columns=col_found, inplace=True)

        # Get the top x highest contributing factors and their corresponding values
        all_influ_factors_item_list = [
            col for col in predictions.columns if col.endswith('Contribution')]
        predictions[['top_columns', 'top_values']] = predictions[all_influ_factors_item_list].apply(
            self.get_top_x, axis=1, result_type='expand')
        top_influ_factors_item_list = [
            'Factor_item_' + str(i+1) for i in range(self.influ_factors_count)]
        top_influ_factors_value_list = [
            'Factor_value_' + str(i+1) for i in range(self.influ_factors_count)]
        predictions[top_influ_factors_item_list] = pd.DataFrame(
            predictions['top_columns'].tolist(), index=predictions.index)
        predictions[top_influ_factors_value_list] = pd.DataFrame(
            predictions['top_values'].tolist(), index=predictions.index)
        predictions.drop(
            columns=['top_columns', 'top_values']+all_influ_factors_item_list, inplace=True)
        
        print('Customer next order dates are ready')
        self.execution_time.append(
            'Customer Order Date Estimation: ' + str((time.time() - start_time)/60))

        for timings in self.execution_time:
            print(timings)

        return predictions
