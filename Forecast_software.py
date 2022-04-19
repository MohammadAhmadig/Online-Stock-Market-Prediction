
# coding: utf-8

# In[ ]:
# ahmadi.mohammad71@gmail.com

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import lightgbm
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error
import glob
import functools


def LightGBM():
    
    #lightgbm regression
    row_count2 = x_train.shape[0]
    split_point2 = int(row_count2*test_ratio)
    X, X_test = x_train[:split_point2], x_train[split_point2:]
    Y, Y_test = y_train[:split_point2], y_train[split_point2:]
    train_data = lightgbm.Dataset(X, label=Y)
    test_data = lightgbm.Dataset(X_test, label=Y_test)
    parameters = {'objective': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'learning_rate': 0.1,
              'seed' : 42}
    model = lightgbm.train(parameters,
                           train_data,
                           valid_sets=test_data,
                           num_boost_round=5000,
                           early_stopping_rounds=100)
    return model

def LR():
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    return reg

def date(d,cv_type):
    if(cv_type =='last 5%'):
        rcount = d.shape[0]
        splitpoint = int(rcount*test_ratio)
        xtest = d[splitpoint:]
        return xtest.Date
        
    else:#-----
        rcount = d.shape[0]
        splitpoint = int(rcount*test_ratio)
        xtest = d[splitpoint:]
        return xtest.Date
    
def make_sliding(df, N):
    dfs = [df.shift(-i).applymap(lambda x: [x]) for i in range(0, N+1)]
    return functools.reduce(lambda x, y: x.add(y), dfs)

prior_file_dict = dict()
test_ratio = 19/20 # 5 %
while(1):
    
    file_dict = dict()
    # Load data
    extension = 'csv'
    files = glob.glob('*.{}'.format(extension))
    for file in files:# files without .ai name
        if('.ai' in file):
            files.remove(file)

    # files is empty
    if not files:
        print('No csv File seen.')
        break

    targets = ['ZigZag', 'ZigZag Length']
    regression_names = ["LinearRegression","Lightgbm"]


    for j in range(len(files)):
        data = pd.read_csv(files[j], parse_dates=['Date'])
        tempdata = pd.read_csv(files[j])

        #test_date
        test_date = date(tempdata,'last 5%')
        # new features
        data['month'] = data.Date.dt.month
        #data['week'] = data.Date.dt.week
        data['week_day'] = data.Date.dt.weekday
        data['year'] = data.Date.dt.year
        data['days_in_month'] = data.Date.dt.days_in_month
        data = data.drop("Date", axis=1)
        
        
        file_dict[files[j]] = data.shape[0]
        if files[j] not in prior_file_dict:
            
            text = ""
            for target in targets:
                x_data = data.loc[:,data.columns != target]
                y_data = data[target]
                

                # test 5 % last
                row_count = data.shape[0]
                split_point = int(row_count*test_ratio)
                x_train, x_test = x_data[:split_point], x_data[split_point:]
                y_train, y_test = y_data[:split_point], y_data[split_point:]


                regressions = [
                    LR(),
                    LightGBM()]


                i=0
                min_rmse = 10000
                best_teqnique =''
                for regr in regressions:

                    if(target == 'ZigZag' and i==1):
                        break
                    if(target == 'ZigZag'):
                        y_pred = regressions[0].predict(x_test)
                    if(target == 'ZigZag Length'):
                        y_pred = regr.predict(x_test)

                    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
                    if(rmse < min_rmse):
                        min_rmse = rmse
                        best_teqnique = regression_names[i]
                        if(target == 'ZigZag'):
                            zigzag = y_pred
                        elif(target == 'ZigZag Length'):
                            zigzaglength = y_pred

                    percent_error= ((abs(y_test - y_pred) / abs(y_test))*100) 
                    percent_error = percent_error.mean()
                    print("RMSE = " , rmse)
                    text = text + "\nRoot mean squared error of " + regression_names[i] +" for "+ target +" = " + str(rmse) +"\n"
                    text = text +'---------------------------------------------------------'
                    text = text + "\nPercent error of " + regression_names[i] +" for "+ target +" = " + str(percent_error) +"\n"
                    text = text +'---------------------------------------------------------'
                    i=i+1

                # predict next value
                
                window_size = 14
                x = make_sliding(data,window_size)
                x = x.loc[:,data.columns == target]
                x[['c1','c2','c3','c4','c5',
                   'c6','c7','c8','c9','c10',
                   'c11','c12','c13','c14','c15']] = pd.DataFrame(x[target].tolist(), index= x.index)
                x = x.drop(target, axis=1)
                x_test3 = x[x.shape[0]-window_size+1:x.shape[0]-window_size+2]
                x = x[:x.shape[0]-window_size+1]

                x_test2 = x[x.shape[0]-1:].drop("c15", axis=1)
                x_test3 = x_test3.drop("c15", axis=1)
                train2 = x[:x.shape[0]-1]
                y_train = train2.c15
                x_train = train2.drop("c15", axis=1)

                # validation of lightgbm
                row_count2 = x_train.shape[0]
                split_point2 = int(row_count2*test_ratio)
                X, X_test = x_train[:split_point2], x_train[split_point2:]
                Y, Y_test = y_train[:split_point2], y_train[split_point2:]

                if(best_teqnique == regression_names[0]):# Linear regression
                    mdl = LinearRegression()
                    mdl.fit(x_train, y_train)
                else: # Lightgbm
                    train_data = lightgbm.Dataset(X, label=Y)
                    test_data = lightgbm.Dataset(X_test, label=Y_test)
                    parameters = {'objective': 'regression','boosting': 'gbdt','metric': 'rmse','learning_rate': 0.1,'seed' : 42}
                    mdl = lightgbm.train(parameters,train_data,valid_sets=test_data,num_boost_round=5000,early_stopping_rounds=100)
                    #mdl = regressions[1]
                y_pred2 = mdl.predict(x_test2)
                x_test3.c14 = y_pred2[0]
                y_pred3 = mdl.predict(x_test3)
                if(target == 'ZigZag'):
                    next_zigzag2 = y_pred2
                    next_zigzag3 = y_pred3
                text = text + "\n two Next prediction value of " + best_teqnique +" for "+ target +" = "+str(y_pred2[0])+" and "+ str(y_pred3[0])+"\n"
                text = text +'---------------------------------------------------------'

            file = open(files[j][:-4] +'_Log.txt', 'w')
            file.write(text)
            file.close()

            # Create ai file
            ai_file = pd.DataFrame(columns = ['Date', 'Close', 'Open', 'High', 'Low', 'ZigZag', 'ZigZag Length', 'ZigZag Deg']) 

            zigzag = np.append(zigzag, next_zigzag2[0])
            zigzag = np.append(zigzag, next_zigzag3[0])
            zigzaglength = np.append(zigzaglength, y_pred2[0])
            zigzaglength = np.append(zigzaglength, y_pred3[0])
            ai_file.Date = test_date
            ai_file = ai_file.append(pd.Series(), ignore_index=True)
            ai_file = ai_file.append(pd.Series(), ignore_index=True)
            ai_file.ZigZag = zigzag
            zigzaglength = zigzaglength.astype(int)
            ai_file['ZigZag Length'] = zigzaglength
            ai_file.to_csv(files[j][:-4] +'.ai.csv', index=False)
            
        elif( file_dict[files[j]] != prior_file_dict[files[j]] ):
            
            #read ai csv
            ai_predicted = pd.read_csv(files[j][:-4] +'.ai.csv')
            num_of_new = abs(file_dict[files[j]] - prior_file_dict[files[j]])
            ai_predicted = ai_predicted[:-1] # remove second prediction
            zigzag = ai_predicted['ZigZag']
            zigzaglength = ai_predicted['ZigZag Length']
            
            for count in range(num_of_new):
                
                ai_predicted.loc[ai_predicted.shape[0]-1,'Date'] = tempdata.Date[prior_file_dict[files[j]]+count] # add Date
                new_data = data[:data.shape[0]-(num_of_new-(count+1))]
                
                for target in targets:
                    # predict next value
                    
                    window_size = 14
                    x = make_sliding(new_data,window_size)
                    x = x.loc[:,new_data.columns == target]
                    x[['c1','c2','c3','c4','c5',
                       'c6','c7','c8','c9','c10',
                       'c11','c12','c13','c14','c15']] = pd.DataFrame(x[target].tolist(), index= x.index)
                    x = x.drop(target, axis=1)
                    x_test3 = x[x.shape[0]-window_size+1:x.shape[0]-window_size+2]
                    x = x[:x.shape[0]-window_size+1]

                    x_test2 = x[x.shape[0]-1:].drop("c15", axis=1)
                    x_test3 = x_test3.drop("c15", axis=1)
                    train2 = x[:x.shape[0]-1]
                    y_train = train2.c15
                    x_train = train2.drop("c15", axis=1)

                    # validation of lightgbm
                    row_count2 = x_train.shape[0]
                    split_point2 = int(row_count2*test_ratio)
                    X, X_test = x_train[:split_point2], x_train[split_point2:]
                    Y, Y_test = y_train[:split_point2], y_train[split_point2:]

                    if(target == 'ZigZag'):
                        mdl = LinearRegression()
                        mdl.fit(x_train, y_train)
                    else: 
                        train_data = lightgbm.Dataset(X, label=Y)
                        test_data = lightgbm.Dataset(X_test, label=Y_test)
                        parameters = {'objective': 'regression','boosting': 'gbdt','metric': 'rmse','learning_rate': 0.1,'seed' : 42}
                        mdl = lightgbm.train(parameters,train_data,valid_sets=test_data,num_boost_round=5000,early_stopping_rounds=100)
                        #mdl = regressions[1]
                    y_pred2 = mdl.predict(x_test2)
                    x_test3.c14 = y_pred2[0]
                    y_pred3 = mdl.predict(x_test3)
                    if(target == 'ZigZag'):
                        next_zigzag2 = y_pred2
                        next_zigzag3 = y_pred3
                    
                zigzag = np.append(zigzag, next_zigzag2[0])
                zigzaglength = np.append(zigzaglength, y_pred2[0])
                ai_predicted = ai_predicted.append(pd.Series(), ignore_index=True)
                ai_predicted.ZigZag = zigzag
                zigzaglength = zigzaglength.astype(int)
                ai_predicted['ZigZag Length'] = zigzaglength
            
            # --- 

            zigzag = np.append(zigzag, next_zigzag3[0])
            zigzaglength = np.append(zigzaglength, y_pred3[0])
            ai_predicted = ai_predicted.append(pd.Series(), ignore_index=True)
            ai_predicted.ZigZag = zigzag
            zigzaglength = zigzaglength.astype(int)
            ai_predicted['ZigZag Length'] = zigzaglength
            ai_predicted.to_csv(files[j][:-4] +'.ai.csv', index=False)
            print("Result Updated")
            
            
    print("Waiting Time")
    prior_file_dict = file_dict
    time.sleep(120)
    

