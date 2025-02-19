import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from pprint import pprint, pformat
import copy
import zipfile

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, f1_score

def eda_df(df):
    """Провести EDA для датафрейма"""
    df_describe = df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    # посчитать долю пропусков
    df_describe.loc["%nan"] = (np.round(df[df_describe.columns].isna().mean()*100, 2)).to_list()
    # посчитать дисперсию
    columns_var = []
    for column in df_describe.columns:
        columns_var.append(df[column].var())
    df_describe.loc['var'] = columns_var
    return df_describe


def show_boxes(df, columns, ncols = 3, type="box", row_height=500, total_width=1200):
    """Показать 'ящики_с_усами' для набора df.
    Ящики будут показаны для столбцов датафрема, перечисленных в columns.
    Графики будут показаны в несколько столбцов, количество которых задается в параметре ncols."""
    nrows = int(round((len(columns) + 0.5) / ncols, 0))
    nrows = nrows if nrows > 1 else 1

    if type == "box":
        title = "Ящики с усами"
    elif type == "hist":
        title = "Гистрограммы"
    else:
        raise f"Не реализована обработка типа графика {type}"


    fig = make_subplots(rows=nrows, cols=ncols)
    fig.update_layout(
        title_x=0.5,
        title_text=title,
        height=row_height*nrows, 
        width=total_width
    )
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if type == "box":
                fig.add_box(y=df[columns[i]], name=columns[i], row=r+1, col=c+1)
            elif type == "hist":
                fig.add_histogram(x=df[columns[i]], name=columns[i], row=r+1, col=c+1)
            else:
                raise f"Не реализована обработка типа графика {type}"
            i += 1
            if i >= len(columns):
                break
        if i >= len(columns):
            break
    fig.show()          

def show_boxes_plt(df, columns, ncols = 3, type="box", row_height=500, total_width=1200):
    """Показать 'ящики_с_усами' для набора df.
    Ящики будут показаны для столбцов датафрема, перечисленных в columns.
    Графики будут показаны в несколько столбцов, количество которых задается в параметре ncols."""
    nrows = int(round((len(columns) + 0.5) / ncols, 0))
    nrows = nrows if nrows > 1 else 1

    if type == "box":
        title = "Ящики с усами"
    elif type == "hist":
        title = "Гистрограммы"
    else:
        raise f"Не реализована обработка типа графика {type}"
            
    plt.figure(figsize=(ncols * 5, nrows * 3))
    
    for i, column in enumerate(columns, start=1):
        plt.subplot(nrows, ncols, i)
        if type == "box":
            sns.boxplot(df[column])
        elif type == "hist":
            sns.histplot(df[column], kde=True)
        else:
            raise f"Не реализована обработка типа графика {type}"
        # Добавить название столбца как заголовок графика
        plt.title(column)
    plt.tight_layout()
    plt.show()
        
def iqr_values(values):
    Q3 = np.quantile(values, 0.75, axis=0)
    Q1 = np.quantile(values, 0.25, axis=0)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    #print(Q1, Q3, IQR, lower, upper)
    return Q1, Q3, IQR, lower, upper

def remove_columns(column, params): 
    """функция для удаления столбцов"""
    if column in params["columns_X"]:
        params["columns_X"].remove(column)
    if column in params["columns_with_nan"]:
        params["columns_with_nan"].remove(column)
    if column in params["num_columns"]:
        params["num_columns"].remove(column)
    if column in params["cat_columns"]:
        params["cat_columns"].remove(column)
        
def apply_OneHotEncoder(df, columns, columns_X):
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns))
    new_df = pd.concat([df, one_hot_df], axis=1)
    new_columns_X = copy.deepcopy(columns_X)
    new_columns_X.extend(encoder.get_feature_names_out(columns).tolist())
    for col in columns:
        if col in columns_X:
            new_columns_X.remove(col)
    return new_df, new_columns_X

def apply_OrdinalEncoder(df, columns, columns_X):
    #TODO: apply_OneHotEncoder и apply_OrdinalEncoder по разному работают с передаваемыми объектами
    columns_cats = list(columns.values())
    columns_list = list(columns.keys())
    columns_ordered = [f'{c}_ordered' for c in columns_list]
    encoder = OrdinalEncoder(categories = columns_cats)
    df[columns_ordered] = encoder.fit_transform(df[columns_list])  
    for col in columns_list:
        if col in columns_X:
            columns_X.remove(col)
    columns_X += columns_ordered
    return df, columns_X


def check_model(model, X_train, y_train, X_test, y_test):
    """Оценить качество модели"""
    # добавить f1_score и roc_auc_score
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)    
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)    
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    
    print('Train data:')
    print(f"  MSE:    {round(mse_train,4)}")
    print(f"  RMSE:   {round(rmse_train,4)}")
    print(f"  MAE:    {round(mae_train,4)}")
    print(f"  r2:     {round(r2_train,4)}")
    print(f"  median: {round(y_train.median(),4)}")

    print('Test data:')
    print(f"  MSE:    {round(mse_test,4)}")
    print(f"  RMSE:   {round(rmse_test,4)}")
    print(f"  MAE:    {round(mae_test,4)}")
    print(f"  r2:     {round(r2_test,4)}")    
    print(f"  median: {round(y_test.median(),4)}")