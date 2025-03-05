import pandas as pd
import pandas.api.types as pd_types
import numpy as np

import plotly.express as plotly_px
import plotly.graph_objects as plotly_go
import plotly.subplots as plotly_subplt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import (OneHotEncoder, 
                                   OrdinalEncoder,
                                   StandardScaler, 
                                   MinMaxScaler, 
                                   MaxAbsScaler, 
                                   RobustScaler, 
                                   PolynomialFeatures,
                                   QuantileTransformer, 
                                   PowerTransformer
                                  )

import copy
import joblib

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


    fig = plotly_subplt.make_subplots(rows=nrows, cols=ncols)
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

def show_boxes_plt(df, columns_x, ncols = 3, type="box", row_height=500, total_width=1200, column_y=None):
    """Показать 'ящики_с_усами' для набора df.
    Ящики будут показаны для столбцов датафрема, перечисленных в columns.
    Графики будут показаны в несколько столбцов, количество которых задается в параметре ncols."""
    nrows = int(round((len(columns_x) + 0.59) / ncols, 0))
    nrows = nrows if nrows > 1 else 1

    if type == "box":
        title = "Ящики с усами"
    elif type == "hist":
        title = "Гистрограммы"
    else:
        raise f"Не реализована обработка типа графика {type}"
            
    plt.figure(figsize=(ncols * 5, nrows * 3))
    
    for i, column in enumerate(columns_x, start=1):
        plt.subplot(nrows, ncols, i)
        if type == "box":
            if column_y is None:
                sns.boxplot(x=df[column])
            else:
                sns.boxplot(x=df[column], y=df[column_y])
        elif type == "hist":
            sns.histplot(df[column], kde=True)
        else:
            raise f"Не реализована обработка типа графика {type}"
        # Добавить название столбца как заголовок графика
        plt.title(column)
    plt.tight_layout()
    plt.show()
        
def iqr_values(values):
    """границы для ящика-с-усами"""
    Q3 = np.quantile(values, 0.75, axis=0)
    Q1 = np.quantile(values, 0.25, axis=0)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    return Q1, Q3, IQR, lower, upper

    
def nans_percents(df):
    return df.isna().sum()/len(df)*100    

def encode_features(src_df, onehot_cols=None, onehot_drop=None, ordinal_cols=None, columns_X=None):
    df = src_df.copy()  
    new_columns_X = copy.deepcopy(columns_X)
    if onehot_cols:
        encoder = OneHotEncoder(sparse_output=False, drop=onehot_drop)
        one_hot_encoded = encoder.fit_transform(df[onehot_cols])
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(onehot_cols))
        df = pd.concat([df, one_hot_df], axis=1)
        new_columns_X += encoder.get_feature_names_out(onehot_cols).tolist()
        for col in onehot_cols:
            if col in columns_X:
                new_columns_X.remove(col)
        df.drop(onehot_cols, axis=1, inplace=True)
        
    if ordinal_cols and ordinal_cols:
        ordinal_columns_cats = list(ordinal_cols.values())
        ordinal_columns_list = list(ordinal_cols.keys())
        encoder = OrdinalEncoder(categories = ordinal_columns_cats)
        df[ordinal_columns_list] = encoder.fit_transform(df[ordinal_columns_list])  

    return df, new_columns_X

def fill_with_mode(data, group_col, target_col):
    global_mode = data[target_col].mode()[0]
    def fill_group_mode(x):
        group_mode = x.mode()
        if not group_mode.empty:
            return group_mode[0]
        else:
            return global_mode
    data[target_col] = data.groupby(group_col)[target_col].transform(fill_group_mode)
    
def prepare_dataset(dataset_df, params, scaler=None, train_size=0.7):
    """Разделить датасет на тренировочную и тестовую выборки и прогнать через нормализатор, если он указан"""
    X_train, X_test, y_train, y_test = train_test_split(dataset_df[params["columns_X"]], 
                                                        dataset_df[params["target_column"]], 
                                                        train_size=train_size, 
                                                        stratify=dataset_df[params["target_column"]],
                                                        random_state=42)
    # Нормировка признаков
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)        
    return X_train, X_test, y_train, y_test