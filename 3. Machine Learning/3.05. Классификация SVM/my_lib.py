import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from pprint import pprint, pformat
import copy

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, f1_score

import joblib

from pathlib import Path

from my_config import *

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
    
def fill_with_mode(data, group_col, target_col):
    global_mode = data[target_col].mode()[0]
    def fill_group_mode(x):
        group_mode = x.mode()
        if not group_mode.empty:
            return group_mode[0]
        else:
            return global_mode
    data[target_col] = data.groupby(group_col)[target_col].transform(fill_group_mode)
    
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


def add_scores(models_scores, idx, y_train, y_train_pred, y_test, y_test_pred, params="", coef=""):
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)    
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)    
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
        
    models_scores.loc[idx, "mse_train"] = mse_train
    models_scores.loc[idx, "mse_test"] = mse_test
    models_scores.loc[idx, "rmse_train"] = rmse_train
    models_scores.loc[idx, "rmse_test"] = rmse_test
    models_scores.loc[idx, "mae_train"] = mae_train
    models_scores.loc[idx, "mae_test"] = mae_test
    models_scores.loc[idx, "r2_score_train"] = r2_train
    models_scores.loc[idx, "r2_score_test"] = r2_test
    models_scores.loc[idx, "params"] = params
    models_scores.loc[idx, "coef"] = coef
    
    joblib.dump(models_scores, Path(result_foler, models_scores_file_name), compress=3)
    
def fit_model(models_scores, model_name, model, X_train, y_train, X_test, y_test, best_params=None):

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    r2_train = round(model.score(X_train, y_train),4)
    r2_test = round(model.score(X_test, y_test),4)
    mae_train = round(mean_absolute_error(y_train, y_train_pred), 4)
    mae_test = round(mean_absolute_error(y_test, y_test_pred), 4)
    params = model.get_params()
    
    models_scores.loc[model_name, 'r2_train'] = r2_train
    models_scores.loc[model_name, 'r2_test'] = r2_test
    models_scores.loc[model_name, 'mae_train'] = mae_train
    models_scores.loc[model_name, 'mae_test'] = mae_test
    if best_params is not None:
        models_scores.loc[model_name, 'best_params'] = pformat(best_params)
    
    joblib.dump(models_scores, Path(result_foler, models_scores_file_name), compress=3)
    joblib.dump(model, Path(result_foler, model_template_filename % model_name), compress=3)    
    
    
def calculate_metrics_and_plot_roc(model, X_train, y_train, X_test, y_test, model_name):
    # Обучение модели
    model.fit(X_train, y_train)

    # Предсказание вероятностей на обучающей и тестовой выборке
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Расчет AUC-ROC
    roc_auc_train = roc_auc_score(y_train, y_train_proba)
    roc_auc_test = roc_auc_score(y_test, y_test_proba)
    print(f"ROC-AUC Train: {roc_auc_train:.2f}")
    print(f"ROC-AUC Test: {roc_auc_test:.2f}")

    # Построение ROC-кривой
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='ROC Curve Train')
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='ROC Curve Test')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({type(model).__name__})')
    plt.legend()
    plt.show()

    # Поиск порога, максимизирующего F1-score
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = [f1_score(y_test, y_test_proba >= t) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Optimal Threshold: {optimal_threshold:.2f}")

    # Пересчет метрик с учетом оптимального порога
    y_train_pred_optimal = (y_train_proba >= optimal_threshold).astype(int)
    y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

    train_precision = precision_score(y_train, y_train_pred_optimal)
    train_recall = recall_score(y_train, y_train_pred_optimal)
    train_f1_score = f1_score(y_train, y_train_pred_optimal)

    test_precision = precision_score(y_test, y_test_pred_optimal)
    test_recall = recall_score(y_test, y_test_pred_optimal)
    test_f1_score = f1_score(y_test, y_test_pred_optimal)

    print(f"Optimal Training Precision: {train_precision:.2f}")
    print(f"Optimal Training Recall: {train_recall:.2f}")
    print(f"Optimal Training F1: {train_f1_score:.2f}")
    print(f"Optimal Test Precision: {test_precision:.2f}")
    print(f"Optimal Test Recall: {test_recall:.2f}")
    print(f"Optimal Test F1: {test_f1_score:.2f}")

    metrics = {
        'params': [
            'Training_Precision', 'Test_Precision',
            'Training_Recall', 'Test_Recall',
            'ROC_AUC_Train', 'ROC_AUC_Test',
            'Train_F1', 'Test_F1'
        ],
        'values': [
            train_precision, test_precision,
            train_recall, test_recall,
            roc_auc_train, roc_auc_test,
            train_f1_score, test_f1_score
        ],
        'metrics': [
            model_name, model_name,
            model_name, model_name,
            model_name, model_name,
            model_name, model_name
        ]
    }

    return metrics
