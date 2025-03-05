# Импорт метрик для оценки качества моделей классификации
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    roc_auc_score, 
    roc_curve, 
    f1_score,  #f1-мера
    accuracy_score,  # Метрика точности для классификации
    classification_report,  # Отчет о классификации
    confusion_matrix
)

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as plotly_px
import plotly.graph_objects as plotly_go
import plotly.subplots as plotly_subplt

import numpy as np

from pathlib import Path

import joblib

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class ModelClass():
    
    def __init__(self, name):
        self.name = name
        self.model_params = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None
        self.train_precision = None
        self.test_precision = None
        self.train_recall = None
        self.test_recall = None
        self.train_roc_auc = None
        self.test_roc_auc = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.train_f1_score = None
        self.test_f1_score = None
        self.specific_data = None
        self.y_train = None
        self.y_test = None
        self.y_train_proba = None
        self.y_test_proba = None
    
    def create_model(self, model_class, model_params, X_train, X_test, y_train, y_test):
        self.model_params = model_params
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model_class(**self.model_params)
    
    def fit(self):
        self.model.fit(self.X_train, self.y_train)    

    def calc_metrics(self):
        """Посчитать метрики модели"""
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_train_prob = self.model.predict_proba(self.X_train)[:, 1]
        self.y_test_pred = self.model.predict(self.X_test)
        self.y_test_prob = self.model.predict_proba(self.X_test)[:, 1]
    
        # матрица ошибок
        self.conf_matrix_train = confusion_matrix(self.y_train, self.y_train_pred)
        self.conf_matrix_test = confusion_matrix(self.y_test, self.y_test_pred)

        # Расчет AUC-ROC
        self.train_roc_auc = roc_auc_score(self.y_train, self.y_train_prob)
        self.test_roc_auc = roc_auc_score(self.y_test, self.y_test_prob)

        # Поиск порога, максимизирующего F1-score
        thresholds = np.arange(0.0, 1.0, 0.01)
        f1_scores = [f1_score(self.y_test, self.y_test_prob >= t) for t in thresholds]
        self.optimal_threshold = thresholds[np.argmax(f1_scores)]

        # Пересчет метрик с учетом оптимального порога
        self.y_train_pred_optimal = (self.y_train_prob >= self.optimal_threshold).astype(int)
        self.y_test_pred_optimal = (self.y_test_prob >= self.optimal_threshold).astype(int)

        self.train_precision = precision_score(self.y_train, self.y_train_pred_optimal)
        self.train_recall = recall_score(self.y_train, self.y_train_pred_optimal)

        self.test_precision = precision_score(self.y_test, self.y_test_pred_optimal)
        self.test_recall = recall_score(self.y_test, self.y_test_pred_optimal)

        self.train_accuracy = accuracy_score(self.y_train, self.y_train_pred_optimal)
        self.test_accuracy = accuracy_score(self.y_train, self.y_train_pred_optimal)

        self.train_f1_score = f1_score(self.y_train, self.y_train_pred_optimal)
        self.test_f1_score = f1_score(self.y_train, self.y_train_pred_optimal)
        
    def show_quality(self): #X_train, X_test, y_train, y_test, check_result, title, grid_search, model_cl):
        """Показать различные метрики и промежуточные переменные обучения"""
        #def show_quality2(X_train, X_test, y_train, y_test, check_result, title, grid_search, model_cl):
        fig = plotly_subplt.make_subplots(rows=2, cols=2, 
                                        subplot_titles=['ROC AUC', 'Metrics', 'Confusion Matrix Train', 'Confusion Matrix Test'],
                                        vertical_spacing = 0.1,
                                        row_width=[0.4, 0.6])
        fig.update_layout(
            title_x=0.5,
            title_text=self.name,
            width = 1000,
            height = 800,
            legend = dict(yanchor="bottom", y=0.63, xanchor="right", x=0.44),
            margin = {'t':80, 'b':50, 'l':10, 'r':10}
            
        )

        # Построение ROC кривой
        fpr_test, tpr_test, thresholds = roc_curve(self.y_test, self.y_test_prob)
        fpr_train, tpr_train, thresholds = roc_curve(self.y_train, self.y_train_prob)
        roc_train_g = plotly_go.Scatter(x=fpr_train, y=tpr_train, name="ROC curve Train", line={'color':'green'})
        roc_test_g = plotly_go.Scatter(x=fpr_test, y=tpr_test, name="ROC curve Test", line={'color':'blue'})
        roc_diag_g = plotly_go.Scatter(x=[0, 1], y=[0, 1], line={'color':'gray', 'dash': 'dash'}, showlegend=False)

        fig.add_trace(roc_train_g, row=1, col=1)
        fig.add_trace(roc_test_g, row=1, col=1)
        fig.add_trace(roc_diag_g, row=1, col=1)
        
        fig.update_layout(
            xaxis1 = {'title_text': "False Positive Rate"},
            yaxis1 = {'title_text': "True Positive Rate"}
        )    
        

        # Bar с метриками
        df_metrics = pd.DataFrame([[self.test_accuracy,  self.train_accuracy],
                                   [self.test_precision, self.train_precision],
                                   [self.test_recall,    self.train_recall],
                                   [self.test_roc_auc,   self.train_roc_auc],
                                   [self.test_f1_score,  self.train_f1_score]], 
                                  columns = ["Test", "Train"], 
                                  index=["accuracy", "precision", "recall", "ROC AUC", "F1"])
        metrics_train = plotly_go.Bar(x=df_metrics.index, y=df_metrics.Train, 
                        showlegend=True, text=round(df_metrics.Train,4), textangle=0, 
                        xaxis='x2', yaxis='y2', name="Train Metrics")
        metrics_test = plotly_go.Bar(x=df_metrics.index, y=df_metrics.Test, 
                        showlegend=True, text=round(df_metrics.Test,4), textangle=0, 
                        xaxis='x2', yaxis='y2', name="Test Metrics")

        fig.add_trace(metrics_train, row=1, col=2) 
        fig.add_trace(metrics_test, row=1, col=2) 

        # Confusion Matrix 
        cm_normalized_train = self.conf_matrix_train.astype('float') / self.conf_matrix_train.sum(axis=1)[:, np.newaxis]
        heatmap_train = plotly_go.Heatmap(z=cm_normalized_train, x=['0', '1'], y=['0', '1'], colorscale='Blues', 
                                        text=np.round(cm_normalized_train, 3), texttemplate="%{text}", showscale=False)

        cm_normalized_test = self.conf_matrix_test.astype('float') / self.conf_matrix_test.sum(axis=1)[:, np.newaxis]
        heatmap_test = plotly_go.Heatmap(z=cm_normalized_test, x=['0', '1'], y=['0', '1'], colorscale='Blues', 
                                        text=np.round(cm_normalized_test, 3), texttemplate="%{text}", showscale=False)

        fig.add_trace(heatmap_train, row=2, col=1)
        fig.add_trace(heatmap_test,  row=2, col=2) 

        fig.update_layout(
            xaxis1 = {'title': 'Predict'},
            xaxis2 = {'title': 'Predict'},
            yaxis1 = {'title': 'Goals'},
            yaxis2 = {'title': 'Goals'}
        )    
        
        fig.show()
    
    @staticmethod
    def metrics_names():
        return ['Training_Precision', 'Test_Precision',
                'Training_Recall', 'Test_Recall',
                'ROC_AUC_Train', 'ROC_AUC_Test',
                'Accuarcy_Train', 'Accuarcy_Test',
                'F1_score_Train', 'F1_score_Test'
                ]
    
    def metrics(self):
        """Сформировать словарь о сзначениями метрик модели"""
        metrics_as_dict = {
                'params': ModelClass.metrics_names(),
                'values': [
                    self.train_precision, self.test_precision,
                    self.train_recall, self.test_recall,
                    self.train_roc_auc, self.test_roc_auc,
                    self.train_accuracy, self.test_accuracy,
                    self.train_f1_score, self.test_f1_score
                ],
                'model_name': [self.name for i in range(len(ModelClass.metrics_names()))]
            }      
        return metrics_as_dict
    
def load_or_create_and_fit_GridSearchCV(model_name, model_class, param_grid, X_train, y_train,
                                        settings, 
                                        scoring='roc_auc', 
                                        need_save=True, n_jobs=6, verbose=1):
    """Загрузить ранее обученные GridSearchCV из кеша. 
    Если в кеше нет - создать и потренировать, найдя лучшие параметры"""
    grid_search_filename_cache = settings.cache_gridsearch_filename(model_name)
    grid_search_filename = settings.result_gridsearch_filename(model_name)

    if Path.is_file(grid_search_filename_cache):
        print(f"GridSearchCV() загружен из {grid_search_filename_cache}")
        grid_search = joblib.load(grid_search_filename_cache)
    else:
        print(f"Создается и выполняется GridSearchCV для модели {model_name} класса {model_class.__name__}")
        # Создаем объект GridSearchCV с моделью логистической регрессии и сеткой параметров
        grid_search = GridSearchCV(model_class(), param_grid, cv=5, n_jobs=n_jobs, verbose=verbose, scoring=scoring)
        # Обучаем модель на данных с использованием кросс-валидации
        grid_search.fit(X_train, y_train)
    
        if need_save:
            _ = joblib.dump(grid_search, grid_search_filename)
    return grid_search

def load_or_create_and_fit_model(model_name, model_class, model_params, 
                                 X_train, X_test, y_train, y_test,
                                 settings, 
                                 need_save=True):
    """Загрузить ранее обученную модель из кеша.
    Если в кеше нет - обучить на переданных данных с заданными параметрами.
    """
    model_filename_cache = settings.cache_model_filename(model_name)
    model_filename = settings.result_model_filename(model_name)

    if Path.is_file(model_filename_cache):
        model = joblib.load(model_filename_cache)
        print(f"Модель {type(model.model).__name__} загружена из {model_filename_cache}")
    else:
        print(f"Создается и тренируется модель {model_name} класса {model_class.__name__}")
        model = ModelClass(model_name)
        model.create_model(model_class, model_params, X_train, X_test, y_train, y_test)
        model.fit()
        model.calc_metrics()
        if need_save:
            _ = joblib.dump(model, model_filename)
    return model