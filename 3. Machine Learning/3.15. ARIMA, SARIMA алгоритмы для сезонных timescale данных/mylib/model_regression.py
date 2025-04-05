from sklearn.linear_model import(LinearRegression, # для построения моделей линейной регрессии
                                 LogisticRegression, 
                                 Ridge, 
                                 Lasso, 
                                 ElasticNet
)
from sklearn.preprocessing import PolynomialFeatures # для преобразования исходных признаков в полиномиальные, для построения моделей полиномиальной регрессии
# Импорт метрик для оценки качества моделей
from sklearn.metrics import (
    mean_squared_error,  # Средняя квадратичная ошибка для регрессии
    mean_absolute_error, 
    root_mean_squared_error, 
    r2_score  # Коэффициент детерминации для регрессии
)

import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from mylib.model_base import ModelBase

class ModelRegression(ModelBase):
    
    def __init__(self, name):
        super().__init__(name)
        
        self.mse_train = None
        self.r2_train = None
        self.rmse_train = None
        self.mae_train = None

        self.mse_test = None
        self.r2_test = None
        self.rmse_test = None
        self.mae_test = None

    def calc_metrics(self):
        """Посчитать метрики модели"""
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)
        
        self.mse_train = mean_squared_error(self.y_train, self.y_train_pred)
        self.r2_train = r2_score(self.y_train, self.y_train_pred)
        self.rmse_train = root_mean_squared_error(self.y_train, self.y_train_pred)
        self.mae_train = mean_absolute_error(self.y_train, self.y_train_pred)   
        self.median_train = self.y_train.median() 

        self.mse_test = mean_squared_error(self.y_test, self.y_test_pred)
        self.r2_test = r2_score(self.y_test, self.y_test_pred)
        self.rmse_test = root_mean_squared_error(self.y_test, self.y_test_pred)
        self.mae_test = mean_absolute_error(self.y_test, self.y_test_pred)    
        self.median_test = self.y_test.median() 
    
        
    def show_quality(self): 
        """Показать различные метрики"""
        print('Train data:')
        print(f"  MSE:    {round(self.mse_train,4)}")
        print(f"  RMSE:   {round(self.rmse_train,4)}")
        print(f"  MAE:    {round(self.mae_train,4)}")
        print(f"  r2:     {round(self.r2_train,4)}")
        print(f"  median: {round(self.median_train,4)}")

        print('Test data:')
        print(f"  MSE:    {round(self.mse_test,4)}")
        print(f"  RMSE:   {round(self.rmse_test,4)}")
        print(f"  MAE:    {round(self.mae_test,4)}")
        print(f"  r2:     {round(self.r2_test,4)}")    
        print(f"  median: {round(self.median_train,4)}")        
    
    @staticmethod
    def metrics_names():
        return ['Train_MSE', 'Test_MSE',
                'Train_RMSE', 'Test_RMSE',
                'Train_MAE', 'Test_MAE',
                'Train_R2', 'Test_R2',
                'Train_median', 'Test_Median'
                ]
    
    def metrics(self):
        """Сформировать словарь о сзначениями метрик модели"""
        metrics_as_dict = {
                'params': ModelRegression.metrics_names(),
                'values': [
                    self.mse_train, self.mse_test,
                    self.rmse_train, self.rmse_test,
                    self.mae_train, self.mae_test,
                    self.r2_train, self.r2_test,
                    self.median_train, self.median_train
                ],
                'model_name': [self.name for i in range(len(ModelRegression.metrics_names()))]
            }      
        return metrics_as_dict

    @staticmethod    
    def load_or_create_and_fit_model(model_name, model_class, model_params, 
                                    X_train, X_test, y_train, y_test,
                                    settings, 
                                    need_save=True):
        """Загрузить ранее обученную модель из кеша.
        Если в кеше нет - обучить на переданных данных с заданными параметрами.
        """
        return ModelBase._load_or_create_and_fit_model(ModelRegression, 
                                                       model_name, model_class, model_params, 
                                                       X_train, X_test, y_train, y_test,
                                                       settings, 
                                                       need_save)