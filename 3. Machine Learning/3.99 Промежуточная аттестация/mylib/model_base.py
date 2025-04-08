import abc
from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class ModelBase(abc.ABC):

    def __init__(self, name):
        self.name = name
        self.model_params = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None

    def create_model(self, model_class, model_params, X_train, X_test, y_train, y_test):
        self.model_params = model_params
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model_class(**self.model_params)
    
    def fit(self):
        self.model.fit(self.X_train, self.y_train)    

    @abc.abstractmethod
    def calc_metrics(self):
        raise NotImplemented
    
    @abc.abstractmethod
    def show_quality(self): 
        raise NotImplemented
    
    @staticmethod
    def metrics_names():
        raise NotImplemented
    
    @staticmethod
    def metrics(self):
        raise NotImplemented

    @staticmethod
    def _load_or_create_and_fit_model(model_meta_class, 
                                      model_name, model_class, model_params, 
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
            model = model_meta_class(model_name)
            model.create_model(model_class, model_params, X_train, X_test, y_train, y_test)
            model.fit()
            model.calc_metrics()
            if need_save:
                _ = joblib.dump(model, model_filename)
        return model
    
    @staticmethod
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
            #grid_search = GridSearchCV(model_class(), param_grid, cv=5, n_jobs=n_jobs, verbose=verbose, scoring=scoring)
            #grid_search = GridSearchCV(model_class(), param_grid, cv=5, verbose=verbose, scoring=scoring)
            grid_search = RandomizedSearchCV(model_class(), param_grid, cv=5, n_jobs=n_jobs, verbose=verbose, scoring=scoring)
            
            # Обучаем модель на данных с использованием кросс-валидации
            grid_search.fit(X_train, y_train)
        
            if need_save:
                _ = joblib.dump(grid_search, grid_search_filename)
        return grid_search    