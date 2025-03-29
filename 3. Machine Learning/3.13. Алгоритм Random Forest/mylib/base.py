from dataclasses import dataclass
import builtins
from pathlib import Path


def get_type(type_name):
    try:
        return getattr(builtins, type_name)
    except AttributeError:
        try:
            obj = globals()[type_name]
        except KeyError:
            return None
        return repr(obj) if isinstance(obj, type) else None
    

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


@dataclass 
class Settings():
    enviroment: object
    
    def __post_init__(self):
        self.dataset_folder = str(Path(Path.cwd(), self.enviroment["DATASET_SUBFOLDER"]))
        self.cache_folder = str(Path(Path.cwd(), self.enviroment["CACHE_SUBFOLDER"]))
        self.result_folder = str(Path(Path.cwd(), self.enviroment["RESULT_SUBFOLDER"]))
        
    def cache_gridsearch_filename(self, model_name): 
        return Path(self.cache_folder, self.enviroment["GRID_SEARCH_TEMPLATE_FILENAME"] % model_name)
    def cache_model_filename(self, model_name): 
        return Path(self.cache_folder, self.enviroment["MODEL_CLASS_TEMPLATE_FILENAME"] % model_name)
    def result_gridsearch_filename(self, model_name): 
        return Path(self.result_folder, self.enviroment["GRID_SEARCH_TEMPLATE_FILENAME"] % model_name) 
    def result_model_filename(self, model_name): 
        return Path(self.result_folder, self.enviroment["MODEL_CLASS_TEMPLATE_FILENAME"] % model_name)         