from pydantic import BaseModel, BaseSettings, Field
import os

class Classes(BaseModel):
    name: str = ''
    imgs: list = []


class Opts(BaseModel):
    name: str = ''
    batch: int = 32
    imgs: list = []  # Датасет
    coef: int  = 1   # Коэфициент аугментации
    load: int  = 0   # Кол-во загрузок из интернета
    
class Settings(BaseSettings):
    path_static: str = os.path.join('app', 'static')
    path_dataset: str = os.path.join('app', 'static', 'dataset')
    path_temp: str = os.path.join('app', 'static', 'temp')
    path_train: str = os.path.join('data', 'Train')
    path_data: str = 'data'
    count_img_prev = Field('5', env='COUNT_PREV')
    gpu_host = Field('', env='HOST_GPU')
    gpu_port = Field('', env='PORT_GPU')
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'