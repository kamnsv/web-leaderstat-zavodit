from pydantic import BaseModel, BaseSettings, Field


class Classes(BaseModel):
    name: str = ''
    imgs: list = []


class Opts(BaseModel):
    name: str = ''
    batch: int = 32
    epoch: int = 10
    lr:  float = 0.0001 
    imgs: list = []  # Датасет
    coef: int  = 1   # Коэфициент аугментации
    load: int  = 0   # Кол-во загрузок из интернета
    
class Settings(BaseSettings):
    path_dataset: str = '/app/static/dataset'
    path_temp: str = '/app/temp'
    count_img_prev = Field('5', env='COUNT_PREV')
   
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
