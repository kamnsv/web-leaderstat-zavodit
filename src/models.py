from pydantic import BaseModel, BaseSettings, Field


class Classes(BaseModel):
    name: str = ''
    imgs: list = []
    



class Settings(BaseSettings):
    path_dataset: str = '/app/dataset'
    url: str = Field('30', env='URL_API')
    per: int = Field('10', env='PER_PAGE')
    token: str = Field(..., env='TOKEN_API')
    debug: bool = Field('True', env='DEBUG')
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
