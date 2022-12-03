from .models import Classes, Settings
from datetime import datetime
import os

cfg = Settings()

async def get_classes():
    return [Classes(name=x) for x in os.listdir(cfg.path_dataset)]