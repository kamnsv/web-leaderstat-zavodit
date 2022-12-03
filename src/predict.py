from .models import Classes, Settings, Opts
from datetime import datetime
import os
import random
from bing_image_downloader.downloader import download

cfg = Settings()

async def get_classes():
    
    return [Classes(
                    name=x, 
                    imgs=await rnd_prev_imgs(x))
                for x in os.listdir(cfg.path_dataset)
           ]
            
async def rnd_prev_imgs(cls_dir):
 
    lst = os.listdir(os.path.join(cfg.path_dataset, cls_dir))
    count = int(cfg.count_img_prev)
    if count  > len(lst): return lst
    return random.choices(lst, k=count)
    
    
async def add_class(data, ws):
    opt = Opts(**data)
    
    #
    path = os.path.join(cfg.path_dataset, opt.name)
    await ws.send_json({'path': path})
    os.makedirs(path, exist_ok=True)
    
    #
    await ws.send_json({'load': opt.load})
    if (opt.load): await load_inet(cfg.path_dataset, opt.load, opt.name)

    

async def load_inet(path, count, search):
    download(search,
         limit=count,
         output_dir=path,
         adult_filter_off=True,
         force_replace=False,
         filter='',
         timeout=60,
         verbose=True)
         #+filterui:photo-photo+filterui:color2-color