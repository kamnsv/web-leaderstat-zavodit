from .models import Classes, Settings, Opts
from datetime import datetime
import os
import random
from icrawler.builtin import GoogleImageCrawler
import shutil
from .ResNet import Predict

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
    if count  >= len(lst): return lst
    return random.choices(lst, k=count)
    
    
    
async def load_inet(path, count, search):
    google_Crawler = GoogleImageCrawler(storage={'root_dir': path})
    google_Crawler.crawl(keyword=search, max_num=count)


async def save_file(data, path):
        
    try:
        contents = data.file.read()
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, data.filename)
        with open(fname, 'wb') as f:
            f.write(contents)
    except:
        return {"message": "There was an error uploading the file(s)"}
    finally:
        data.file.close()
    return {"message": f"Successfuly uploaded {data.filename}"} 
    
    
    
async def add_class(data, ws):
    opt = Opts(**data)
    
    #
    path = os.path.join(cfg.path_dataset, opt.name)
    await ws.send_json({'action': 'fit', 'path': path})
    os.makedirs(path, exist_ok=True)
    
    #
    await ws.send_json({'action': 'fit', 'loading': opt.load})
    if (opt.load): 
        await load_inet(path, opt.load, opt.name)
        await ws.send_json({'action': 'fit', 'loaded': opt.load})
        
        
        
async def predict_img(data, ws):
    name = data['name']
    path = os.path.join(cfg.path_temp, name)
    count = len(os.listdir(path))
    await ws.send_json({'action': 'predict', 'count': count, 'path': path})
    for i, fname in enumerate(os.listdir(path)):
        await ws.send_json({'action': 'predict',
                            'predicted': i + 1, 
                            'count': count, 
                            'res': Predict(os.path.join(path, fname)), 
                            'src': f'/temp/{name}/{fname}'
                            })

    
