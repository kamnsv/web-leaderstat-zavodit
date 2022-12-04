from .models import Classes, Settings, Opts
from datetime import datetime
import os
import random
from icrawler.builtin import GoogleImageCrawler
import shutil
from .ResNet import Predict, Fit
from . import ResNet
import asyncio

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
    await send_ws(ws, {'action': 'fit', 'status': 'start'})
    opt = Opts(**data)
    
    #
    path = os.path.join(cfg.path_dataset, opt.name)
    await send_ws(ws, {'action': 'fit', 'status': 'path', 'path': path})
    os.makedirs(path, exist_ok=True)
    
    #
    await send_ws(ws, {'action': 'fit', 'status': 'loading', 'loading': opt.load})
    if (opt.load): 
        await load_inet(path, opt.load, opt.name)
        await send_ws(ws, {'action': 'fit', 'status': 'loaded'})
       
    # 
    
    count_copy = 0
    train = os.path.join(cfg.path_train, opt.name)
    await send_ws(ws, {'action': 'fit', 'status': 'start_copy', 'path': path, 'target': train})
    os.makedirs(train, exist_ok=True)
    for fname in os.listdir(path):
        src = os.path.join(path, fname)
        target = os.path.join(train, fname)
        if os.path.isfile(src) and not os.path.isfile(target):
            shutil.copy(src, target)
            count_copy += 1
    await ws.send_json({'action': 'fit', 'status': 'end_copy', 'count': count_copy})
    
    def Callback(MustDone, Ready, TimePassed, TimeNeed):
        me = {'action': 'fit', 
              'status': 'progress', 
              'must_done': MustDone,
              'ready': Ready,
              'time_passed': TimePassed,
              'time_need': TimeNeed,
        }
        sync_send_ws(ws, me)
    

    ResNet.Callback = Callback
    
    
    await send_ws(ws, {'action': 'fit', 'status': 'fit'})
    #
    Fit(cfg.path_data+os.sep, BatchSz=opt.batch, Au=opt.coef)
    
    await send_ws(ws, {'action': 'fit', 'status': 'end'})    
        
async def predict_img(data, ws):
    name = data['name']
    path = os.path.join(cfg.path_temp, name)
    count = len(os.listdir(path))
    await send_ws(ws, {'action': 'predict', 'count': count, 'path': path})
    for i, fname in enumerate(os.listdir(path)):
        await send_ws(ws, {'action': 'predict',
                            'predicted': i + 1, 
                            'count': count, 
                            'res': Predict(os.path.join(path, fname)), 
                            'src': f'/temp/{name}/{fname}'
                            })


def sync_send_ws(ws, me):
    try:
        loop = asyncio.get_running_loop()
    except:  
        loop = None

    if loop and loop.is_running():
        tsk = loop.create_task(send_ws(ws, me))
    else:
        asyncio.run(send_ws(ws, me))
 
async def send_ws(ws, me):
    try:
        await ws.send_json(me)
    except Exception as e:
        print('send_ws', me, e)