from fastapi import Query, Response, WebSocket, File, UploadFile, Form
from .models import Classes
from .predict import get_classes, cfg, add_class, save_file, predict_img
from fastapi.staticfiles import StaticFiles
import os


def set_routes(app):
    
    @app.get('/classes', response_model=list[Classes])
    async def classes(resp: Response):
        result = await get_classes()
        return result
    
    
    @app.post("/put")
    async def put_file(name: str = Form(...), data: UploadFile = File(...)):
        return await save_file(data, os.path.join(cfg.path_dataset, name))

    @app.post("/tmp")
    async def put_file(name: str = Form(...), data: UploadFile = File(...)):
        await save_file(data, os.path.join(cfg.path_temp, name))
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
        #try:
            while True:
                data = await websocket.receive_json()
                if 'fit' == data['action']:
                    await add_class(data, websocket)
                elif 'predict' == data['action']:
                    await predict_img(data, websocket)
                else:
                    await websocket.send_json({'action': data['action'], 'error': 'unknown'})
        #except Exception as e:
         #   print("Client disconnected", e)
        
       
    
    src_dir = os.path.dirname(__file__)
    static_dir = os.path.join(src_dir, "static/")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")