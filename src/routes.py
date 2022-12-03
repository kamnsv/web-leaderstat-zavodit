from fastapi import Query, Response, WebSocket, File, UploadFile, Form
from .models import Classes
from .predict import get_classes, cfg
from fastapi.staticfiles import StaticFiles
import os

def set_routes(app):
    
    @app.get('/classes', response_model=list[Classes])
    async def classes(resp: Response):
        result = await get_classes()
        return result
    

    @app.post("/put")
    async def put_file(name: str = Form(...), data: UploadFile = File(...)):

        try:
            contents = data.file.read()
            path = os.path.join(cfg.path_dataset, name, data.filename)
            os.makedirs(os.path.join(cfg.path_dataset,name), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(contents)
        else:
  
            return {"message": "There was an error uploading the file(s)"}
        finally:
            data.file.close()

        return {"message": f"Successfuly uploaded {data.filename}"}    
        
        
        
    websockets = {}
    @app.websocket("/ws/{id_client}")
    async def websocket_endpoint(websocket: WebSocket, id_client: str):
        await websocket.accept()
        websockets.update({id_client: websocket})
        #await websocket.send_text(f"Message text was: {data}")
    
    
    src_dir = os.path.dirname(__file__)
    static_dir = os.path.join(src_dir, "static/")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")