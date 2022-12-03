from fastapi import FastAPI
from .routes import set_routes


app = FastAPI(docs_url="/docs")

set_routes(app)