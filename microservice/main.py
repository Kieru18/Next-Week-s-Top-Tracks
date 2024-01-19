from typing import Union, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from contextlib import asynccontextmanager
from microservice.router import router
#from microservice.models import 
from starlette_context.plugins.request_id import RequestIdPlugin
from starlette_context.middleware import RawContextMiddleware
import uvicorn


def load_models():
    pass

@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models()
    yield
    # clear_models() or any action just before shutdown


app = FastAPI(lifespan=lifespan)
app.add_middleware(RawContextMiddleware, plugins=(RequestIdPlugin(force_new_uuid=True),))
app.include_router(router)

def main():
    uvicorn.run("microservice.main:app", host="localhost", log_level="info")

if __name__ == '__main__':
    main()
