from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict.router import routerModel

app = FastAPI(
    title="Proyecto Machine Learning Back",
    description="API para modelos de machine learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


app.include_router(router=routerModel)
