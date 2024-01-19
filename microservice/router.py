from fastapi import APIRouter
from datetime import date

router = APIRouter()

@router.post("/predict")
async def predict(track_id: str, date: date, model):
    return {"Hello": "World"}
