from fastapi import FastAPI

from predictions_server import (
    ModelType,
    Prediction,
    TrackData,
    PredictonsServer
)


app = FastAPI()


@app.post("/predict")
async def predict_labels(model: ModelType, track_data: list[TrackData]) -> list[Prediction]:
    predictions_server = PredictonsServer(model_type=model)
    return predictions_server.serve_predictions(track_data=track_data)
