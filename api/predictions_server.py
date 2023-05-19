import pickle

from pydantic import BaseModel
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import itertools
import operator

from datetime import date
from typing import Optional, Union
from enum import Enum


class ModelType(str, Enum):
    BASE = 'BASE'
    FINAL = 'FINAL'


class TrackData(BaseModel):
    id: str
    name: str
    popularity: int
    duration_ms: int
    explicit: int
    id_artist: str
    release_date: Union[date, str]
    danceability: float
    energy: float
    key: int
    mode: Optional[int] = None
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    time_signature: int


class Prediction(BaseModel):
    artist_id: str
    labels: list[str]


UNNECESSARY_COLUMNS = ['name', 'id', 'release_date', 'mode', 'id_artist']
CATEGORICAL_COLUMNS = ['time_signature', 'explicit', 'key']


class PredictonsServer:
    def __init__(self, model_type: ModelType):
        self._model_type = model_type

    def serve_predictions(self, track_data: list[TrackData]) -> list[Prediction]:
        preapred_data = self._prepare_data(data=DataFrame([row.dict() for row in track_data]))
        model = self._load_model()
        predictions = model.predict(preapred_data)
        merged_predictions = [(data.id_artist, label) for data, label in zip(track_data, predictions)]
        grouped_predictions = itertools.groupby(merged_predictions, operator.itemgetter(0))
        return [
            Prediction(artist_id=artist, labels=[label[1] for label in label]) for artist, label in grouped_predictions
        ]

    def _prepare_data(self, data: DataFrame) -> DataFrame:
        data.drop(UNNECESSARY_COLUMNS, axis=1, inplace=True)
        return data

    def _normalize_data(self, data: DataFrame) -> DataFrame:
        data_to_normalize = data.drop(columns=CATEGORICAL_COLUMNS)
        normalized_data = (data_to_normalize - data_to_normalize.min()) / \
            (data_to_normalize.max() - data_to_normalize.min())
        data[list(data_to_normalize.columns)] = normalized_data[list(data_to_normalize.columns)]
        return data

    def _load_model(self) -> Union[GaussianNB, RandomForestClassifier]:
        model_path = '../models/gnb_model.pickle' if self._model_type is ModelType.BASE \
            else '../models/rfc_model.pickle'
        return pickle.load(open(model_path, 'rb'))
