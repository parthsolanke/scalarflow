import io
import os
from contextlib import redirect_stdout

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scalarflow.nn import network


def _build_network():
    num_in = int(os.getenv("SCALARFLOW_NUM_IN", "2"))
    layer_sizes = [
        int(size) for size in os.getenv("SCALARFLOW_LAYER_SIZES", "4,2,1").split(",") if size
    ]
    activation = os.getenv("SCALARFLOW_ACTIVATION", "")
    return network(num_in=num_in, layer_sizes=layer_sizes, unit_activation=activation)


app = FastAPI(title="ScalarFlow", description="HTTP server for ScalarFlow networks")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = _build_network()


class PredictRequest(BaseModel):
    inputs: list[list[float]]


class TrainRequest(BaseModel):
    inputs: list[list[float]]
    outputs: list[float]
    epochs: int = 100


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/summary")
def summary():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.summary()
    return {"summary": buffer.getvalue()}


@app.post("/predict")
def predict(request: PredictRequest):
    predictions = model.predict(request.inputs)
    return {"predictions": [float(p) for p in predictions]}


@app.post("/train")
def train(request: TrainRequest):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.fit(request.inputs, request.outputs, epochs=request.epochs)
    return {"epochs": request.epochs, "log": buffer.getvalue()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
