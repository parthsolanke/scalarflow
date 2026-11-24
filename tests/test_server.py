from fastapi.testclient import TestClient

from scalarflow.server import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summary():
    response = client.get("/summary")

    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "neural network" in body["summary"]


def test_predict_returns_one_prediction_per_input():
    response = client.post("/predict", json={"inputs": [[0.0, 0.0], [1.0, 1.0]]})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert all(isinstance(p, float) for p in predictions)


def test_predict_rejects_invalid_body():
    response = client.post("/predict", json={"inputs": "not-a-matrix"})

    assert response.status_code == 422


def test_train_returns_log():
    response = client.post(
        "/train",
        json={"inputs": [[0.0, 0.0], [1.0, 1.0]], "outputs": [0.0, 1.0], "epochs": 10},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["epochs"] == 10
    assert "Epoch" in body["log"]
