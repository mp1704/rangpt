import sys
sys.path.append("src")
import qa_app
from qa_app import app
from fastapi.testclient import TestClient


def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "hello world"}