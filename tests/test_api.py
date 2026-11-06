import shutil

import pytest
from fastapi.testclient import TestClient

from reflexone.api import create_app


def test_api_root_opens_interactive_documentation(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/docs"


def test_api_readiness_fails_when_released_profiles_are_missing(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json() == {
            "ready": False,
            "loaded_profiles": [],
            "missing_profiles": ["commands", "spam", "support"],
        }


def test_api_lists_only_loaded_profiles_and_rejects_bad_requests(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json()["ready"] is False
        assert client.get("/profiles").json()["loaded"] == []
        assert client.post("/predict", json={"profile": "spam", "text": "hello"}).status_code == 503
        assert client.post("/predict", json={"profile": "unknown", "text": "hi"}).status_code == 422
        assert client.post("/predict", json={"profile": "support", "text": "hi"}).status_code == 422
        assert client.post("/predict", json={"profile": "spam", "text": " "}).status_code == 422


def test_api_batch_is_bounded_and_validates_before_any_prediction(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        body = {"requests": [{"profile": "spam", "text": "hello"}] * 33}
        assert client.post("/predict/batch", json=body).status_code == 422
        assert client.post("/predict/batch", json={"requests": []}).status_code == 422


def test_real_classical_profile_api_contract(tmp_path):
    import reflexone.tasks as tasks

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"text": f"family lunch tomorrow number {i}", "label": "ham", "id": f"ham-{i}"}
        for i in range(20)
    ] + [
        {"text": f"win cash prize urgent click number {i}", "label": "spam", "id": f"spam-{i}"}
        for i in range(20)
    ]
    import zipfile

    # Local fixture is the same source schema consumed by production preparation.
    raw = data_dir / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "sms.zip", "w") as archive:
        archive.writestr("SMSSpamCollection", "\n".join(f"{r['label']}\t{r['text']}" for r in rows))
    # prepare_task supports a raw fixture cache, without touching the network.
    tasks.prepare_task("spam", data_dir)
    tasks.train_task("spam", data_dir, tmp_path / "artifacts" / "spam")
    with TestClient(create_app(tmp_path / "artifacts")) as client:
        assert client.get("/health").json()["ready"] is True
        readiness = client.get("/ready")
        assert readiness.status_code == 503
        assert readiness.json()["missing_profiles"] == ["commands", "support"]
        result = client.post("/predict", json={"profile": "spam", "text": "win cash prize now"})
        assert result.status_code == 200
        assert result.json()["label"] == "spam"
        assert set(result.json()["probabilities"]) == {"ham", "spam"}
    shutil.move(str(tmp_path / "artifacts" / "spam"), str(tmp_path / "artifacts" / "commands"))
    with pytest.raises(ValueError, match="profile|directory|expected"):
        with TestClient(create_app(tmp_path / "artifacts")):
            pass
