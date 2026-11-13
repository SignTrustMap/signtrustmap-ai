import pytest
from fastapi.testclient import TestClient
from loguru import logger
from main import app
from src.infrastructure.schema_exporter import export_schema_dict


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


def test_export_schema_dict_content():
    data = export_schema_dict()
    assert isinstance(data, dict)
    expected_tables = {
        "model_versions",
        "active_learning_selections",
        "model_artifacts",
        "model_classes",
        "model_evaluation_metrics",
        "ai_system_configs",
        "support_shots",
        "text_embeddings",
        "evaluation_embeddings",
    }
    assert expected_tables.issubset(set(data.keys()))
    assert "candidate_embeddings" not in data
    for table_name in expected_tables:
        assert "columns" in data[table_name]
        assert "foreign_keys" in data[table_name]
        assert "indexes" in data[table_name]
        assert len(data[table_name]["columns"]) > 0

    mv_cols = {c["name"]: c for c in data["model_versions"]["columns"]}
    assert mv_cols["model_type"]["type"] == "ENUM"
    assert mv_cols["model_type"]["enum_values"] == ["DETECTOR", "CLASSIFIER"]
    assert mv_cols["created_at"]["type"] == "TIMESTAMP WITH TIME ZONE"
    assert mv_cols["created_at"]["default"] == "CURRENT_TIMESTAMP"
    assert mv_cols["is_active"]["type"] == "BOOLEAN"
    assert mv_cols["is_active"]["default"] is False

    al_cols = {c["name"]: c for c in data["active_learning_selections"]["columns"]}
    assert al_cols["selection_reason"]["type"] == "ENUM"
    assert al_cols["selection_reason"]["enum_values"] == ["UNCERTAIN", "HIGH_CONFIDENCE"]
    assert al_cols["selected_at"]["default"] == "CURRENT_TIMESTAMP"
    logger.info("Structured schema dictionary verified successfully.")


def test_api_get_db_schema(api_client: TestClient):
    response = api_client.get("/api/v1/system/db-schema")
    assert response.status_code == 200
    payload = response.json()
    assert payload["dialect"] == "postgresql"
    assert payload["tables_count"] >= 9
    assert isinstance(payload["tables"], dict)
    assert "model_versions" in payload["tables"]
    assert "support_shots" in payload["tables"]
    assert "text_embeddings" in payload["tables"]
    assert "evaluation_embeddings" in payload["tables"]
    assert "candidate_embeddings" not in payload["tables"]

    cols = {c["name"]: c for c in payload["tables"]["model_versions"]["columns"]}
    assert cols["model_type"]["type"] == "ENUM"
    assert cols["model_type"]["enum_values"] == ["DETECTOR", "CLASSIFIER"]
    assert cols["created_at"]["type"] == "TIMESTAMP WITH TIME ZONE"
    assert cols["created_at"]["default"] == "CURRENT_TIMESTAMP"
    assert cols["is_active"]["default"] is False
    logger.info("Endpoint /api/v1/system/db-schema verified successfully.")
