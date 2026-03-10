"""Tests for the FastAPI web API endpoints."""
from __future__ import annotations

import pytest
import yaml

httpx = pytest.importorskip("httpx", reason="httpx required for TestClient")
from starlette.testclient import TestClient

from web.api import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


# ===================================================================
# 1. GET /api/config returns config dict
# ===================================================================

class TestGetConfig:
    def test_status_ok(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200

    def test_returns_json_dict(self, client):
        resp = client.get("/api/config")
        data = resp.json()
        assert isinstance(data, dict)

    def test_has_expected_keys(self, client):
        resp = client.get("/api/config")
        data = resp.json()
        expected_keys = [
            "seed", "format", "startDate", "endDate",
            "salesRows", "customers", "stores", "products",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_seed_is_integer(self, client):
        resp = client.get("/api/config")
        data = resp.json()
        assert isinstance(data["seed"], int)

    def test_format_is_string(self, client):
        resp = client.get("/api/config")
        data = resp.json()
        assert isinstance(data["format"], str)
        assert data["format"] in ("csv", "parquet", "deltaparquet")


# ===================================================================
# 2. POST /api/config accepts partial updates
# ===================================================================

class TestPostConfig:
    def test_update_seed(self, client):
        resp = client.post("/api/config", json={"values": {"seed": 99}})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Verify the change took effect
        data = client.get("/api/config").json()
        assert data["seed"] == 99

    def test_update_format(self, client):
        resp = client.post("/api/config", json={"values": {"format": "csv"}})
        assert resp.status_code == 200
        data = client.get("/api/config").json()
        assert data["format"] == "csv"

    def test_update_dates(self, client):
        resp = client.post("/api/config", json={
            "values": {"startDate": "2020-01-01", "endDate": "2020-12-31"}
        })
        assert resp.status_code == 200
        data = client.get("/api/config").json()
        assert data["startDate"] == "2020-01-01"
        assert data["endDate"] == "2020-12-31"

    def test_update_sales_rows(self, client):
        resp = client.post("/api/config", json={"values": {"salesRows": 5000}})
        assert resp.status_code == 200
        data = client.get("/api/config").json()
        assert data["salesRows"] == 5000

    def test_update_returns_ok_field(self, client):
        resp = client.post("/api/config", json={"values": {"workers": 2}})
        body = resp.json()
        assert body["ok"] is True

    def test_invalid_body_returns_422(self, client):
        resp = client.post("/api/config", json={"bad_key": 123})
        assert resp.status_code == 422


# ===================================================================
# 3. GET /api/presets returns preset list
# ===================================================================

class TestGetPresets:
    def test_status_ok(self, client):
        resp = client.get("/api/presets")
        assert resp.status_code == 200

    def test_returns_dict_or_list(self, client):
        resp = client.get("/api/presets")
        data = resp.json()
        # Presets endpoint always returns a dict (keyed by category) or empty dict
        assert isinstance(data, dict)


# ===================================================================
# 4. POST /api/presets/apply applies a preset
# ===================================================================

class TestApplyPreset:
    def test_unknown_preset_returns_404(self, client):
        resp = client.post("/api/presets/apply", json={"name": "nonexistent_preset_xyz"})
        assert resp.status_code == 404

    def test_apply_requires_name(self, client):
        resp = client.post("/api/presets/apply", json={})
        assert resp.status_code == 422

    def test_apply_known_preset(self, client):
        """If presets are available, apply the first one and verify ok."""
        presets_resp = client.get("/api/presets")
        presets = presets_resp.json()
        if not presets:
            pytest.skip("No presets available")

        # Get first preset name from the structure
        first_name = None
        if isinstance(presets, dict):
            for category_presets in presets.values():
                if isinstance(category_presets, list) and category_presets:
                    first_name = category_presets[0].get("name") if isinstance(category_presets[0], dict) else str(category_presets[0])
                    break
                elif isinstance(category_presets, dict):
                    first_name = next(iter(category_presets), None)
                    break

        if first_name is None:
            pytest.skip("Could not find a preset name")

        resp = client.post("/api/presets/apply", json={"name": first_name})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


# ===================================================================
# 5. GET /api/validate returns errors/warnings
# ===================================================================

class TestValidateEndpoint:
    def test_status_ok(self, client):
        resp = client.get("/api/validate")
        assert resp.status_code == 200

    def test_returns_errors_and_warnings(self, client):
        resp = client.get("/api/validate")
        data = resp.json()
        assert "errors" in data
        assert "warnings" in data
        assert isinstance(data["errors"], list)
        assert isinstance(data["warnings"], list)

    def test_valid_config_has_no_errors(self, client):
        """The default config loaded at startup should ideally have no errors."""
        resp = client.get("/api/validate")
        data = resp.json()
        # Note: there may be warnings, but a well-formed default config
        # should not have blocking errors (unless parquet_folder/out_folder unset)
        assert isinstance(data["errors"], list)
        assert len(data["errors"]) == 0, f"Expected no validation errors but got: {data['errors']}"

    def test_bad_dates_produce_error(self, client):
        """Set end before start and validate."""
        client.post("/api/config", json={
            "values": {"startDate": "2030-01-01", "endDate": "2020-01-01"}
        })
        resp = client.get("/api/validate")
        data = resp.json()
        assert any("date" in e.lower() and ("start" in e.lower() or "end" in e.lower() or "after" in e.lower() or "before" in e.lower()) for e in data["errors"]), \
            f"Expected date-related error, got: {data['errors']}"


# ===================================================================
# 6. GET /api/config/yaml returns YAML text
# ===================================================================

class TestGetConfigYaml:
    def test_status_ok(self, client):
        resp = client.get("/api/config/yaml")
        assert resp.status_code == 200

    def test_content_type_is_text(self, client):
        resp = client.get("/api/config/yaml")
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_parseable_yaml(self, client):
        resp = client.get("/api/config/yaml")
        parsed = yaml.safe_load(resp.text)
        assert isinstance(parsed, dict)

    def test_yaml_has_sales_section(self, client):
        resp = client.get("/api/config/yaml")
        parsed = yaml.safe_load(resp.text)
        assert "sales" in parsed


# ===================================================================
# 7. POST /api/config/yaml accepts YAML text
# ===================================================================

class TestPostConfigYaml:
    def test_accept_valid_yaml(self, client):
        # Get current YAML, modify, and post back
        resp = client.get("/api/config/yaml")
        current = yaml.safe_load(resp.text)
        current.setdefault("defaults", {}).setdefault("dates", {})
        current["defaults"]["seed"] = 123

        new_text = yaml.safe_dump(current, sort_keys=False)
        resp = client.post("/api/config/yaml", json={"yaml_text": new_text})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_reject_invalid_yaml(self, client):
        resp = client.post("/api/config/yaml", json={"yaml_text": "{[bad yaml"})
        assert resp.status_code == 400

    def test_reject_non_mapping(self, client):
        resp = client.post("/api/config/yaml", json={"yaml_text": "- list\n- items\n"})
        assert resp.status_code == 400

    def test_roundtrip_preserves_data(self, client):
        # Post a known config
        cfg = {
            "defaults": {"seed": 777, "dates": {"start": "2024-06-01", "end": "2024-12-31"}},
            "sales": {"file_format": "csv", "total_rows": 500},
        }
        resp = client.post("/api/config/yaml", json={"yaml_text": yaml.safe_dump(cfg)})
        assert resp.status_code == 200

        # Read it back
        resp = client.get("/api/config/yaml")
        parsed = yaml.safe_load(resp.text)
        assert parsed["defaults"]["seed"] == 777


# ===================================================================
# 8. GET /api/models returns models YAML
# ===================================================================

class TestGetModels:
    def test_status_ok(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200

    def test_content_type_is_text(self, client):
        resp = client.get("/api/models")
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_parseable_yaml(self, client):
        resp = client.get("/api/models")
        parsed = yaml.safe_load(resp.text)
        assert isinstance(parsed, dict)

    def test_has_models_key(self, client):
        resp = client.get("/api/models")
        parsed = yaml.safe_load(resp.text)
        assert "models" in parsed

    def test_models_has_quantity(self, client):
        resp = client.get("/api/models")
        parsed = yaml.safe_load(resp.text)
        models = parsed.get("models", {})
        assert "quantity" in models


# ===================================================================
# 9. POST /api/models accepts updated YAML
# ===================================================================

class TestPostModels:
    def test_accept_valid_models_yaml(self, client):
        models = {
            "models": {
                "quantity": {"base_poisson_lambda": 2.0, "min_qty": 1, "max_qty": 10},
                "pricing": {"inflation": {"annual_rate": 0.04}},
            }
        }
        resp = client.post("/api/models", json={"yaml_text": yaml.safe_dump(models)})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_reject_non_mapping(self, client):
        resp = client.post("/api/models", json={"yaml_text": "- a\n- b\n"})
        assert resp.status_code == 400

    def test_reject_invalid_yaml(self, client):
        resp = client.post("/api/models", json={"yaml_text": "{{bad yaml"})
        assert resp.status_code == 400


# ===================================================================
# 10. Versioned routes (/v1/api/...) work identically
# ===================================================================

class TestVersionedRoutes:
    def test_v1_config(self, client):
        resp = client.get("/v1/api/config")
        assert resp.status_code == 200
        assert "seed" in resp.json()

    def test_v1_validate(self, client):
        resp = client.get("/v1/api/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" in data

    def test_v1_presets(self, client):
        resp = client.get("/v1/api/presets")
        assert resp.status_code == 200


# ===================================================================
# 11. Favicon and root serve
# ===================================================================

class TestMiscEndpoints:
    def test_favicon(self, client):
        resp = client.get("/favicon.ico")
        assert resp.status_code == 200
        assert "svg" in resp.headers.get("content-type", "")

    def test_generate_status_idle(self, client):
        resp = client.get("/api/generate/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
