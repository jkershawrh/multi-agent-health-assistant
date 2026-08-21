"""Stage 0: Contract validation — OpenAPI specs parse and refs resolve."""
import pathlib
import sys

import pytest
import yaml
from openapi_spec_validator import validate

CONTRACTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "contracts" / "openapi"
SRC_DIR = pathlib.Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

import agent  # noqa: E402
import orchestrator  # noqa: E402

APPS = {
    "agent.yaml": agent.app,
    "orchestrator.yaml": orchestrator.app,
}


def _load_specs():
    if not CONTRACTS_DIR.exists():
        pytest.skip("No contracts/openapi/ directory")
    specs = list(CONTRACTS_DIR.glob("*.yaml")) + list(CONTRACTS_DIR.glob("*.yml"))
    if not specs:
        pytest.skip("No OpenAPI specs found")
    return specs


@pytest.fixture(params=[s.name for s in _load_specs()] if CONTRACTS_DIR.exists() else [])
def spec_path(request):
    return CONTRACTS_DIR / request.param


@pytest.fixture
def spec(spec_path):
    return yaml.safe_load(spec_path.read_text())


class TestOpenAPIContractValidation:

    @pytest.fixture(autouse=True)
    def _specs(self):
        self.specs = _load_specs()

    @pytest.mark.parametrize("spec_file", [s.name for s in _load_specs()] if CONTRACTS_DIR.exists() else [])
    def test_spec_parses(self, spec_file):
        spec = yaml.safe_load((CONTRACTS_DIR / spec_file).read_text())
        assert "openapi" in spec or "swagger" in spec, f"{spec_file} missing openapi version"
        validate(spec)

    @pytest.mark.parametrize("spec_file", [s.name for s in _load_specs()] if CONTRACTS_DIR.exists() else [])
    def test_spec_has_info(self, spec_file):
        spec = yaml.safe_load((CONTRACTS_DIR / spec_file).read_text())
        assert "info" in spec, f"{spec_file} missing info block"
        assert "title" in spec["info"], f"{spec_file} missing info.title"

    @pytest.mark.parametrize("spec_file", [s.name for s in _load_specs()] if CONTRACTS_DIR.exists() else [])
    def test_spec_has_paths(self, spec_file):
        spec = yaml.safe_load((CONTRACTS_DIR / spec_file).read_text())
        assert "paths" in spec, f"{spec_file} missing paths"
        assert len(spec["paths"]) > 0, f"{spec_file} has no path definitions"

    @pytest.mark.parametrize("spec_file", [s.name for s in _load_specs()] if CONTRACTS_DIR.exists() else [])
    def test_all_operations_have_responses(self, spec_file):
        spec = yaml.safe_load((CONTRACTS_DIR / spec_file).read_text())
        for path, methods in spec.get("paths", {}).items():
            for method, operation in methods.items():
                if method.startswith("x-") or method == "parameters":
                    continue
                assert "responses" in operation, (
                    f"{spec_file}: {method.upper()} {path} missing responses"
                )

    @pytest.mark.parametrize("spec_file", [s.name for s in _load_specs()] if CONTRACTS_DIR.exists() else [])
    def test_schema_refs_resolve(self, spec_file):
        text = (CONTRACTS_DIR / spec_file).read_text()
        spec = yaml.safe_load(text)
        components = spec.get("components", {}).get("schemas", {})

        def _find_refs(obj, path=""):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref = obj["$ref"]
                    if ref.startswith("#/components/schemas/"):
                        schema_name = ref.split("/")[-1]
                        assert schema_name in components, (
                            f"{spec_file}: unresolved $ref {ref} at {path}"
                        )
                for k, v in obj.items():
                    _find_refs(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _find_refs(item, f"{path}[{i}]")

        _find_refs(spec)

    @pytest.mark.parametrize("spec_file", APPS)
    def test_paths_match_running_application(self, spec_file):
        spec = yaml.safe_load((CONTRACTS_DIR / spec_file).read_text())
        assert set(spec["paths"]) == set(APPS[spec_file].openapi()["paths"])

    def test_workflow_request_limits_match_running_application(self):
        spec = yaml.safe_load((CONTRACTS_DIR / "orchestrator.yaml").read_text())
        committed = spec["components"]["schemas"]["WorkflowRequest"]["properties"]
        generated = orchestrator.app.openapi()["components"]["schemas"][
            "WorkflowRequest"
        ]["properties"]
        for field, constraints in {
            "query": ("minLength", "maxLength"),
            "workflow_type": ("enum",),
        }.items():
            for constraint in constraints:
                assert committed[field].get(constraint) == generated[field].get(constraint)

    def test_json_rpc_envelope_is_bounded_and_unambiguous(self):
        spec = yaml.safe_load((CONTRACTS_DIR / "agent.yaml").read_text())
        schemas = spec["components"]["schemas"]
        request_id = schemas["JsonRpcRequest"]["properties"]["id"]
        response = schemas["JsonRpcResponse"]

        assert request_id["minLength"] == 1
        assert request_id["maxLength"] == 128
        assert response["properties"]["jsonrpc"]["const"] == "2.0"
        assert response["oneOf"] == [
            {"required": ["result"]},
            {"required": ["error"]},
        ]
