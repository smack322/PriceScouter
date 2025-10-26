import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import os
from pathlib import Path

import importlib.util
import sys
from pathlib import Path
import contextlib
import pytest
import json
import types


@pytest.fixture(autouse=True)
def _ci_env():
    # prevent network/LLM during tests
    os.environ.setdefault("DISABLE_LLM", "1")

@pytest.fixture
def parsed_default():
    return {"query": "test", "max_price": None, "zip_code": "19406", "country": "US"}

# ----- Fake Streamlit -----
class _FakeLinkColumn:
    def __init__(self, label=None, display_text=None, help=None):
        self.label = label
        self.display_text = display_text
        self.help = help

class _FakeColumnConfig:
    LinkColumn = _FakeLinkColumn

class FakeSt:
    def __init__(self):
        self.calls = {"dataframe": [], "link_button": [], "markdown": [], "caption": []}
        self.column_config = _FakeColumnConfig()

    def dataframe(self, *args, **kwargs):
        self.calls["dataframe"].append((args, kwargs))

    def link_button(self, label, url):
        self.calls["link_button"].append((label, url))

    def markdown(self, md):
        self.calls["markdown"].append(md)

    def caption(self, cap):
        self.calls["caption"].append(cap)

    @contextlib.contextmanager
    def expander(self, _title):
        yield

def _load_module_as(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find module at: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register under requested name (alias)
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture()
def fake_st(monkeypatch):
    # Adjust this path if your repo layout differs
    repo_root = Path(__file__).resolve().parents[1]   # repo root (.. from tests/)
    app_path = repo_root / "frontend" / "chatbot_ui.py"

    # Load chatbot_ui.py but expose it to tests as module name "streamlit_ui"
    ui = _load_module_as("streamlit_ui", app_path)

    fake = FakeSt()
    # Replace the 'st' object used inside the UI with our fake
    monkeypatch.setattr(ui, "st", fake, raising=True)
    yield ui, fake

def _load_env_once():
    # Load .env if present (local dev). In CI we rely on injected env vars.
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)

_load_env_once()

os.environ.setdefault("DISABLE_LLM", "1")



class _FakeLLM:
    def invoke(self, _msgs):
        class _Resp:
            content = json.dumps({"query": "fallback", "vendor": None, "limit": None})
        return _Resp()

def test_extract_params_handles_bad_json(monkeypatch):
    import agents.app as app
    monkeypatch.setattr(app, "extract_llm", _FakeLLM())  # <- no network
    state = {"messages": [DummyMessage("fallback")]}
    result = app.extract_params(state)
    assert result["query"] == "fallback"
    assert result["vendor"] is None

    AMAZON_US = "ATVPDKIKX0DER"

def make_fake_api(products):
    """
    products: list[dict] shaped like Keepa products.
    """
    class FakeAPI:
        def __init__(self, prods):
            self._prods = prods

        def product_finder(self, criteria, domain="US"):
            # We don't need to parse criteria for tests; return the ASINs we have.
            return [p.get("asin") for p in self._prods]

        def query(self, asins, domain="US", **kwargs):
            # Return only the requested ASINs in the same order.
            keep = {p.get("asin"): p for p in self._prods}
            return [keep[a] for a in asins if a in keep]

    return FakeAPI(products)