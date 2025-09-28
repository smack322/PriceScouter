from frontend.chatbot_ui import normalize_price_str_to_float, sanitize_input_fn

def test_normalize_price_str_to_float():
    assert normalize_price_str_to_float("$12.99") == 12.99
    assert normalize_price_str_to_float(7.5) == 7.5
    assert normalize_price_str_to_float(None) is None

def test_input_empty():
    # TC-REQ-001-01
    for val in ["", " "]:
        result = sanitize_input_fn(val)
        assert result["error"] == "INPUT_EMPTY"
        assert "user-facing message" in result["message"].lower()
        assert result["forwarded"] is False

def test_input_too_long():
    # TC-REQ-001-02
    long_input = "a" * 300
    result = sanitize_input_fn(long_input)
    # Configuration-dependent
    assert result["error"] in ["INPUT_TOO_LONG", None]
    assert len(result["safe_input"]) <= 256
    assert result["forwarded"] is False

def test_happy_path_normalization():
    # TC-REQ-001-03
    inputs = ["iPhone 15", "iphone 15 case", "🍎 iphone"]
    for val in inputs:
        result = sanitize_input_fn(val)
        assert result["error"] is None
        # Normalized to lowercase, spaces collapsed
        assert result["safe_input"] == ' '.join(val.lower().split())
        assert result["forwarded"] is True
        # Emoji handling: strip or retain, test accordingly
        # Example: if emojis are stripped:
        assert "🍎" not in result["safe_input"]  # Remove if emojis are retained

def test_dangerous_input():
    # TC-REQ-001-04
    dangerous_inputs = ["<script>alert(1)</script>", "x' OR 1=1 --"]
    for val in dangerous_inputs:
        result = sanitize_input_fn(val)
        assert result["error"] == "DANGEROUS"
        assert result["forwarded"] is False
        # Optionally check for escaping or blocking
        assert "<script>" not in result["safe_input"]
        assert "'" not in result["safe_input"] or "--" not in result["safe_input"]