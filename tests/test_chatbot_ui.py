from frontend.chatbot_ui import normalize_price_str_to_float

def test_normalize_price_str_to_float():
    assert normalize_price_str_to_float("$12.99") == 12.99
    assert normalize_price_str_to_float(7.5) == 7.5
    assert normalize_price_str_to_float(None) is None