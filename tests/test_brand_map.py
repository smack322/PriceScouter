from comparison.brand_map import canonicalize_brand, same_brand, display_brand

def test_alias_pg_to_procter_gamble():
    assert canonicalize_brand("P&G") == "procter gamble"
    assert canonicalize_brand("Procter & Gamble, Inc.") == "procter gamble"
    assert same_brand("P. & G.", "Procter and Gamble")

def test_strip_legal_suffixes_and_punct():
    assert canonicalize_brand("Apple Inc.") == "apple"
    assert canonicalize_brand("Nestlé S.A.") == "nestle"
    assert canonicalize_brand("3M™") == "3m"

def test_vendor_override_example():
    # If you later configure overrides like {"amazon": {"hewlett packard": "hp"}}
    assert canonicalize_brand("Hewlett-Packard", vendor="amazon") in {"hp", "hewlett packard"}

def test_display_brand_pretty_case():
    assert display_brand("hp") == "HP"
    assert display_brand("procter gamble") == "Procter & Gamble"

def test_accent_and_abbrev_fold():
    assert canonicalize_brand("Nestlé S.A.") == "nestle"
    assert canonicalize_brand("Nestle S. A.") == "nestle"
    assert canonicalize_brand("Café™ S.A.") == "cafe"