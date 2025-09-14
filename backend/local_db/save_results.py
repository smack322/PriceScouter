from typing import Iterable, Mapping
from sqlalchemy import select
from db import SessionLocal, Vendor, Product, Offer, Search, SearchResult

def to_cents(price_float_or_none):
    if price_float_or_none is None:
        return None
    return int(round(float(price_float_or_none) * 100))

def upsert_vendor(session, name, site_url=None):
    v = session.execute(select(Vendor).where(Vendor.name == name)).scalar_one_or_none()
    if v: return v
    v = Vendor(name=name, site_url=site_url)
    session.add(v); session.flush()
    return v

def upsert_product(session, title, brand=None, category=None, gtin=None, asin=None):
    q = None
    if asin:
        q = select(Product).where(Product.asin == asin)
    elif gtin:
        q = select(Product).where(Product.gtin == gtin)
    if q:
        p = session.execute(q).scalar_one_or_none()
        if p: return p
    p = Product(title=title, brand=brand, category=category, gtin=gtin, asin=asin)
    session.add(p); session.flush()
    return p

def save_search_with_offers(query: str, provider: str, items: Iterable[Mapping]):
    """
    items: iterable of dicts with normalized fields:
      title, price_value(float), source(vendor name), rating, link(url),
      brand, category, gtin, asin, shipping_value(float), raw_payload(dict)
    """
    with SessionLocal() as s:
        srch = Search(query=query, provider=provider)
        s.add(srch); s.flush()
        for rank, it in enumerate(items, start=1):
            vendor = upsert_vendor(s, it.get("source") or "Unknown")
            product = upsert_product(
                s, title=it.get("title") or "Untitled",
                brand=it.get("brand"), category=it.get("category"),
                gtin=it.get("gtin"), asin=it.get("asin")
            )
            offer = Offer(
                product_id=product.id,
                vendor_id=vendor.id,
                price_cents=to_cents(it.get("price_value")),
                shipping_cents=to_cents(it.get("shipping_value")),
                currency="USD",
                rating=it.get("rating"),
                url=it.get("link"),
                source=provider,
                raw_payload=it.get("raw_payload"),
            )
            s.add(offer); s.flush()
            s.add(SearchResult(search_id=srch.id, offer_id=offer.id, rank=rank))
        s.commit()
