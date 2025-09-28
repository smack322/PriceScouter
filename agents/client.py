import asyncio
from agents.app import app  # your compiled graph


async def run_agents(query: str, *, zip_code="19406", country="US", max_price: float | None = None, top_n: int = 15):
    # The graph still expects messages, but your extractor will see everything in the message.
    user_msg = {
        "role": "user",
        "content": (
            f"Query: {query}\n"
            f"Max price: {max_price}\n"
            f"ZIP: {zip_code}\nCountry: {country}"
        )
    }
    state = await app.ainvoke({"messages": [user_msg]})
    rows = (state.get("results") or [])[:top_n]
    return rows