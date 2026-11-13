"""Keep small settings in the analyst's browser (localStorage), e.g. the Qwen link and login.

Values live in this browser only, under keys prefixed "simguard:". They survive page reloads and
restarts of the app and the API, and are never written to the server's disk.
"""

from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component(
    "browser_store", path=str(Path(__file__).resolve().parent / "components" / "browser_store")
)


def browser_store(action, key, value=None, nonce=0, widget_key=None):
    """Run get / set / clear in the browser. Returns {"nonce", "value", "ok"} once the browser
    has answered, or None on the very first run (the browser answers on the next rerun)."""
    return _component(action=action, store_key=key, value=value, nonce=nonce, default=None,
                      key=widget_key or f"browser_store_{key}")
