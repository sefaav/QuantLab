"""Strategy Explorer gallery: a card per registered strategy.

Always driven by ``available_strategies()``, never a hard-coded list -- a
strategy without a registered profile still gets a card (with a
"documentation coming soon" placeholder) instead of silently disappearing.

Every card is the same fixed height (so a short overview and a long one
produce identically-sized cards), with a visible "Open" button that
navigates to that strategy's detail page. An earlier design instead
stretched an invisible ``st.button`` to cover the whole card via absolute
positioning -- unreliable across Streamlit versions/themes since it
depended on the exact DOM structure of Streamlit-internal containers, not
a documented API -- a plain, visible button is simpler and actually works.

The fixed height is applied via this CSS, NOT ``st.container``'s own
``height=`` parameter: that parameter always renders as a scrollable region
(``overflow-y: auto``) regardless of whether the content actually overflows,
which is exactly the stray scrollbar this design avoids -- ``overflow:
hidden`` here clips instead of scrolling (the truncated summary text below
is already sized to fit, so clipping is not expected to ever trigger in
practice).
"""

from __future__ import annotations

from typing import Any

_CARD_HEIGHT = 230
_SUMMARY_CHAR_LIMIT = 150

_CARD_HEIGHT_CSS = f"""
<style>
div[class*="st-key-explorer_card_"] {{
    height: {_CARD_HEIGHT}px;
    overflow: hidden;
}}
</style>
"""


def _truncate(text: str, limit: int) -> str:
    """Truncate on a word boundary.

    Keeps every card's summary the same rough length regardless of how
    long that strategy's overview is.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:-") + "..."


def render(st: Any) -> None:
    """Render the gallery of strategy cards."""
    from quantlab.dashboard.explorer.profile import get_profile
    from quantlab.strategies.base import available_strategies

    st.subheader("Strategies")
    st.caption(
        "Pick a strategy to explore its economics, mathematics, "
        "assumptions, diagnostics and an interactive research lab."
    )
    st.html(_CARD_HEIGHT_CSS)
    strategies = available_strategies()
    columns = st.columns(3)
    for index, name in enumerate(strategies):
        profile = get_profile(name)
        column = columns[index % 3]
        with (
            column,
            st.container(key=f"explorer_card_{name}", border=True),
        ):
            if profile is not None:
                st.markdown(f"##### {profile.display_name}")
                st.caption(profile.category)
                summary = profile.overview_md.strip().split("\n\n")[0]
                st.write(_truncate(summary, _SUMMARY_CHAR_LIMIT))
            else:
                st.markdown(f"##### {name}")
                st.caption("Documentation coming soon.")
            if st.button("Open", key=f"explorer_open_{name}", width="stretch"):
                st.session_state["explorer_strategy"] = name
                st.rerun()
