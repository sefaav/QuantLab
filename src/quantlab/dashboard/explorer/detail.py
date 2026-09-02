"""Strategy Explorer detail page: full profile + interactive lab.

Section layout intentionally merges "Mathematical definition" and
"Signals" into one expander -- every bundled profile documents its signal
pipeline as part of the same walk-through, so a separate, identically
worded section would just be a duplicate, not new information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quantlab.dashboard.explorer.profile import ParameterDoc


def render(st: Any, strategy_name: str) -> None:
    """Render the detail page for one strategy, or a fallback if unregistered."""
    from quantlab.dashboard.explorer.profile import get_profile

    if st.button("<- Back to gallery", key="explorer_back"):
        st.session_state.pop("explorer_strategy", None)
        st.rerun()
        return

    profile = get_profile(strategy_name)
    if profile is None:
        st.warning(
            f"No Strategy Explorer content is registered yet for "
            f"'{strategy_name}'. It is still fully usable in Backtest/"
            "Walk-forward mode -- only this page's documentation is "
            "missing."
        )
        return

    st.title(profile.display_name)
    st.caption(profile.category)

    with st.expander("Overview", expanded=True):
        st.markdown(profile.overview_md)
    with st.expander("Economic intuition"):
        st.markdown(profile.economic_intuition_md)
    with st.expander("Mathematical definition & signals"):
        st.markdown(profile.mathematical_definition_md)
    with st.expander("Assumptions"):
        st.markdown(profile.assumptions_md)
    with st.expander("Parameters"):
        _render_parameters(st, profile.parameters)
    with st.expander("Diagnostics"):
        st.markdown(profile.diagnostics_md)
    # A plain `st.expander` still runs its body every rerun even while
    # collapsed -- this one does real work (data loads, OLS fits, ADF/
    # cointegration tests, chart builds), so it uses the stateful/lazy
    # variant (`key` + `on_change="rerun"`) instead: `.open` reports
    # whether it is actually expanded, and the lab only runs then. Simply
    # visiting this page (or interacting with any OTHER widget on it) no
    # longer silently re-triggers the lab's full computation.
    lab_expander = st.expander(
        "Interactive laboratory",
        key=f"explorer_lab_expander_{strategy_name}",
        on_change="rerun",
    )
    if lab_expander.open:
        with lab_expander:
            profile.lab(st)
    with st.expander("Interpretation"):
        st.markdown(profile.interpretation_md)
    with st.expander("Limitations & failure modes"):
        st.markdown(profile.limitations_md)
    if profile.references_md:
        with st.expander("References / Further reading"):
            st.markdown(profile.references_md)


def _render_parameters(st: Any, parameters: list[ParameterDoc]) -> None:
    if not parameters:
        st.caption("This strategy has no configurable parameters.")
        return
    for index, parameter in enumerate(parameters):
        st.markdown(f"**`{parameter.name}`** -- default: `{parameter.default}`")
        st.markdown(
            f"- **What**: {parameter.what}\n"
            f"- **Where**: {parameter.where}\n"
            f"- **Why**: {parameter.why}\n"
            f"- **Typical range**: {parameter.typical_range}\n"
            f"- **Increasing it**: {parameter.effect_increase}\n"
            f"- **Decreasing it**: {parameter.effect_decrease}\n"
            f"- **Trade-offs**: {parameter.tradeoffs}"
        )
        if parameter.interactions:
            st.markdown(f"- **Interactions**: {parameter.interactions}")
        if index < len(parameters) - 1:
            st.divider()
