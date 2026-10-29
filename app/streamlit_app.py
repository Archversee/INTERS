"""Sim-to-SOP app.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
import ui
from views import demo, explorer, generator, guidelines, home, progress, references_page, runs

from sim2sop import config
from sim2sop.profiles import list_profiles

st.set_page_config(page_title="Sim-to-SOP", page_icon=":material/flight_takeoff:", layout="wide")

ui.PAGES.update({
    "home": st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
    "runs": st.Page(runs.render, title="1 · Instructor flights", icon=":material/flight:", url_path="flights"),
    "references": st.Page(references_page.render, title="2 · Official documents", icon=":material/menu_book:",
                          url_path="documents"),
    "guidelines": st.Page(guidelines.render, title="3 · Guidelines", icon=":material/checklist:",
                          url_path="guidelines"),
    "progress": st.Page(progress.render, title="Progress", icon=":material/trending_up:", url_path="progress"),
    "demo": st.Page(demo.render, title="Guided demo", icon=":material/play_circle:", url_path="demo"),
    "generator": st.Page(generator.render, title="Flight generator", icon=":material/science:",
                         url_path="generator"),
    "explorer": st.Page(explorer.render, title="Flight data explorer", icon=":material/monitoring:",
                        url_path="explorer"),
})
nav = st.navigation({
    "": [ui.PAGES["home"]],
    "Workflow": [ui.PAGES[k] for k in ("runs", "references", "guidelines")],
    "See the results": [ui.PAGES["demo"], ui.PAGES["progress"]],
    "Tools": [ui.PAGES["generator"], ui.PAGES["explorer"]],
})

with st.sidebar:
    names = list_profiles()
    st.session_state.setdefault("profile_name", "air_c172_circuit" if "air_c172_circuit" in names else names[0])
    st.selectbox("Procedure", names, key="profile_name", format_func=lambda n: ui.PROFILE_NAMES.get(n, n))

    st.session_state.setdefault("llm_provider", config.LLM_PROVIDER if config.LLM_PROVIDER in ui.PROVIDERS else "none")
    st.session_state.setdefault("llm_model", config.LLM_MODEL)
    provider, model = st.session_state["llm_provider"], st.session_state["llm_model"]
    ok, models = ui.llm_status(provider, model)
    if provider == "none":
        st.caption(":material/smart_toy: **AI writer:** off — built-in template wording")
    elif ok and (provider != "ollama" or model in models):
        st.caption(f":material/smart_toy: **AI writer:** :green[connected] · {model}")
    else:
        st.caption(":material/smart_toy: **AI writer:** :red[not reachable] — using the built-in template")
    with st.popover("AI writer settings", icon=":material/settings:", width="stretch"):
        st.selectbox("Provider", list(ui.PROVIDERS), key="llm_provider", format_func=ui.PROVIDERS.get)
        if provider == "ollama" and models:
            if st.session_state["llm_model"] not in models:
                st.session_state["llm_model"] = models[0]
            st.selectbox("Model", models, key="llm_model")
            st.caption(f"Server: {config.OLLAMA_URL}")
        elif provider != "none":
            st.text_input("Model", key="llm_model")
        st.caption("The AI only words the guidelines. Every number is checked against the flight data, "
                   "and limits always come from the official documents.")

ui.show_queued()   # review popup / toast from the last action, on whatever page is open
nav.run()
