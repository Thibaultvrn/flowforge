"""Standalone Streamlit page registered by st.navigation in app.py."""

from pathlib import Path
import sys

_UI_DIR = Path(__file__).resolve().parent
if str(_UI_DIR) not in sys.path:
    sys.path.insert(0, str(_UI_DIR))

from components.methodology import render

render()
