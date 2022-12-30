import streamlit as st
from multiapp import MultiApp
from apps import Home, Flood_Detection

st.set_page_config(layout="wide")


apps = MultiApp()

# Add all your application here

apps.add_app("Home", Home.app)
apps.add_app("Flood Detection", Flood_Detection.app)


# The main app
apps.run()
