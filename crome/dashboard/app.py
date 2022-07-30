import streamlit as st

from crome.dashboard.views import explore_dataset_view, explore_models_view, detect_incidents_view

st.set_page_config(page_title='CROMEx', layout='wide', initial_sidebar_state='auto', menu_items=None)

page_names_to_funcs = {
    'Explore Dataset': explore_dataset_view,
    'Explore Models': explore_models_view,
    'Real-time Incident Detection': detect_incidents_view,
}

with st.sidebar:
    st.selectbox('Select Page', page_names_to_funcs.keys(), key='page_name')

page_names_to_funcs[st.session_state.page_name]()
