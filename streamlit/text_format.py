import streamlit as st

def upload_success(url):
    st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:14px;border-radius:2%;">{url}</p>',
                unsafe_allow_html=True)

def awaiting_csv(url):
    st.markdown(f'<p style="background-color:#FF0000;color:#FFFFFF;font-size:14px;border-radius:2%;">{url}</p>',
                unsafe_allow_html=True)