import streamlit as st
import  numpy as np

import pandas as pd


st.title("Sample App")

st.text("Sample text for the app")

#
# st.dataframe(pd.DataFrame({
#     '1st_Col':[1,2,4,5,6]
#     ,'2nd_col':[352,6,32,6,2]
# }))
#
#
# chart_data = pd.DataFrame(
#     np.random.randn(20,3),
#     columns=['1st','2nd','3rd']
# )
#
# st.line_chart(chart_data)
#
#
# map_data = pd.DataFrame(
#     np.random.randn(1000,2) / [50,50] + [37.76,-122.3],
#     columns=['lat','lon']
#
# )
#
# st.map(map_data)

if st.checkbox('Show Details'):
    st.dataframe(pd.DataFrame({
         '1st_Col':[1,2,4,5,6]
         ,'2nd_col':[352,6,32,6,2]
     }))


df = pd.DataFrame(np.random.randn(20,3),columns=["a","b","c"])

col = st.selectbox('Select the Columns to be Displayed',df.columns)

st.line_chart(df[col])



df = pd.DataFrame(np.random.randn(20,3),columns=["a","b","c"])

cols = st.multiselect('Select the Columns to be Displayed',df.columns)

st.line_chart(df[cols])

