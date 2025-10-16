import pandas as pd
import pickle
import streamlit as st


st.title("Sentiment Analysis App ")

dbfile = open('LogisticRegression.pickle', 'rb')
model = pickle.load(dbfile)

#input form

text = st.text_input("Enter your review = ")
review_data = {'predict_sentiment':[text]}
review_data_df = pd.DataFrame(review_data)


if st.button("Predict"):
        
    df = pd.DataFrame({
           'review': [text], 
           	       })
    st.dataframe(df)
    
    result = model.predict(review_data_df['predict_sentiment'])[0]
    if int(result) == 0:
        result = "Negative"
    else:
        result = "Positive" 
    print(result)
    st.write(result)
    
    st.write("Success!")
    st.balloons()
    
    