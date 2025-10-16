import pandas as pd
import pickle

from fastapi import FastAPI

app = FastAPI()

dbfile = open('LogisticRegression.pickle', 'rb')
model = pickle.load(dbfile)



@app.get("/sentiment")
def read_item(news: str):
    news_data= {'predict_sentiment':[news]}
    news_data_df= pd.DataFrame(news_data)
    df = pd.DataFrame({
        'news':[news],
    })
    
    result = model.predict(news_data_df['predict_sentiment'])[0]
   
    if int(result) == 0:
        Sentiment = "Negative" 
    else:
        Sentiment= "Positive"
    
    return {"sentiment": Sentiment}
    