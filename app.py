import uvicorn
from fastapi import FastAPI


import pickle
import numpy as np
import pandas as pd
from function import dataset


app = FastAPI()

pickle_model = open("titanic_ml_model.sav", "rb")
model = pickle.load(pickle_model)




#  http://127.0.0.1:8000/docs

@app.post('/predict')
def predict_is_survived(pclass: int,sex: int,age: int,sibsp: int,parch: int,fare: int,embarked: int):
    """    Pclass is a proxy for socio-economic status (SES)
0 = Upper; 1 =  Middle; 2 = Lower  

Sex: 
Male = 0 
Female = 1

Sibps: Number of Siblings/Spouses Aboard

Sibling: Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
Spouse: Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances
        Ignored)
        

Parch : Number of Parents/Children Aboard     

Parent: Mother or Father of Passenger Aboard Titanic
Child: Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

fare Passenger Fare (British pound)  0 < fare < 512

Embarked: Port of Embarkation (0 = Southampton; 1 = Cherbourg; 2 = Queenstown)
"""
    user_data = dataset(pclass, sex, age, sibsp, parch, fare, embarked)
    
    
    prediction = model.predict(user_data)
    
    if prediction == 1:
        return {"You will survive"}
    
    elif prediction == 0:
        return {"You will die :("}



if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)

