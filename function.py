import pandas as pd
import numpy as np 



def dataset(pclass, sex, age, sibsp, parch, fare, embarked):
    observation = {"pclass": [pclass], "sex": [sex], "age": [age],
                   "sibsp": [sibsp], "parch": [parch],
                   "fare": [fare], "embarked": [embarked]}

    dataframe = pd.DataFrame(data = observation)
    
    return dataframe


    