import streamlit as pd
import pandas as pd 
from scikit-learn.model_selection import train_test_split
from scikit-learn.linear_model import LinearRegression
df = pd.read_cvs("student_score.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
st.title("exam score predictor")
st.write("Enter house studied to predict the exam score.")
hours = st.number_input("Hours Studied:", min_value=0.0, step=0.1)
if st.button("predict score"):
  predicted_score = model.predict([[hours]])[0]
  st.success(f"predicted Score:{predicted_score:.2f}")
st.write("### sample training data")
st.dataframe(df)
