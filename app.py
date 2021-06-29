from numpy.core.fromnumeric import size
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Used Car Price Prediction")

data = pd.read_csv("data\Car_Data.csv")

lr = LinearRegression()
Z = data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lr.fit(Z, data['price'])



nav = st.sidebar.radio("Navigation",["Home","Prediction"])

if nav == "Home":
    st.image("data\car.jpg",width= 700)
    if st.checkbox("Show Dataset"):
        st.dataframe(data)
    
    st.header("Want to know how the feature effects the Price ?")
    feature = st.selectbox("Select a feature", ["Horsepower","Curb-weight","Engine-size","Highway-mpg"])
    
    if feature == "Horsepower":
        st.subheader("Horsepower vs Price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["horsepower"],data["price"])
        plt.ylim(0)
        plt.xlabel("Horsepower")
        plt.ylabel("Price")
        st.pyplot(ax)

    elif feature == "Curb-weight":
        st.subheader("Curb-weight vs Price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["curb-weight"],data["price"])
        plt.ylim(0)
        plt.xlabel("Curb-weight")
        plt.ylabel("Price")
        st.pyplot(ax)

    elif feature == "Engine-size":
        st.subheader("Engine-size vs Price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["engine-size"],data["price"])
        plt.ylim(0)
        plt.xlabel("Engine-size")
        plt.ylabel("Price")
        st.pyplot(ax)

    elif feature == "Highway-mpg":
        st.subheader("Highway-mpg vs Price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["highway-mpg"],data["price"])
        plt.ylim(0)
        plt.xlabel("Highway-mpg")
        plt.ylabel("Price")
        st.pyplot(ax)            

    st.header("Model Development")
    st.markdown("**Prediction Model -** _Multiple Linear Regression_")
    st.markdown("**Features Used:**")
    st.markdown("* _Horsepower_")
    st.markdown("* _Curb-weight_")
    st.markdown("* _Engine-size_")
    st.markdown("* _Highway-mpg_")
    st.markdown("**Accuracy -** 84.12%")


elif nav == "Prediction":
    st.image("data\money.jpg",width= 500)
    st.header("Know your Car Price")
    val1 = st.number_input("Enter your Car's Horsepower",50,200,step = 3)
    val2 = st.number_input("Enter your Car's Curb-weight (pounds)",1500,4000,step = 5)
    val3 = st.number_input("Enter your Car's Engine-size (cc)",50,350,step = 2)
    val4 = st.number_input("Enter your Car's Highway-mpg",15,45,step = 1)

    val = np.array([val1,val2,val3,val4]).reshape(1,-1)

    pred = lr.predict(val)[0]
    if st.button("Predict"):
        st.success(f"Your predicted car price is $ {pred}")


