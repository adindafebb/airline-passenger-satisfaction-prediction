import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from joblib import dump, load

# Customized Icon
img2 = Image.open('icon plane.png')
st.set_page_config(page_title = 'Airline Passenger Satisfaction Prediction', page_icon=img2)
        

st.write("""
## Airline Passenger Satisfaction Prediction App

This application is the output of Adinda Febby Nuraini's final project as a requirement for completing undergraduate studies in Statistics at Universitas Sebelas Maret. This application is used to predict passenger **satisfaction**! 

The predictions are built from the Random Forest machine learning model which has been trained on more than 62,000 datasets to achieve an accuracy of 96.45% and an F1-Score of 96.89%.
""")

img = Image.open('plane.jpg')
img = img.resize((700,418))
st.image(img, use_column_width = False)

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Age = st.sidebar.slider('Age (tahun)', 7, 90 , 28)
        Flight_Distance = st.sidebar.slider('Flight Distance (mil)', 30, 5000, 445)
        Departure_Delay = st.sidebar.slider('Departure Delay (minutes)', 0, 800, 15)
        Arrival_Delay = st.sidebar.slider('Arrival Delay (minutes)', 0, 800, 15)
        Departure_and_Arrival_Time_Convenience = st.sidebar.slider('Departure and Arrival Time Convenience', 1, 5, 3)
        Ease_of_Online_Booking = st.sidebar.slider('Ease of Online Booking', 1, 5, 3)
        Checkin_Service = st.sidebar.slider('Check-in Service', 1, 5, 3)
        Online_Boarding = st.sidebar.slider('Online Boarding', 1, 5, 3)
        Gate_Location = st.sidebar.slider('Gate Location', 1, 5, 3)
        Onboard_Service = st.sidebar.slider('On-board Service', 1, 5, 3)
        Seat_Comfort = st.sidebar.slider('Seat Comfort', 1, 5, 3)
        Leg_Room_Service = st.sidebar.slider('Leg Room Service', 1, 5, 3)
        Cleanliness = st.sidebar.slider('Cleanliness', 1, 5, 3)
        Food_and_Drink = st.sidebar.slider('Food and Drink', 1, 5, 3)
        Inflight_Service = st.sidebar.slider('In-flight Service', 1, 5, 3)
        Inflight_Wifi_Service = st.sidebar.slider('In-flight Wifi Service', 1, 5, 3)
        Inflight_Entertainment = st.sidebar.slider('In-flight Entertainment', 1, 5, 3)
        Baggage_Handling = st.sidebar.slider('Baggage Handling', 1, 5, 3)
        Gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
        Customer_Type = st.sidebar.selectbox('Customer Type', ('First-time','Returning'))
        Type_of_Travel = st.sidebar.selectbox('Type of Travel',('Business', 'Personal'))
        Class = st.sidebar.selectbox('Class', ('Business', 'Economy', 'Economy Plus'))
        
        data = {'Age' : Age,
                'Flight Distance' : Flight_Distance,
                'Departure Delay' : Departure_Delay,
                'Arrival Delay' : Arrival_Delay,
                'Departure and Arrival Time Convenience' : Departure_and_Arrival_Time_Convenience,
                'Ease of Online Booking' : Ease_of_Online_Booking,
                'Check-in Service' : Checkin_Service,
                'Online Boarding' : Online_Boarding,
                'Gate Location': Gate_Location,
                'On-board Service': Onboard_Service,
                'Seat Comfort': Seat_Comfort,
                'Leg Room Service': Leg_Room_Service,
                'Cleanliness': Cleanliness,
                'Food and Drink': Food_and_Drink,
                'In-flight Service': Inflight_Service,
                'In-flight Wifi Service': Inflight_Wifi_Service,
                'In-flight Entertainment': Inflight_Entertainment,
                'Baggage Handling': Baggage_Handling,
                'Gender' : Gender,
                'Customer Type': Customer_Type,
                'Type of Travel' : Type_of_Travel,
                'Class': Class}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

    
airline_raw = pd.read_csv('airline_cleaned_fix.csv')
airline = airline_raw.drop(columns=['Satisfaction'])
df = pd.concat([input_df,airline],axis=0)

# Encoding of ordinal features
encode = ['Gender','Customer Type','Type of Travel','Class']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]               
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using input parameters (shown below).')
    st.write(df)
    

# Reads in saved classification model
load_rfc = load('model_fix_skripsi.joblib')  


# Apply model to make predictions
prediction = load_rfc.predict(df)
prediction_proba = load_rfc.predict_proba(df)

st.subheader('Class Label')
st.write(pd.DataFrame({
    'Satisfaction': ['Neutral or Dissatisfied','Satisfied']
}))

airline_species = np.array(['Neutral or Dissatisfied','Satisfied'])


st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write()
if airline_species[prediction]=="Satisfied":
    st.write('The passenger is more likely to be Satisfied for airline service with probability', round(prediction_proba[0,1]*100, 2),'%.')
else:
     st.write('The passenger is more likely to be Neutral or Dissatisfied for airline service with probability', round(prediction_proba[0,0]*100, 2),'%.')


# Remove made by streamlit
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.write("""
contact : adindafebb.2002@gmail.com""")
