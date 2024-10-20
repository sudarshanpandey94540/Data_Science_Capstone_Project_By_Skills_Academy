import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('app.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the cleaned data
cleaned_data = pd.read_csv('Car_Details_Cleaned_Dataset.csv')

# Define the categorical columns
category_col = ['Car_Brand', 'Car_Name', 'Fuel', 'Seller_Type', 'Transmission', 'Owner']

# Function for encoding data
def preprocess_data(df):
    for feature in category_col:
        if feature in df.columns:
            df[feature] = label_encoders[feature].transform(df[feature])
    return df

# Add CSS for the background image
image_path = "https://raw.githubusercontent.com/vishal-verma-96/Capstone_Project_By_Skill_Academy/main/new_background_image.jpg"  # Replace with your new image URL
st.markdown(
    f"""
    <style>
    .header {{
        background-image: url('{image_path}');
        background-size: cover;
        background-position: center;
        height: 200px; 
        opacity: 0.9; 
        position: relative; 
        z-index: 1; 
        display: flex; 
        align-items: center; 
        padding: 0 60px;}}
    .header h1 {{
        color: #f0f0f0; 
        margin: 0; 
        padding: 25px; 
        text-align: left; 
        font-size: 2.5em; 
        flex: 1;}}
    .body-content {{
        margin-top: 30px;
    }}
    </style>
    <div class="header">
        <h1><i>Smart Car Price Estimator</i></h1>
    </div>
    <div class="body-content">
    """,
    unsafe_allow_html=True
)

# Sidebar for input
st.sidebar.markdown("""
This tool estimates the resale value of a car based on its attributes.
### How to use:
1. **Provide Car Details:** Input the car's specifications via the options provided.
2. **Estimate Price:** Click the 'Estimate Price' button to view the resale price.
""")

# Input sliders and dropdowns
km_driven = st.slider("How many kilometers has the car been driven?", min_value=int(cleaned_data["Km_Driven"].min()),
                       max_value=int(cleaned_data["Km_Driven"].max()))
year = st.slider("Year the car was purchased:", min_value=int(cleaned_data["Year"].min()), max_value=int(cleaned_data["Year"].max()))
selected_brand = st.selectbox("Choose the car's brand:", cleaned_data["Car_Brand"].unique())
brand_filtered_df = cleaned_data[cleaned_data['Car_Brand'] == selected_brand]
selected_model = st.selectbox("Choose the car's model:", brand_filtered_df["Car_Name"].unique())
selected_fuel = st.radio("Select the type of fuel:", cleaned_data["Fuel"].unique())
selected_seller_type = st.radio("What type of seller are you?", cleaned_data["Seller_Type"].unique())
selected_transmission = st.radio("Transmission type:", cleaned_data["Transmission"].unique())
selected_owner = st.radio("Number of previous owners:", cleaned_data["Owner"].unique())

# Create a DataFrame from the user inputs
input_data = pd.DataFrame({
    'Car_Brand': [selected_brand],
    'Car_Name': [selected_model],
    'Year': [year],
    'Km_Driven': [km_driven],
    'Fuel': [selected_fuel],
    'Seller_Type': [selected_seller_type],
    'Transmission': [selected_transmission],
    'Owner': [selected_owner]
})

# Preprocess the user input data
input_data_encoded = preprocess_data(input_data.copy())

# Standardize numerical features
input_data_encoded[['Year', 'Km_Driven']] = scaler.transform(input_data_encoded[['Year', 'Km_Driven']])

# Make prediction using the loaded model
if st.button("Estimate Price"):
    predicted_price = loaded_model.predict(input_data_encoded)
    st.subheader("Estimated Resale Price:")
    st.write(f"The estimated resale price is: **_${predicted_price[0]:,.2f}_**")
    
    # Add an image for car price prediction
    st.image("https://your-new-image-url.com/car_image.jpg", caption="Your Future Car!", use_column_width=True)  # Replace with your image URL
