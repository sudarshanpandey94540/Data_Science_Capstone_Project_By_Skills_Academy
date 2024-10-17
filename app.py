import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the pre-trained model
model = pickle.load(open('app.pkl', 'rb'))

# Load the dataset with cleaned car details
df = pd.read_csv('Car_Details_Cleaned_Dataset.csv')

# List of categorical columns in the dataset
categorical_columns = ['Car_Brand', 'Car_Name', 'Fuel', 'Seller_Type', 'Transmission', 'Owner']

# Function to encode categorical data
def encode_data(input_df, encoders):
    for column in input_df.columns:
        if column in encoders:
            input_df[column] = encoders[column].transform(input_df[column])
    return input_df

# Prepare LabelEncoders based on the original training data
label_encoders = {}
for col in categorical_columns:
    encoder = LabelEncoder()
    encoder.fit(df[col])
    label_encoders[col] = encoder

# CSS for custom header design with a background image
header_image_url = "https://github.com/vishal-verma-96/Capstone_Project_By_Skill_Academy/blob/main/automotive.jpg?raw=true"
st.markdown(
    f"""
    <style>
    .custom-header {{
        background-image: url('{header_image_url}');
        background-size: cover;
        height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.7);
        margin-bottom: 40px;
    }}
    .custom-header h1 {{
        font-size: 3em;
    }}
    </style>
    <div class="custom-header">
        <h1>Car Price Estimator</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar instructions for user guidance
st.sidebar.markdown("""
## Instructions:
1. Select the car details (brand, model, fuel type, etc.).
2. Adjust the sliders for year of purchase and kilometers driven.
3. Hit the "Predict Selling Price" button to see the estimated car price.
""")

# Encode the entire dataset for consistency with model training
encoded_df = encode_data(df.copy(), label_encoders)

# Input widgets for user to select car features
kilometers_driven = st.slider("Kilometers Driven:", min_value=int(df["Km_Driven"].min()), max_value=int(df["Km_Driven"].max()), step=1000)
year_of_purchase = st.slider("Year of Purchase:", min_value=int(df["Year"].min()), max_value=int(df["Year"].max()), step=1)

# Dropdowns and radio buttons for categorical features
selected_brand = st.selectbox("Car Brand:", df["Car_Brand"].unique())
filtered_models = df[df['Car_Brand'] == selected_brand]["Car_Name"].unique()
selected_model = st.selectbox("Car Model:", filtered_models)
selected_fuel = st.radio("Fuel Type:", df["Fuel"].unique())
selected_seller = st.radio("Seller Type:", df["Seller_Type"].unique())
selected_transmission = st.radio("Transmission:", df["Transmission"].unique())
selected_owner_type = st.radio("Owner Type:", df["Owner"].unique())

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'Car_Brand': [selected_brand],
    'Car_Name': [selected_model],
    'Year': [year_of_purchase],
    'Km_Driven': [kilometers_driven],
    'Fuel': [selected_fuel],
    'Seller_Type': [selected_seller],
    'Transmission': [selected_transmission],
    'Owner': [selected_owner_type]
})

# Encode user input using the pre-fitted label encoders
input_encoded = encode_data(input_data.copy(), label_encoders)

# Standardizing the numerical features
scaler = StandardScaler()
input_encoded[['Year', 'Km_Driven']] = scaler.fit_transform(input_encoded[['Year', 'Km_Driven']])

# Predict the car's selling price when the user clicks the button
if st.button("Predict Selling Price"):
    predicted_price = model.predict(input_encoded)
    st.subheader("Estimated Selling Price:")
    st.write(f"**₨ {predicted_price[0]:,.2f}**")
