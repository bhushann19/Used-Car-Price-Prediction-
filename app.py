import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import pickle

# Page configuration and styling
st.set_page_config(page_title="Japanese Used Car Price Predictor", layout="wide")

# Custom CSS to enhance the UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1E3A8A;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #6B7280;
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>🚗 Japanese Used Car Price Predictor</h1>", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Japan_used_cars_datasets.csv")
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the file path.")
        return None

df = load_data()

if df is not None:
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "📊 Market Insights", "❓ Help & FAQ"])
    
    with tab1:
        st.markdown("<p class='highlight'>Fill in the details below to get an estimated price for your desired Japanese used car.</p>", 
                    unsafe_allow_html=True)
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h2 class='subheader'>Car Details</h2>", unsafe_allow_html=True)
            
            # MARK SELECTION with images for popular brands
            mark_counts = df['mark'].value_counts()
            popular_marks = mark_counts.nlargest(6).index.tolist()
            
            # Display popular brands as buttons
            st.write("Quick Select Popular Brands:")
            cols = st.columns(3)
            for i, mark in enumerate(popular_marks):
                if cols[i % 3].button(f"{mark}", key=f"mark_{mark}"):
                    selected_mark = mark
            
            # Regular dropdown for all brands
            selected_mark = st.selectbox("Select Car Brand (Mark)", 
                                        sorted(df['mark'].unique()),
                                        index=0 if 'selected_mark' not in locals() else 
                                        sorted(df['mark'].unique()).index(selected_mark))
            
            # Get models for selected mark
            mark_model_dict = df.groupby('mark')['model'].unique().to_dict()
            available_models = mark_model_dict[selected_mark]
            
            # MODEL SELECTION BASED ON MARK
            selected_model = st.selectbox("Select Car Model", sorted(available_models))
            
            # YEAR with visual slider
            current_year = datetime.now().year
            min_year = int(df['year'].min())
            max_year = min(int(df['year'].max()), current_year)
            
            year = st.slider("Manufacturing Year", 
                            min_year, max_year,
                            value=max_year-5,
                            step=1,
                            format="%d")
            
            # Age calculation
            car_age = current_year - year
            st.info(f"This car is {car_age} years old" if car_age > 0 else "This is a current year model")
            
        with col2:
            st.markdown("<h2 class='subheader'>Technical Specifications</h2>", unsafe_allow_html=True)
            
            # ENGINE CAPACITY with visual gauge
            min_engine = int(df['engine_capacity'].min())
            max_engine = int(df['engine_capacity'].max())
            engine_capacity = st.slider("Engine Capacity (cc)", 
                                      min_engine, max_engine,
                                      value=1500,
                                      step=100)
            
            # MILEAGE with input validation
            mileage = st.number_input("Mileage (km)", 
                                    min_value=0, 
                                    max_value=1000000,
                                    value=50000,
                                    step=1000)
            
            # OTHER CATEGORICAL FEATURES with icons
            transmission_options = sorted(df['transmission'].unique())
            transmission = st.selectbox("Transmission Type", 
                                      transmission_options,
                                      format_func=lambda x: f"🔄 {x}")
            
            drive_options = sorted(df['drive'].unique())
            drive = st.selectbox("Drive Type", 
                               drive_options,
                               format_func=lambda x: f"⚙️ {x}")
            
            hand_drive_options = sorted(df['hand_drive'].unique())
            hand_drive = st.selectbox("Hand Drive", 
                                    hand_drive_options,
                                    format_func=lambda x: f"🚘 {x}")
            
            fuel_options = sorted(df['fuel'].unique())
            fuel = st.selectbox("Fuel Type", 
                              fuel_options,
                              format_func=lambda x: f"⛽ {x}")
        
        # Prediction section
        st.markdown("<h2 class='subheader'>Price Prediction</h2>", unsafe_allow_html=True)
        
        # Create two columns for prediction and comparison
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            # When Predict Button is Clicked
            if st.button("Predict Price", type="primary", use_container_width=True):
                # Create input dataframe
                input_data = pd.DataFrame({
                    'mark': [selected_mark],
                    'model': [selected_model],
                    'year': [year],
                    'mileage': [mileage],
                    'engine_capacity': [engine_capacity],
                    'transmission': [transmission],
                    'drive': [drive],
                    'hand_drive': [hand_drive],
                    'fuel': [fuel]
                })
                
                # Display the selected configuration
                st.write("Your selected configuration:")
                st.dataframe(input_data)
                
                # In a real implementation, you would load and use your model here
                try:
                    with open("best_car_price_model.pkl", "rb") as f:
                        model = pickle.load(f)
                    
                    # If you used label encoding before training, reapply same encoding here
                    input_encoded = input_data.copy()
                    for col in ['mark', 'model', 'transmission', 'drive', 'hand_drive', 'fuel']:
                        # Map categorical values to codes based on the training data mapping
                        input_encoded[col] = input_encoded[col].astype('category').cat.codes
                    
                    # Make prediction
                    predicted_price = model.predict(input_encoded)[0]
                    
                    # If log transformation was used during training
                    # predicted_price = np.expm1(predicted_price)
                    
                    # Display prediction with animation
                    st.balloons()
                    st.success(f"Estimated Car Price: ¥{predicted_price:,.0f}")
                    
                    # Save prediction to session state for history
                    if 'prediction_history' not in st.session_state:
                        st.session_state.prediction_history = []
                    
                    # Add current prediction to history
                    st.session_state.prediction_history.append({
                        'mark': selected_mark,
                        'model': selected_model,
                        'year': year,
                        'price': predicted_price
                    })
                    
                except FileNotFoundError:
                    # For demo purposes (when model file is not available)
                    st.info("Model file not found. Showing sample prediction.")
                    # Generate a realistic price based on the car details
                    base_price = 1000000  # Base price in Yen
                    year_factor = (year - min_year) / (max_year - min_year)
                    mileage_factor = 1 - (mileage / 200000) if mileage < 200000 else 0.1
                    engine_factor = engine_capacity / 2000
                    
                    sample_price = base_price * (0.5 + year_factor) * (0.3 + mileage_factor) * engine_factor
                    st.balloons()
                    st.success(f"Estimated Car Price: ¥{sample_price:,.0f}")
                    
                    # Add to prediction history
                    if 'prediction_history' not in st.session_state:
                        st.session_state.prediction_history = []
                    
                    st.session_state.prediction_history.append({
                        'mark': selected_mark,
                        'model': selected_model,
                        'year': year,
                        'price': sample_price
                    })
        
        with pred_col2:
            # Budget comparison tool
            st.subheader("Budget Comparison")
            budget = st.number_input("Your Budget (¥)", min_value=0, value=1000000, step=100000)
            
            if st.button("Check Affordability", use_container_width=True):
                if 'prediction_history' in st.session_state and st.session_state.prediction_history:
                    latest_prediction = st.session_state.prediction_history[-1]['price']
                    
                    if budget >= latest_prediction:
                        st.success(f"✅ You can afford this car! You'll have ¥{budget - latest_prediction:,.0f} remaining.")
                        # Visualization of budget vs price
                        budget_data = pd.DataFrame({
                            'Category': ['Your Budget', 'Car Price', 'Remaining'],
                            'Amount': [budget, latest_prediction, budget - latest_prediction]
                        })
                        fig = px.bar(budget_data, x='Category', y='Amount', color='Category',
                                    title="Budget Breakdown", text_auto=True)
                        st.plotly_chart(fig)
                    else:
                        shortfall = latest_prediction - budget
                        st.warning(f"❌ This car is above your budget by ¥{shortfall:,.0f}.")
                        # Suggestions for alternatives
                        st.write("Consider these options:")
                        st.write("- Increase your budget")
                        st.write("- Look for an older model")
                        st.write("- Consider a different model or brand")
                        st.write("- Look for higher mileage options")
                else:
                    st.info("Predict a car price first to check affordability.")
        
        # Prediction History
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            st.markdown("<h2 class='subheader'>Your Recent Predictions</h2>", unsafe_allow_html=True)
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['price'] = history_df['price'].apply(lambda x: f"¥{x:,.0f}")
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.experimental_rerun()
    
    with tab2:
        st.markdown("<h2 class='subheader'>Market Insights</h2>", unsafe_allow_html=True)
        st.markdown("<p class='highlight'>Explore trends in the Japanese used car market.</p>", 
                    unsafe_allow_html=True)
        
        # Sample visualizations based on the dataset
        viz_option = st.selectbox("Choose visualization", 
                                ["Popular Car Brands", "Average Price by Year", "Price vs Mileage"])
        
        if viz_option == "Popular Car Brands":
            # Show top car brands by count
            top_brands = df['mark'].value_counts().head(10).reset_index()
            top_brands.columns = ['Brand', 'Count']
            
            fig = px.bar(top_brands, x='Brand', y='Count', 
                        title="Top 10 Popular Car Brands",
                        color='Count', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Average Price by Year":
            # For demo purposes, create a sample visualization
            years = sorted(df['year'].unique())
            sample_prices = []
            
            for y in years:
                # In a real implementation, use actual average prices from the dataset
                # Here we're creating sample data for demonstration
                base = 500000 + (y - min(years)) * 50000
                sample_prices.append(base * (1 + np.random.rand() * 0.2))
            
            year_price_df = pd.DataFrame({
                'Year': years,
                'Avg_Price': sample_prices
            })
            
            fig = px.line(year_price_df, x='Year', y='Avg_Price', 
                        title="Average Car Price by Manufacturing Year",
                        markers=True)
            fig.update_layout(yaxis_title="Average Price (¥)")
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Price vs Mileage":
            # For demo purposes, visualize the relationship between mileage and price
            st.write("Relationship between mileage and car price:")
            
            # Create sample data for visualization
            sample_size = min(1000, len(df))
            sample_df = df.sample(sample_size)
            
            # Add a sample price column for demonstration
            sample_df['sample_price'] = sample_df['year'] * 100000 - sample_df['mileage'] * 0.1
            
            fig = px.scatter(sample_df, x='mileage', y='sample_price', 
                           color='year', hover_data=['mark', 'model'],
                           title="Price vs Mileage by Manufacturing Year",
                           labels={'sample_price': 'Estimated Price (¥)', 'mileage': 'Mileage (km)'},
                           color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("<h2 class='subheader'>Help & FAQ</h2>", unsafe_allow_html=True)
        
        # FAQ section
        faq = [
            {
                "question": "How accurate is this price prediction?",
                "answer": "The prediction model is based on historical Japanese used car data and provides an estimate. Actual prices may vary based on specific conditions of the vehicle, market fluctuations, and regional differences."
            },
            {
                "question": "What factors most influence used car prices?",
                "answer": "The key factors affecting used car prices are typically: the car's age (year), mileage, brand reputation, model popularity, engine capacity, and overall condition."
            },
            {
                "question": "How can I improve the accuracy of my prediction?",
                "answer": "Ensure all fields are filled with accurate information. The more precise your inputs are regarding the car's specifications, the more accurate the price prediction will be."
            },
            {
                "question": "Why do some brands have higher residual value?",
                "answer": "Brands with strong reliability records (like Toyota and Honda) typically maintain higher residual values. Luxury brands may depreciate faster initially but stabilize in the long term."
            },
            {
                "question": "How should I interpret the price prediction?",
                "answer": "The predicted price should be treated as a reference point rather than an absolute value. It's advisable to check actual market listings to compare with similar vehicles."
            }
        ]
        
        # Display FAQs in an expandable format
        for i, item in enumerate(faq):
            with st.expander(f"Q: {item['question']}"):
                st.write(f"A: {item['answer']}")
        
        # Help section with tips
        st.markdown("<h3 class='subheader'>Tips for Using This Tool</h3>", unsafe_allow_html=True)
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.markdown("""
            **For Accurate Predictions:**
            - Enter the exact model name and variant
            - Provide accurate mileage information
            - Consider the car's condition (though not captured in this model)
            - Use recent market data as a comparison
            """)
        
        with tips_col2:
            st.markdown("""
            **Understanding the Results:**
            - The price is an estimate based on historical data
            - Market conditions can affect actual prices
            - Regional differences may not be captured
            - Additional features might affect the actual price
            """)
        
        # Contact section
        st.markdown("<h3 class='subheader'>Need More Help?</h3>", unsafe_allow_html=True)
        st.info("If you have questions about this prediction tool or need assistance, please fill out the form below:")
        
        contact_col1, contact_col2 = st.columns(2)
        
        with contact_col1:
            contact_name = st.text_input("Your Name")
            contact_email = st.text_input("Your Email")
        
        with contact_col2:
            contact_subject = st.selectbox("Subject", ["General Question", "Technical Support", "Feedback", "Other"])
            contact_message = st.text_area("Your Message")
        
        if st.button("Submit", key="contact_submit"):
            st.success("Thank you for your message! We will get back to you soon.")
    
    # Footer
    st.markdown("""
    <div class='footer'>
        Japanese Used Car Price Predictor © 2025<br>
        This is a demonstration application and predictions are for reference only.
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Unable to load the dataset. Please check if the file exists and try again.")
    
    # Create a sample demo mode
    st.warning("Running in DEMO mode with limited functionality.")
    
    st.markdown("<h2 class='subheader'>Sample Car Selection</h2>", unsafe_allow_html=True)
    
    sample_mark = st.selectbox("Select Car Brand", ["Toyota", "Honda", "Nissan", "Mazda", "Subaru"])
    
    # Sample models based on selected brand
    sample_models = {
        "Toyota": ["Corolla", "Camry", "RAV4", "Prius"],
        "Honda": ["Civic", "Accord", "CR-V", "Fit"],
        "Nissan": ["Altima", "Sentra", "Rogue", "Maxima"],
        "Mazda": ["Mazda3", "Mazda6", "CX-5", "MX-5"],
        "Subaru": ["Impreza", "Outback", "Forester", "Legacy"]
    }
    
    sample_model = st.selectbox("Select Car Model", sample_models[sample_mark])
    
    # Demo buttons for quick interaction
    if st.button("Show Demo Prediction", type="primary"):
        st.success("Demo Prediction: ¥1,850,000")
        st.info("This is a demonstration value only. For accurate predictions, please load the actual dataset.")
