import mysql.connector
import pandas as pd
from mysql.connector import Error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle
import os

def train_model(car_model):
    try:
        # Connect to the MySQL database
        conn = mysql.connector.connect(
            host="localhost",
            user="abhijith_user", 
            password="Abhi@1212",  
            database="car_datas"
        )

        if conn.is_connected():
            cursor = conn.cursor()

            # Query to retrieve data from the database for the specific car model
            query = f"SELECT year, kilometers, price FROM {car_model.lower()}"
            cursor.execute(query)

            # Fetch all the rows
            rows = cursor.fetchall()

            # Convert the data into a pandas DataFrame
            df = pd.DataFrame(rows, columns=['Year', 'Kilometers', 'Price'])

            # Features (Year, Kilometers_Travelled) and Target (Price)
            X = df[['Year', 'Kilometers']]
            y = df['Price']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Create and train the Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            print(f"{car_model} Model Performance:")
            print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            print(f"R^2 Score: {r2:.2f}")

            # Save the trained model to a file
            model_filename = f'{car_model.lower()}_price_model.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"{car_model} model trained and saved to {model_filename}")

    except Error as e:
        print(f"Error occurred: {e}")

    finally:
        # Close the database connection
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("Database connection closed.")

def load_model(car_model):
    # Load the model for the specific car model from the file
    model_filename = f'{car_model.lower()}_price_model.pkl'
    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        raise FileNotFoundError(f"Model file {model_filename} not found. Please train the model first.")

def predict_price(model, year, kilometers):
    user_input = np.array([[year, kilometers]])
    predicted_price = model.predict(user_input)
    return predicted_price[0]

if __name__ == "__main__":
    # Example: Train models for different car brands
    for car_model in ['ford_Ecosport', 'Honda_City', 'Hyundai_Verna', 'Tata_Nexon', 'Suzuki_Swift_VXi', 'Volkswagen_Polo']:
        train_model(car_model)
