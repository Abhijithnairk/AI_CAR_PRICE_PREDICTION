import json
import mysql.connector
from mysql.connector import Error

try:
    # Connect to the MySQL database
    conn = mysql.connector.connect(
        host="localhost",
        user="abhijith_user",  
        password="Abhi@1212",  
        database="car_datas"  # Ensure the database name matches your actual database
    )

    if conn.is_connected():
        cursor = conn.cursor()

        # Load the data from the JSON file
        with open('data.json', 'r') as file:
            data = json.load(file)

        # Insert data into respective brand tables
        for brand, details in data.items():
            table_name = f"{brand.lower()}"  # Dynamically determine table name based on brand
            min_length = min(len(details['Year']), len(details['Kilometers']), len(details['Price']))
            for i in range(min_length):
                query = f"""
                INSERT INTO {table_name} (year, kilometers, price)
                VALUES (%s, %s, %s)
                """
                values = (details['Year'][i], details['Kilometers'][i], details['Price'][i])
                cursor.execute(query, values)

        # Commit the transaction
        conn.commit()

        print("Data inserted successfully!")

except Error as e:
    print(f"Error occurred: {e}")

finally:
    # Close the database connection
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("Database connection closed.")
