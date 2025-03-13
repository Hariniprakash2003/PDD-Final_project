import psycopg2

try:
    connection = psycopg2.connect(
        host="localhost",
        database="patient_details",
        user="postgres",
        password="postgres"  # Replace with your password
    )
    print("Connection successful")
    connection.close()
except Exception as e:
    print("Error:", e)
