from flask import Blueprint , render_template,request
import psycopg2

submit_details = Blueprint("submit_details" , __name__ , static_folder="static" , template_folder="templates")

@submit_details.route("/upload" ,methods=['GET','POST'])
@submit_details.route("/",methods=['GET','POST'])
def upload():
    # Database connection configuration
    host = "localhost"
    database = "patient_details"
    user = "postgres"
    psword = "postgres"  # Replace with your actual password


    if request.method == 'POST':
        # Retrieve form data
        name = request.form['name']
        age = int(request.form['age'])
        gender = request.form['gender']
        phone = request.form['phone']
        email = request.form['email']

        # Connect to the PostgreSQL database
        connection = psycopg2.connect(host=host, database=database, user=user, password=psword)
        cursor = connection.cursor()

        # Insert form data into the table, with patient_id as a serial type (auto-generated)
        insert_query = "INSERT INTO patient_details (name, age, gender, phone, email) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(insert_query, (name, age, gender, phone, email))
        connection.commit()

        # Close the database connection
        cursor.close()
        connection.close()
    return render_template("Pages/upload.html" , email = email )





