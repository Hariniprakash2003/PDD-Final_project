# from flask import Blueprint , render_template,request
# import psycopg2
# sign = Blueprint("sign" , __name__ , static_folder="static" , template_folder="templates")

# @sign.route("/sign_details" ,methods=['GET','POST'])
# @sign.route("/",methods=['GET','POST'])
# def signup():
#     # Database connection configuration
#     host = "localhost"
#     database = "patient_details"
#     user = "postgres"
#     pssword = "root"  # Replace with your actual password

#     if request.method == 'POST':
#         # Retrieve form data
#         email = request.form['email']
#         psword = request.form['password']

#         # Connect to the PostgreSQL database
#         connection = psycopg2.connect(host=host, database=database, user=user, password=pssword)
#         cursor = connection.cursor()

#         # Insert form data into the table
#         insert_query = "INSERT INTO users (email, password) VALUES (%s,crypt(%s, gen_salt('bf')));"
#         cursor.execute(insert_query,(email,psword))
#         connection.commit()

#         # Close the database connection
#         cursor.close()
#         connection.close()
#     return render_template("Pages/details.html")




from flask import Blueprint, render_template, request
import psycopg2

sign = Blueprint("sign", __name__, static_folder="static", template_folder="templates")

@sign.route("/sign_details", methods=['GET', 'POST'])
def signup():
    # Database connection configuration
    host = "localhost"
    database = "user_details"  # Verify your database name
    user = "postgres"
    password = "postgres"  # Replace with your actual password

    if request.method == 'POST':
        try:
            # Retrieve form data
            email = request.form['email']
            psword = request.form['password']

            # Connect to the PostgreSQL database
            connection = psycopg2.connect(host=host, database=database, user=user, password=password)
            cursor = connection.cursor()

            # Insert form data into the table
            insert_query = "INSERT INTO users (email, password) VALUES (%s, crypt(%s, gen_salt('bf')));"
            print("Running query:", insert_query)  # Debugging log
            cursor.execute(insert_query, (email, psword))
            connection.commit()

            # Close the database connection
            cursor.close()
            connection.close()
            print("Signup successful")  # Debugging log
            return render_template("Pages/details.html")

        except Exception as e:
            print("Error occurred:", e)  # Log the error
            return "An error occurred during signup", 500

    return render_template("Pages/index-signup.html")
