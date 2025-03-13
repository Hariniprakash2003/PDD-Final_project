from pickle import FALSE
from flask import Blueprint , render_template,request
import psycopg2
import pyautogui
retreive = Blueprint("retreive" , __name__ , static_folder="static" , template_folder="templates")

@retreive.route("/data" ,methods=['GET','POST'])
@retreive.route("/",methods=['GET','POST'])
def data():
    
    # Database connection configuration
    host = "localhost"
    database = "patient_details"
    user = "postgres"
    psword = "postgres"  

    if request.method == 'POST':
        # Retrieve form data

        # Connect to the PostgreSQL database
        connection = psycopg2.connect(host=host, database=database, user=user, password=psword)
        cursor = connection.cursor()

        # Check presence of form data in table
        select_query = "SELECT * from patient_details "
        cursor.execute(select_query)

        data = cursor.fetchall()
    
        # Close the database connection
        cursor.close()
        connection.close()

    
        #Code for redirecting pages 
        if data :
            # Code for entering into details page
            return render_template("Pages/details.html" , data = data ) 
        else:
            # Code for indicating incorrect username or password
            pyautogui.alert("Invalid Username or Password")
            return render_template("index.html")