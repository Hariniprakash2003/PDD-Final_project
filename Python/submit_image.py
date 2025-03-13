# from flask import Blueprint, render_template, request, redirect
# import torch
# from torchvision.transforms import transforms
# from PIL import Image
# import trainer
# import psycopg2

# submit_image = Blueprint("submit_image", __name__, static_folder="static", template_folder="templates")

# @submit_image.route("/image", methods=['GET', 'POST'])
# @submit_image.route("/", methods=['GET', 'POST'])
# def check():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Database connection configuration
#     host = "localhost"
#     database = "patient_details"
#     user = "postgres"
#     psword = "postgres"  # Replace with your actual password

#     if request.method == 'POST':
#         img = request.files['image']
#         email = request.form['email']
#         #model_path = './classification_model.pth'

#         # trainer.classification_model.load_state_dict(torch.load('./classification_model.pth', map_location=torch.device('cpu')), strict=False)

#         trainer.classification_model.load_state_dict(
#       torch.load('./classification_model.pth', map_location=torch.device(device)), strict=False
# )
# trainer.classification_model.to(device)  # Ensure model is on the correct device
# trainer.classification_model.eval()


#         # trainer.classification_model.load_state_dict(torch.load('./classification_model.pth', map_location=torch.device('cpu')))

#         test_image_path = Image.open(img).convert('RGB')
#         test_image = trainer.test_transformations(test_image_path).to(device).unsqueeze(0)

#         # predicting
#         prediction = trainer.classification_model.to(device).eval()(test_image)
#         result = torch.argmax(prediction,dim=1)
#         #image = Image.open(img).convert("L")
#         # Connect to the PostgreSQL database
#         if result.item() == 1:
#             msg = "Positive"

#         elif result.item() == 0:
#             msg = "Negative"

#         connection = psycopg2.connect(host=host, database=database, user=user, password=psword)
#         cursor = connection.cursor()

#         # Insert form data into the table, with patient_id as a serial type (auto-generated)
#         insert_query = "UPDATE patient_details SET result = %s WHERE email = %s "
#         cursor.execute(insert_query, (msg,email))
#         connection.commit()

#         select_query = "SELECT * from patient_details WHERE email = %s "
#         cursor.execute(select_query, (email,))

#         data = cursor.fetchall()
#         res = prediction[0][result.item()].item()*100
#         res = round((res - (-300.0)) * (100.0 - 80.0) / (1000.0 - (-300.0)) + 80.0,2)
#         # Close the database connection
#         cursor.close()
#         connection.close()
#         return render_template("Pages/result.html", data=data , res=res )

#     return redirect("/upload")



from flask import Blueprint, render_template, request, redirect
import torch
from torchvision.transforms import transforms
from PIL import Image
import trainer
import psycopg2
import io

submit_image = Blueprint("submit_image", __name__, static_folder="static", template_folder="templates")

@submit_image.route("/image", methods=['GET', 'POST'])
@submit_image.route("/", methods=['GET', 'POST'])
def check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Database connection configuration
    host = "localhost"
    database = "patient_details"
    user = "postgres"
    psword = "postgres"  # Replace with your actual password

    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image uploaded!", 400
        
        img = request.files['image']
        if img.filename == '':
            return "No selected file!", 400
        
        email = request.form.get('email')
        if not email:
            return "Email is required!", 400

        # Load model only if not already loaded
        if not hasattr(trainer, 'classification_model_loaded'):
            trainer.classification_model.load_state_dict(
                torch.load('./classification_model.pth', map_location=device), strict=False
            )
            trainer.classification_model.to(device)
            trainer.classification_model.eval()
            trainer.classification_model_loaded = True  # Avoid reloading model multiple times

        # Process image
        image = Image.open(io.BytesIO(img.read())).convert('RGB')
        test_image = trainer.test_transformations(image).to(device).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            prediction = trainer.classification_model(test_image)
            result = torch.argmax(prediction, dim=1).item()

        msg = "Positive" if result == 1 else "Negative"

        # Connect to PostgreSQL
        try:
            connection = psycopg2.connect(host=host, database=database, user=user, password=psword)
            cursor = connection.cursor()

            # Update result in DB
            cursor.execute("UPDATE patient_details SET result = %s WHERE email = %s", (msg, email))
            connection.commit()

            # Fetch updated data
            cursor.execute("SELECT * FROM patient_details WHERE email = %s", (email,))
            data = cursor.fetchall()

        except psycopg2.Error as e:
            return f"Database error: {e}", 500

        finally:
            cursor.close()
            connection.close()

        # Convert prediction to percentage
        res = round((prediction[0][result].item() - (-300.0)) * (100.0 - 80.0) / (1000.0 - (-300.0)) + 80.0, 2)

        return render_template("Pages/result.html", data=data, res=res)

    return redirect("/upload")
