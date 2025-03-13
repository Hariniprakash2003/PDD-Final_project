# from flask import Flask , render_template 
# from Python.check_db import check_db
# from Python.submit_details import submit_details
# from Python.sign import sign
# from Python.submit_image import submit_image

# app = Flask(_name_, static_folder='static')
# app.secret_key = 'pdd_101420$$$'
# app.register_blueprint(check_db,url_prefix="/details")
# app.register_blueprint(sign,url_prefix="/sign_details")
# app.register_blueprint(submit_details,url_prefix="/upload")
# app.register_blueprint(submit_image,url_prefix="/image")

# @app.route("/",methods=['GET','POST'])
# def index():
#     return render_template("index.html")

# @app.route("/signup")
# def signup():
#     return render_template("Pages/index-signup.html")

# if _name_ == '_main_':
#     app.run(debug=True)







    
# from flask import Flask , render_template 
# from Python.check_db import check_db
# from Python.submit_details import submit_details
# from Python.sign import sign
# from Python.submit_image import submit_image

# app = Flask(__name__, static_folder='static')
# app.secret_key = 'pdd_101420$$$'
# app.register_blueprint(check_db,url_prefix="/details")
# app.register_blueprint(sign,url_prefix="/sign_details")
# app.register_blueprint(submit_details,url_prefix="/upload")
# app.register_blueprint(submit_image,url_prefix="/image")

# @app.route("/",methods=['GET','POST'])
# def index():
#     return render_template("index.html")

# @app.route("/signup")
# def signup():
#     return render_template("Pages/index-signup.html")

# if __name__ == '__main__':
#     app.run(debug=True)

#     from flask import Flask

# app = Flask(__name__)


# @app.route("/")
# def home():
#     return "Flask is running!"

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask , render_template 
from Python.check_db import check_db
from Python.submit_details import submit_details
from Python.sign import sign
from Python.submit_image import submit_image

app = Flask(__name__, static_folder='static')
app.secret_key = 'pdd_101420$$$'
app.register_blueprint(check_db,url_prefix="/details")
app.register_blueprint(sign,url_prefix="/sign_details")
app.register_blueprint(submit_details,url_prefix="/upload")
app.register_blueprint(submit_image,url_prefix="/image")

@app.route("/",methods=['GET','POST'])
def index():
    return render_template("index.html")

@app.route("/signup")
def signup():
    return render_template("Pages/index-signup.html")

if __name__ == '__main__':
    app.run(debug=True)

    from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return "Flask is running!"

if __name__ == "__main__":
    app.run(debug=True)
