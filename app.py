from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image
from models_and_predict import predict_all  

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route("/", methods=["GET", "POST"])
def index():
    image_name = None
    error = None
    label = None
    if request.method == "POST":
        if "image" not in request.files:
            error = "No image uploaded."
        else:
            file = request.files["image"]
            if file.filename == "":
                error = "Please select an image!"
            elif not allowed_file(file.filename):
                error = "File type not supported."
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                try:
                    # Doar conversie la RGB pentru a evita erorile de format
                    cls, verdict, prob = predict_all(filepath)
                    image_name = filename  # Numele original al fisierului
                    
                    if cls == "persoana":
                        label = (
                            f"Person - Real ({prob*100:.2f}%)"
                            if verdict == "real"
                            else f"Person - AI Generated ({(1-prob)*100:.2f}%)"
                        )
                    elif cls == "obiect":
                        label = (
                            f"Object - Real ({prob*100:.2f}%)"
                            if verdict == "real"
                            else f"Object - AI Generated ({(1-prob)*100:.2f}%)"
                        )
                    else:
                        label = f"Unknown class: {cls}"
                except Exception as e:
                    error = f"Invalid image file or could not process image: {str(e)}"
    return render_template("index.html", image=image_name, label=label, error=error)
if __name__ == "__main__":
    app.run(debug=True)