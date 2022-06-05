from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import os
from infomation_extractor import Predictor
app = Flask(__name__, template_folder='./')


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def show_template():
    return render_template("./static/main.html")

@app.route("/extract", methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        # Get image from POST request
        f = request.files['file']
        # Save image to ./uploads
        f.save(os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

        # Extract info API HERE
        predictor = Predictor()
        predictor.load_detect_model()
        predictor.load_reg_model()
        book_info=predictor.predict("uploads/10.jpg")
        print(book_info)
        extracted_infos = {
			"status": book_info[0], #any status <> "OK" means failed to extract
            "title": book_info[1],
            "sub_title": book_info[2],
            "author": book_info[3],
            "date": book_info[4],
            "others": book_info[5]
        }
        return jsonify(extracted_infos)
    else:
        return ''

if __name__ == '__main__':
    app.run(debug=True)
