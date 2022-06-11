from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import os
from infomation_extractor import Predictor
app = Flask(__name__, template_folder='./')


UPLOAD_FOLDER = './static/src/uploads'
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
        
        book_info=predictor.predict("./static/src/uploads/" + secure_filename(f.filename))
        print('####################')
        print(book_info)
        print('####################')
        extracted_infos = {
			"status": "OK", #any status <> "OK" means failed to extract
            "file": secure_filename(f.filename),
            "title": book_info[0],
            "author": book_info[1],
            "publisher": book_info[2],
            "volume": book_info[3],
            "translator": book_info[4],
            "date": book_info[5]
        }
        return jsonify(extracted_infos)
    else:
        return ''

if __name__ == '__main__':
    predictor = Predictor()
    predictor.load_detect_model()
    predictor.load_reg_model()
    predictor.load_text_detect_model()
    app.run(debug=True)
