from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import os

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

        extracted_infos = {
			"status": "OK", #any status <> "OK" means failed to extract
            "title": "info",
            "sub_title": "info2",
            "author": "info3",
            "date": "info4",
            "others": "info5"
        }
        return jsonify(extracted_infos)
    else:
        return ''

if __name__ == '__main__':
    app.run(debug=True)
