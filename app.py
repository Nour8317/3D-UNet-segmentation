from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Define a folder to save uploaded files (ensure this folder exists)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')

        # Validate the number of files
        if len(files) != 4:
            return render_template('index.html', result="Please upload exactly 4 .nii files.")

        saved_files = []
        for file in files:
            # Validate file type
            if not file.filename.endswith('.nii'):
                return render_template('index.html', result="All files must be .nii format.")

            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            saved_files.append(file_path)

        # Perform your processing or predictions with the saved files
        # For example:
        # result = your_model.predict(saved_files)

        return render_template('index.html', result="Prediction complete with 4 files.")

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
