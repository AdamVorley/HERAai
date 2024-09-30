import csv
import datetime
from datetime import datetime as dt
from flask import Flask, request, redirect, render_template, url_for, flash 
import os
import pickle
import pandas as pd
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()


app = Flask(__name__)


# load the model
#model = pickle.load(open('model.pkl', 'rb')) #TODO add model when received


class SavedFile:
    def __init__(self, file_id: int, name: str, path: str, datetime_uploaded: datetime):
        self.id = file_id
        self.name = name
        self.path = path
        self.datetime_uploaded = datetime_uploaded

    def __repr__(self):
        return f"SavedFile(id={self.id}, name='{self.name}', path='{self.path}', datetime_uploaded={self.datetime_uploaded})"



@app.route("/")
def home():
        return render_template(
        "upload.html"
    )

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

# this route is used to upload a file. The file is uploaded into the project 'uploads' folder. The file will be stored on the server
# and the necessary data related to the file such as filename, id etc. in the database. 
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect('/')
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        flash(f'File {file.filename} uploaded successfully!')
        #create saved file object
        saved_file = SavedFile(file_id= "HAI_" + dt.now(), name="placeholder", path=file_path,datetime_uploaded=dt.now())
        #TODO - add this to the database
        return redirect('/')

# this route is used to take manually entered date and save it into a txt file. The file is uploaded into the project 'uploads' folder. The file will be stored on the server
# and the necessary data related to the file such as filename, id etc. in the database. 
@app.route('/manual', methods=['POST'])
def manual_entry():
    # get the data inputted in the form 
    manual_data = request.form.get('manualData')
    # check data is present
    if not manual_data: 
        flash('No data entered')
        return redirect('/')
    
    # Create txt file with the submitted manual data - filename is generated automatically 
    txt_filename = f'manual_data_{dt.now().strftime("%Y%m%d_%H%M%S")}.txt'
    txt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)

    # Write the manual data to the txt file
    with open(txt_filepath, 'w') as txt_file:
        txt_file.write(manual_data)

    flash(f'File saved as {txt_filename}')

    #create saved file object
   # saved_file = SavedFile(file_id= "HAI_" + dt.now(), name="placeholder", path=txt_filepath,datetime_uploaded=dt.now())
    #TODO - add this to the database

   # result = run_model(manual_data)
    result = manual_data
# passing result from model to result page
    return render_template('result.html', output=result)

    #return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)


def run_model(input):
   # output = model.predict([input])[0] 
    output = "This is a test output" + input
    return output 