import csv
import datetime
from datetime import datetime as dt
from flask import Flask, request, redirect, render_template, url_for, flash 
import os
import pickle
import pandas as pd
from sklearn import preprocessing 
import nltk  # Natural Language Toolkit (NLTK) for text processing
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting the data and performing grid search for hyperparameter tuning
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text data into TF-IDF features
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier
from sklearn.pipeline import Pipeline  # For creating machine learning pipelines
from sklearn.metrics import classification_report, accuracy_score  # For evaluating model performance
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
le = preprocessing.LabelEncoder()


app = Flask(__name__)


# load the model
#model = pickle.load(open('model.pkl', 'rb')) #TODO add model when received
single_response_model = pickle.load(open('single_response_model.pkl', 'rb'))

# Initialize NLTK resources
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


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
        selected_option = request.form.get('options')
        #TODO determine which model has been selected ^ 

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

    input = [sentence.strip() for sentence in manual_data.split('.') if sentence.strip()] # split the input into an array

    result = run_model(input, request.form.get('manualOptions'))

    # passing result from model to result page
    return render_template('result.html', output=result)


if __name__ == '__main__':
    app.run(debug=True)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    return ' '.join(tokens)


def run_model(input, selected_model):
   if selected_model == "single":
    return run_single_model(input) 
    


def run_matrix_model(input):
    return "null"

def run_linear_model(input):
    return "null"

def run_target_model(input):
    return "null"

def run_single_model(input):
    label_mapping = {
    'Response A': 0,
    'Response B': 1,
    'Response C': 2,
    'Response D': 3,
    'Response E': 4,  # Adjust these mappings based on your dataset's labels
    'Response F': 5   # Add or remove as necessary
}  

    processed_descriptions = [preprocess_text(desc) for desc in input]  # Preprocess the job descriptions
    predictions = single_response_model.predict(processed_descriptions)  # Predict using the best model
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}  # Reverse the label mapping for readable output
    return [reverse_label_mapping[prediction] for prediction in predictions]  # Return human-readable predictions