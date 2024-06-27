from flask import Flask, request, redirect, render_template, url_for, flash 


app = Flask(__name__)

@app.route("/")
def home():
        return render_template(
        "upload.html"
    )