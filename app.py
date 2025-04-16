# app.py
from flask import Flask, render_template, request
from summarizer import summarize_hindi

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':
        text = request.form['input_text']
        summary = summarize_hindi(text)
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)




