from flask import Flask, render_template, request
import model

app = Flask(__name__)


@app.route('/')
def form_input():
    return render_template("textbox_input.html", data={"text": ""})


@app.route('/', methods=['POST'])
def form_input_post():
    text = request.form['text']
    response = ""
    if text != "":
        response = model.predict_text(text)
        print(response)
    return render_template("textbox_input.html", data={"text": response})


if __name__ == '__main__':
    app.run(debug=True)
