from flask import Flask, render_template, request
import pickle
from preprocess import preprocess_input

app = Flask(__name__)

# Load models
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def predict_news(text):
    processed = preprocess_input(text)

    vector = tfidf.transform([processed])

    pred = model.predict(vector)[0]

    return "Fake News ❌" if pred == 1 else "True News ✅"


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        news = request.form['news']
        prediction = predict_news(news)

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)