from flask import Flask, request, render_template
import joblib

#1st Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the form
    message = request.form['message']
    
    # Transform the input message
    message_vector = vectorizer.transform([message])
    
    # Predict using the model
    prediction = model.predict(message_vector)
    
    # Interpret the result
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
