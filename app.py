from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('iris_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)
        species = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}
        output = species[prediction[0]]
        return render_template('index.html', prediction_text=f'The Iris species is: {output}')
    except:
        return render_template('index.html', prediction_text="Error in prediction. Please check your input.")

if __name__ == '__main__':
    app.run(debug=True)
