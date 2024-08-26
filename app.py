from flask import Flask, render_template, request, jsonify
from prediction import load_model,predict_price

app = Flask(__name__)

# Load models for all car brands
models = {}
for car_model in ['ford_Ecosport', 'Honda_City', 'Hyundai_Verna', 'Tata_Nexon', 'Suzuki_Swift_VXi', 'Volkswagen_Polo']:
    try:
        models[car_model] = load_model(car_model)
    except FileNotFoundError as e:
        print(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    car_model = data['car_model']
    year = int(data['year'])
    kilometers = int(data['kilometers'])

    if car_model in models:
        # Predict the price using the appropriate model
        predicted_price = predict_price(models[car_model], year, kilometers)
        return jsonify({'price': predicted_price})
    else:
        return jsonify({'error': 'Model not found for the selected car brand'}), 400

if __name__ == '__main__':
    app.run(debug=True)
