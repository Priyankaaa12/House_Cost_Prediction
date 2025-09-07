from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained pipeline
pipeline = joblib.load('house_price_pipeline_v2.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'GrLivArea': float(request.form['sqft']),
            'FullBath': int(request.form['full_bath']),
            'HalfBath': int(request.form['half_bath']),
            'OverallQual': int(request.form['quality']),
            'TotalBsmtSF': float(request.form['basement']),
            'GarageCars': int(request.form['garage_cars']),
            'Neighborhood': request.form['neighborhood'],
            'YearBuilt': int(request.form['year_built']),
            'Fireplaces': int(request.form['fireplace'])
        }

        # Prepare features
        features = {
            'GrLivArea': data['GrLivArea'],
            'TotalBath': data['FullBath'] + 0.5 * data['HalfBath'],
            'OverallQual': data['OverallQual'],
            'TotalSF': data['GrLivArea'] + data['TotalBsmtSF'],
            'Age': 2023 - data['YearBuilt'],
            'GarageCars': data['GarageCars'],
            'Neighborhood_Tier': get_neighborhood_tier(data['Neighborhood']),
            'HasFireplace': 1 if data['Fireplaces'] > 0 else 0
        }

        # Make prediction
        X_new = pd.DataFrame([features.values()], columns=pipeline['features'])
        X_scaled = pipeline['scaler'].transform(X_new)
        price = np.expm1(pipeline['model'].predict(X_scaled))[0]

        return render_template('result.html', 
                             price=f"${price:,.2f}",
                             inputs=data)

    except Exception as e:
        return render_template('error.html', error=str(e))

def get_neighborhood_tier(neighborhood):
    """Map neighborhood to tier"""
    premium = ['NoRidge', 'NridgHt', 'StoneBr', 'Timber']
    mid = ['CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst']
    return 1 if neighborhood in premium else 2 if neighborhood in mid else 3

if __name__ == '__main__':
    app.run(debug=True)