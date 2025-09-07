"""
Ultimate House Price Predictor v2.0
- Enhanced neighborhood handling
- Additional predictive features
- Optimized XGBoost parameters
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and Prepare Data
try:
    df = pd.read_csv('train.csv')
    print("[SUCCESS] Data loaded successfully. Shape:", df.shape)
except FileNotFoundError:
    print("[ERROR] 'train.csv' not found in current directory")
    exit()

# Basic cleaning
df = df[df['GrLivArea'] < 4000].copy()
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)

# 2. Enhanced Feature Engineering
# Neighborhood tiering
neighborhood_tiers = {
    'Blmngtn': 'Tier2', 'Blueste': 'Tier3', 'BrDale': 'Tier3',
    'BrkSide': 'Tier3', 'ClearCr': 'Tier2', 'CollgCr': 'Tier2',
    'Crawfor': 'Tier2', 'Edwards': 'Tier3', 'Gilbert': 'Tier2',
    'IDOTRR': 'Tier3', 'MeadowV': 'Tier3', 'Mitchel': 'Tier2',
    'NAmes': 'Tier2', 'NoRidge': 'Tier1', 'NPkVill': 'Tier3',
    'NridgHt': 'Tier1', 'NWAmes': 'Tier2', 'OldTown': 'Tier3',
    'SWISU': 'Tier3', 'Sawyer': 'Tier2', 'SawyerW': 'Tier2',
    'Somerst': 'Tier1', 'StoneBr': 'Tier1', 'Timber': 'Tier1',
    'Veenker': 'Tier1'
}
df['Neighborhood_Tier'] = df['Neighborhood'].map(neighborhood_tiers)
tier_avg = df.groupby('Neighborhood_Tier')['SalePrice'].mean().to_dict()
df['Neighborhood_Value'] = df['Neighborhood_Tier'].map(tier_avg)

# Additional features
df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath']
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
df['Age'] = 2023 - df['YearBuilt']
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

# Garage quality scoring (Ex=5, Gd=4, TA=3, Fa=2, Po=1)
garage_qual_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, None:0}
df['GarageScore'] = df['GarageQual'].map(garage_qual_map) * df['GarageCars']

# Final feature selection
features = [
    'GrLivArea', 'TotalBath', 'OverallQual', 'TotalSF',
    'Age', 'GarageScore', 'Neighborhood_Value', 'HasFireplace'
]
X = df[features]
y = df['SalePrice']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Optimized Model
model = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, np.log1p(y_train))  # Log transform

# 6. Evaluate
y_pred = np.expm1(model.predict(X_test_scaled))  # Convert back
print("\n[PERFORMANCE] Model Results:")
print("- MSE: ${:,.2f}".format(mean_squared_error(y_test, y_pred)))
print("- RMSE: ${:,.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
print("- R²: {:.4f}".format(r2_score(y_test, y_pred)))

# 7. Save Pipeline
pipeline = {
    'features': features,
    'scaler': scaler,
    'model': model,
    'neighborhood_tiers': neighborhood_tiers,
    'tier_avg': tier_avg,
    'garage_qual_map': garage_qual_map
}
joblib.dump(pipeline, 'house_price_pipeline_v2.pkl')
print("\n[SAVED] Pipeline to 'house_price_pipeline_v2.pkl'")

# 8. Enhanced Prediction Function
def predict_house_price_v2(sqft, full_bath, half_bath, quality, year_built, 
                         garage_cars, garage_qual, basement_sqft, 
                         has_fireplace, neighborhood):
    """Make a prediction with all enhanced features"""
    try:
        pipeline = joblib.load('house_price_pipeline_v2.pkl')
        
        # Prepare all features
        features = {
            'GrLivArea': sqft,
            'TotalBath': full_bath + 0.5*half_bath,
            'OverallQual': quality,
            'TotalSF': sqft + basement_sqft,
            'Age': 2023 - year_built,
            'GarageScore': pipeline['garage_qual_map'].get(garage_qual, 0) * garage_cars,
            'Neighborhood_Value': pipeline['tier_avg'].get(
                pipeline['neighborhood_tiers'].get(neighborhood, 'Tier3'),
                np.mean(list(pipeline['tier_avg'].values()))
            ),
            'HasFireplace': int(has_fireplace)
        }
        
        # Scale and predict
        X_new = pd.DataFrame([list(features.values())], columns=list(features.keys()))
        X_scaled = pipeline['scaler'].transform(X_new)
        return np.expm1(pipeline['model'].predict(X_scaled))[0]
    
    except Exception as e:
        print("[ERROR] Prediction failed:", str(e))
        return None

# 9. Test Prediction
example_price = predict_house_price_v2(
    sqft=2000,
    full_bath=2,
    half_bath=1,
    quality=7,
    year_built=2005,
    garage_cars=2,
    garage_qual='TA',
    basement_sqft=800,
    has_fireplace=True,
    neighborhood='NAmes'
)

print("\n[PREDICTION] Enhanced Model Example:")
print("- 2000 sqft, 2.5 baths, Quality 7")
print("- Built 2005, 2-car TA garage, 800 sqft basement")
print("- With fireplace, NAmes neighborhood")
print("Predicted Price: ${:,.2f}".format(example_price))

# 10. Feature Importance
plt.figure(figsize=(10,5))
plt.barh(features, model.feature_importances_)
plt.title("Feature Importance (Enhanced Model)")
plt.tight_layout()
plt.savefig('feature_importance_v2.png', dpi=300)
print("\n[SAVED] Feature importance plot to 'feature_importance_v2.png'")