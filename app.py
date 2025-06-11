from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import io
import base64

app = Flask(__name__)

data = pd.read_csv(r'C:/Users/ajayb/OneDrive/Desktop/fat_prediction/fat.csv')
X = data.drop(columns=['BodyFat'])
y = data['BodyFat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Support Vector Machine': SVR(kernel='linear'),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[model_name] = {
        'R2 Score': round(r2_score(y_test, y_pred) * 100, 2),  # Convert to percentage
        'MSE': round(mean_squared_error(y_test, y_pred), 2)
    }

# Best model selection: High R2 Score but penalizing high MSE
best_model = max(accuracies, key=lambda k: (accuracies[k]['R2 Score'] / 100) - (accuracies[k]['MSE'] / max(acc['MSE'] for acc in accuracies.values())))

def generate_visualization(input_data, predictions, best_model):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=input_data.columns, y=input_data.iloc[0], palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Input Features')
    plt.xlabel('Features')
    plt.ylabel('Values')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot1 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(8, 4))
    colors = ['red' if model == best_model else 'blue' for model in predictions.keys()]
    sns.barplot(x=list(predictions.keys()), y=list(predictions.values()), palette=colors)
    plt.title(f'Model Predictions (Best: {best_model})')
    plt.xlabel('Models')
    plt.ylabel('Predicted Body Fat (%)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot2 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return plot1, plot2

@app.route('/', methods=['GET', 'POST'])
def predict():
    predictions = {}
    plots = []
    if request.method == 'POST':
        try:
            input_data = pd.DataFrame({
                'Density': [float(request.form['density'])],
                'Age': [int(request.form['age'])],
                'Weight': [float(request.form['weight'])],
                'Height': [float(request.form['height'])],
                'Neck': [float(request.form['neck'])],
                'Chest': [float(request.form['chest'])],
                'Abdomen': [float(request.form['abdomen'])],
                'Hip': [float(request.form['hip'])],
                'Thigh': [float(request.form['thigh'])],
                'Knee': [float(request.form['knee'])],
                'Ankle': [float(request.form['ankle'])],
                'Biceps': [float(request.form['biceps'])],
                'Forearm': [float(request.form['forearm'])],
                'Wrist': [float(request.form['wrist'])]
            })
            for model_name, model in models.items():
                predictions[model_name] = model.predict(input_data)[0].round(2)

            plots = generate_visualization(input_data, predictions, best_model)
        except Exception as e:
            predictions['Error'] = str(e)

    return render_template('index.html', predictions=predictions, accuracies=accuracies, best_model=best_model, plot1=plots[0] if plots else None, plot2=plots[1] if plots else None)

if __name__ == '__main__':
    app.run(debug=True)
