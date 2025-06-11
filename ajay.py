from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io
import base64

app = Flask(__name__)

data = pd.read_csv(r'C:/Users/ajayb/OneDrive/Desktop/fat_prediction/fat.csv')
X = data.drop(columns=['BodyFat'])
y = data['BodyFat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

def generate_visualization(input_data, predicted_bodyfat):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=input_data.columns, y=input_data.iloc[0], palette='viridis')
    plt.xticks(rotation=45)
    plt.title('User Input Features')
    plt.xlabel('Features')
    plt.ylabel('Values')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=input_data['Abdomen'], y=[predicted_bodyfat], color='red', s=100)
    plt.title('Abdomen vs Predicted BodyFat')
    plt.xlabel('Abdomen')
    plt.ylabel('Predicted BodyFat')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url2 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return plot_url, plot_url2

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
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
            predicted_bodyfat = model.predict(input_data)[0]
            prediction = f'{predicted_bodyfat:.2f}%'
            plots = generate_visualization(input_data, predicted_bodyfat)
        except Exception as e:
            prediction = f'Error: {str(e)}'

    return render_template('index.html', prediction=prediction, plot1=plots[0] if plots else None, plot2=plots[1] if plots else None)

if __name__ == '__main__':
    app.run(debug=True)
