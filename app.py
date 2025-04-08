from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# 学習モデルの読み込み
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 入力データの取得
    input_data = request.form.to_dict()
    df = pd.DataFrame([input_data])

    # 予測処理
    prediction = model.predict(df)

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
