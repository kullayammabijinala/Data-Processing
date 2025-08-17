from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'dataset' not in request.files:
        return redirect(url_for('index'))
    file = request.files['dataset']
    if file.filename == '':
        return redirect(url_for('index'))

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Read dataset
    df = pd.read_csv(filepath)

    # Handle missing values for numeric columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy='mean')
        df[num_cols] = imputer.fit_transform(df[num_cols])

    # Handle categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        processed_cat = pd.DataFrame(index=df.index)
        for col in cat_cols:
            unique_vals = df[col].nunique()
            if unique_vals > 100:  # Too many categories → Label Encode
                le = LabelEncoder()
                processed_cat[col] = le.fit_transform(df[col].astype(str))
            else:
                # OneHotEncode small-cardinality columns (sparse to save memory)
                ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                temp_df = pd.DataFrame(ohe.fit_transform(df[[col]]),
                                       columns=ohe.get_feature_names_out([col]),
                                       index=df.index)
                processed_cat = pd.concat([processed_cat, temp_df], axis=1)

        # Drop original categorical columns
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, processed_cat], axis=1)

    # Normalize numerical features
    if len(df.columns) > 0:
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])

    # Split dataset
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    processed_path_train = os.path.join(PROCESSED_FOLDER, 'train_processed.csv')
    processed_path_test = os.path.join(PROCESSED_FOLDER, 'test_processed.csv')

    train.to_csv(processed_path_train, index=False)
    test.to_csv(processed_path_test, index=False)

    return render_template('result.html',
                           train_file='train_processed.csv',
                           test_file='test_processed.csv')

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
