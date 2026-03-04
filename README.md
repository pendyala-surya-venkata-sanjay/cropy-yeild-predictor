# 🌾 Crop Yield Prediction System

A Machine Learning–based system that predicts **crop yield per hectare** using environmental and agricultural factors.
The project also provides **soil recommendations, crop suggestions, and weather integration** for better agricultural decision-making.

---

## 📌 Project Overview

Agricultural productivity depends on multiple factors such as rainfall, temperature, soil moisture, humidity, and irrigation.
This project uses **Machine Learning models** to analyze these factors and predict crop yield accurately.

The system provides an interactive **Streamlit web application** that allows farmers or researchers to input conditions and obtain predictions instantly.

---

## 🚀 Features

* 🌱 Crop yield prediction using Machine Learning
* 🌦 Real-time weather integration using OpenWeather API
* 👨‍🌾 Farmer Mode (simple inputs)
* 🔬 Scientist Mode (advanced inputs)
* 🌿 Soil health recommendations
* 🌾 Crop suggestions based on environmental conditions
* 🖥 Interactive Streamlit web interface

---

## 🧠 Machine Learning Models Used

The model training pipeline evaluates multiple algorithms:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost (optional)

The **best performing model** is saved as:

```
best_crop_yield_model.pkl
```

---

## 📂 Project Structure

```
crop-yield-predictor
crop-yield-predictor
│
├── app.py                     # Streamlit web application
│
├── data
│   └── crop_yield_dataset1.csv   # Dataset used for training
│
├── model
│   └── best_crop_yield_model.pkl # Saved trained ML model
│
├── maincode.ipynb             # Jupyter notebook for experimentation
│
├── whole.py                   # ML training pipeline
│
├── images
│   └── logo.png               # Project logo / screenshots
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Ignore unnecessary files
```

---

## 📊 Dataset

The dataset contains agricultural and environmental parameters such as:

* State
* District
* Crop
* Seed Variety
* Rainfall
* Temperature
* Soil pH
* Soil Moisture
* Humidity
* Irrigation Count
* Previous Yield
* Area (Hectares)
* Yield per hectare

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* OpenWeather API
* Machine Learning Pipelines

---

## 💻 Installation

Clone the repository:

```
git clone https://github.com/pendyala-surya-venkata-sanjay/crop-yield-predictor.git
```

Move into the project directory:

```
cd crop-yield-predictor
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the Streamlit application:

```
streamlit run app.py
```

The application will open in your browser.

---

## 📈 Model Training

To train the model again using the dataset:

```
python whole.py
```

or run the **Jupyter Notebook**:

```
maincode.ipynb
```

This will:

* preprocess the dataset
* train multiple machine learning models
* evaluate performance
* save the best model as `.pkl`

---

## 🖥 Application Interface

The system provides two modes:

### 👨‍🌾 Farmer Mode

Simplified inputs designed for farmers with categorized options.

### 🔬 Scientist Mode

Detailed numerical inputs for researchers and analysts.

The system predicts:

* Yield per hectare
* Total yield
* Soil recommendations
* Suitable crop suggestions

---

## 🔮 Future Improvements

* Fertilizer recommendation system
* Pest and disease prediction
* Satellite weather data integration
* Crop price prediction
* Deep learning–based yield prediction

---

## 👨‍💻 Author

**Pendyala Surya Venkata Sanjay**
B.Tech Computer Science Engineering (2027)

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub.
