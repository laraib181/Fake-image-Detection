# 🕵️‍♂️ Fake Image Detection using Error Level Analysis (ELA)

This project leverages **Error Level Analysis (ELA)** and **Deep Learning (CNN)** to detect fake (manipulated) images. The application is deployed using **Streamlit** to provide an easy-to-use web interface.

## 📌 Features

- 🔍 Automatically generates ELA images from inputs  
- 🧠 Trains a Convolutional Neural Network for classification  
- 📊 Evaluates model performance with confusion matrix and accuracy metrics  
- 🌐 Web interface powered by Streamlit  


## 📁 Project Structure
├── Fake-image app.py # Streamlit web interface
├── Copy of fake.ipynb # Jupyter notebook (model development & ELA logic)
├── requirements.txt # Python dependencies
├── runtime.txt # Python version pinning for Streamlit
└── README.md # Project documentation

---

## 🚀 How It Works

1. **ELA Conversion**:  
   - Real and manipulated images are saved.  
   - A JPEG version is re-saved at lower quality.  
   - Difference between original and resaved images is calculated to highlight tampering.

2. **Model Architecture**:  
   - Built using `Keras` with layers including:  
     - Convolution, MaxPooling  
     - Dropout & Flatten  
     - Dense Output Layer

3. **Training**:  
   - One-hot encoded labels for binary classification.  
   - `ImageDataGenerator` used for real-time image augmentation.  
   - Evaluated using confusion matrix and accuracy plots.

---

## 🧰 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
git clone https://github.com/yourusername/fake-image-detection.git
cd fake-image-detection

pip install -r requirements.txt

streamlit run "Fake-image app.py"


