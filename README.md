# 🧠 Brain Tumor Detection Using Deep Learning
## 📌 Project Overview

This project demonstrates the development of a brain tumor detection system utilizing deep learning techniques on MRI images. The system employs Convolutional Neural Networks (CNNs) to classify MRI scans into categories such as glioma, pituitary, meningioma, and healthy tissue. The application is built using Python and leverages libraries like TensorFlow, Keras, and OpenCV.

---

## ⚙️ Technologies Used

- **Programming Language**: Python 3.x
- **Deep Learning Frameworks**: TensorFlow, Keras
- **Computer Vision Library**: OpenCV
- **Data Handling**: NumPy, Pandas
- **Model Evaluation**: Scikit-learn
- **User Interface**: Streamlit (for the web application interface)
- **Development Environment**: Jupyter Notebook, Visual Studio Code

---

## 🧪 Dataset Information

The model is trained on a dataset comprising MRI images categorized into four classes:

1. **Glioma Tumor**
2. **Meningioma Tumor**
3. **Pituitary Tumor**
4. **Healthy Brain**

*Note: Ensure that the dataset is properly labeled and preprocessed for optimal model performance.*

---

## 🚀 Setup and Installation

1. **Clone the Repository**:
    
    ```bash
    git clone https://github.com/yourusername/brain-tumor-detection.git
    cd brain-tumor-detection
    
    ```
    
2. **Create and Activate a Virtual Environment**:
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    
    ```
    
3. **Install Required Dependencies**:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
4. **Run the Application**:
    
    ```bash
    streamlit run app.py
    
    ```
    
    This command will launch the web application in your default browser.
    

---

## 🧪 Model Training

To train the model:

1. **Prepare the Dataset**: Organize the MRI images into folders named after their respective classes (e.g., `glioma`, `meningioma`, `pituitary`, `healthy`).
2. **Preprocess the Data**: Resize images to a consistent dimension (e.g., 224x224 pixels) and normalize pixel values.
3. **Train the Model**: Execute the training script:
    
    ```bash
    python train_model.py
    
    ```
    
    This will train the CNN model and save the trained weights to a file (e.g., `model.h5`).
    

---

## 🖼️ Usage

1. **Upload MRI Image**: Use the web interface to upload an MRI image.
2. **Model Prediction**: The system will preprocess the image, pass it through the trained model, and display the predicted class along with the associated probability.
3. **Visualization**: The application will also visualize the uploaded image and the prediction result.

---

## 📈 Model Evaluation

After training, evaluate the model's performance using metrics such as:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

These metrics can be computed using the `evaluate_model.py` script:

```bash
python evaluate_model.py

```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://chatgpt.com/c/LICENSE) file for details.

---

## 📞 Contact

For further inquiries or contributions, please contact:

- **Email**: [shakerajannatema@gmail.com](mailto:your.email@example.com)
- **GitHub**: [https://github.com/shakeraema](https://github.com/yourusername)

---

Feel free to modify this README to better fit your project's specifics and personal preferences.
