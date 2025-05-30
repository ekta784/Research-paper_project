**Soil Quality and Nutrient Recommendation System Using Modified Multilayer Perceptron (MLP)**
*Advancing Precision Agriculture through Deep Learning*

**🎓 Research & Publications**
I’m proud to share that my this research paper titled Soil Quality and Nutrient Recommendation System Using Modified Multilayer Perceptron (MLP) was accepted at the ICIVC 2025 conference and published after presentation in ed Springer Book Series "Lecture Notes in Networks and Systems", reflecting contributions to the field of intelligent vision and computing.

**Overview:**
This project presents an intelligent, data-driven soil evaluation system that leverages a modified Multilayer Perceptron (MLP) model to classify soil quality and recommend essential nutrients. It addresses the challenges of traditional lab-based soil testing by providing a scalable, cost-effective, and real-time solution for sustainable farming practices.

**Objectives:**

* Automate soil quality classification using machine learning and deep learning techniques.
* Recommend nutrients based on deficiencies detected in the soil.
* Enhance precision agriculture by aiding informed, real-time decision-making.
* Support farmers with actionable insights to improve crop yields and land health.

**Dataset Used:**

* **Source:** LUCAS 2018 TOPSOIL Dataset
* **Samples:** 18,984 soil entries
* **Attributes:**

  * Soil pH (CaCl₂ and H₂O)
  * Organic Carbon
  * Nitrogen (N), Phosphorus (P), Potassium (K)
  * Calcium Carbonate (CaCO₃)
  * Electrical Conductivity
  * Oxalate-extractable Iron (Fe) and Aluminium (Al)

**Technology Stack and Purpose:**

1. **Python**:
   Used as the core development language due to its readability and strong ecosystem for data science.

2. **Pandas and NumPy**:
   Employed for efficient data preprocessing, handling missing values, and performing statistical computations.

3. **Scikit-learn**:
   Used for data standardization (`StandardScaler`), model comparison (Random Forest, SVC, KNN, etc.), and metric evaluations.

4. **Matplotlib and Seaborn**:
   Used for data visualization, helping interpret training performance, ROC-AUC curves, and loss functions.

5. **TensorFlow / Keras**:
   Implemented the customized Multilayer Perceptron (MLP) architecture, including multiple hidden layers, ReLU activations, dropout regularization, and softmax output layer.

6. **Jupyter Notebook / Kaggle Notebook**:
   Development and testing environment providing visual and step-by-step analysis of the model’s pipeline.

**Model Architecture:**

* **Initial Model**:

  * Two hidden layers with ReLU activation
  * Softmax output layer
  * Accuracy: 97.91%

* **Enhanced Model**:

  * Added third hidden layer with 16 neurons
  * Dropout regularization (30% in first two layers, 20% in third)
  * Hyperparameter tuning
  * **Final Accuracy Achieved**: **98.72%**

**Performance Metrics:**

| Class     | Precision | Recall | F1-Score | AUC-ROC |
| --------- | --------- | ------ | -------- | ------- |
| Excellent | 0.9982    | 0.9842 | 0.9912   | 0.9999  |
| Fair      | 0.9740    | 0.9868 | 0.9804   | 0.9996  |
| Good      | 0.9773    | 0.9989 | 0.9830   | 0.9999  |
| Moderate  | 0.9920    | 0.9860 | 0.9890   | 0.9999  |
| Poor      | 0.9973    | 0.9553 | 0.9758   | 0.9999  |

* **Overall Accuracy:** 98.72%
* **Loss Stabilized Below:** 0.4 after 50 epochs
* **Validation Accuracy:** Reached 85% consistently

**Comparison with Other Models:**

| Model                     | Accuracy (%) |
| ------------------------- | ------------ |
| K-Nearest Neighbor (KNN)  | 93.80%       |
| Random Forest Classifier  | 96.38%       |
| Extra Trees Classifier    | 96.58%       |
| Support Vector Classifier | 97.38%       |
| LightGBM                  | 97.31%       |
| MLP (Original)            | 97.91%       |
| **MLP (Modified)**        | **98.72%**   |


**Future Enhancements:**

* Build a **web/mobile application** for farmer accessibility.
* Integrate **IoT sensors and satellite data** for real-time monitoring.
* Support **multilingual interfaces** for rural outreach.
* Use **transfer learning** to adapt to region-specific soil.
* Develop explainable AI (XAI) visualizations for better model transparency.


**Conclusion:**
The Modified Multilayer Perceptron (MMLP) model significantly improves the accuracy of soil classification over traditional methods. With high performance, generalizability, and potential for deployment in real-world agricultural scenarios, this solution contributes meaningfully to modern precision agriculture.

**Author:**
Ekta Vora,Tishya Patel,Tirth patel ,B.Tech in Computer Engineering
Email: [ektavora0708@gmail.com]
LDRP Institute of Technology and Research, Gujarat

