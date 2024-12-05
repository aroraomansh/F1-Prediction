# F1 Race Winner & Strategy Prediction Using Machine Learning  

**Authors:**  
- Khushal Yadav (IIIT Delhi)  
- Naman Birla (IIIT Delhi)  
- Omansh Arora (IIIT Delhi)  

---

## Project Overview  
This repository contains the code and methodology for our machine learning project aimed at predicting Formula 1 (F1) race winners and optimizing pit stop strategies. The project leverages advanced ML techniques and datasets spanning F1 races from 1950 to 2024 to develop predictive models that assist in strategic decision-making.

---

## Features  
- **Winner Prediction Model:** Predicts the winner based on circuit, grid positions, and other race conditions using logistic regression with class balancing techniques (e.g., SMOTE).  
- **Pit Stop Strategy Optimizer:** Uses a Multi-Layer Perceptron (MLP) to identify the optimal number of pit stops and strategies for a race.  
- **Comprehensive Data Handling:** Preprocessing, feature engineering, and handling imbalanced datasets for robust predictions.  
- **Visualizations:** Correlation plots, learning curves, ROC curves, and more for in-depth model evaluation.  

---

## Dataset  
The dataset is sourced from Kaggle, containing historical F1 race data from 1950-2024, including:  
- Driver and constructor stats  
- Circuit details  
- Lap times, pit stops, and results  

For more details, refer to the [dataset source](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020).

---

## Methodology  

### 1. Data Preprocessing  
- Dropped irrelevant features and standardized numerical values for model compatibility.  
- Addressed class imbalance with SMOTE and weighted logistic regression.  

### 2. Pit Stop Strategy Model  
- **Architecture:** Enhanced MLP with ELU activation, Batch Normalization, and Dropout.  
- **Training:** Multi-class cross-entropy loss, Adam optimizer, early stopping.  

### 3. Winner Prediction Model  
- Logistic regression with hyperparameter tuning using GridSearch.  
- K-fold cross-validation for performance assessment.  

---

## Key Technologies  
- **Programming Languages:** Python  
- **Libraries & Frameworks:** PyTorch, Scikit-learn, Pandas, Matplotlib  

---

## Results  
### Pit Stop Strategy Model:  
- Macro Accuracy: **84.43%**  
- Micro Accuracy: **84.43%**  
- R² Score: **0.7594**  

### Winner Prediction Model:  
- Testing Accuracy: **87%**  
- F1-Score: **0.42**  

---

## How to Run  
1. Clone the repository:  
   ```bash  
   git clone <repository-url>  
   cd F1-Race-Prediction  
   ```  
2. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Prepare the dataset by following the preprocessing steps outlined in the `notebooks/` directory.  
4. Train models by running the respective Python scripts:  
   ```bash  
   python train_pit_strategy.py  
   python train_winner_prediction.py  
   ```  

---

## Learnings  
- Mastered EDA, feature engineering, and imbalanced data handling techniques.  
- Explored the impact of architecture and activation functions on MLP performance.  
- Gained insights into applying ML techniques to real-world problems.  

---

## Future Work  
- Incorporate real-time telemetry data for enhanced accuracy.  
- Expand to ensemble learning and deep learning techniques for improved winner prediction.  

---

## License  
This project is licensed under the MIT License. See `LICENSE` for more details.  

---

## Acknowledgments  
Special thanks to [Kaggle](https://www.kaggle.com) and [Ergast API](http://ergast.com/mrd/) for providing the dataset.  

--- 

Feel free to contribute by opening issues or submitting pull requests! 😊