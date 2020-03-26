# MACHINE DATA AND LEARNING ASSIGNMENT 1

**TEAM NUMBER:**  86

**TEAM MEMBERS:**

Tathagata Raha (2018114017)
Arathy Rose Tony(2018101042)

## TASKS

### QUESTION 1

Calculate the bias and variance of a dataset that is not sampled yet, and then predict the degree of the best fit curve

### QUESTION 2

- You have been provided with a training data and a testing data. You need to fit the given data to polynomials of degree 1 to 9(both inclusive).
- Specifically, you have been given 20 subsets of training data containing 400 samples each. For each polynomial, create 20 models trained on the 20 different subsets and find the variance of the predictions on the testing data. Also, find the bias of your trained models on the testing data. Finally plot the bias-variance trade-Off graph.
- Write your observations in the report with respect to underfitting, overfitting and also comment on the type of data just by looking at the bias-variance plot.

## CONTENTS

```
.
├── Q1
│   ├── Q1.ipynb
│   ├── Q1.py
│   ├── Q1_report.pdf
│   └── data.pkl
├── Q2
│   ├── Q2.ipynb
│   ├── Q2.py
│   ├── Q2_report.pdf
│   ├── X_test.pkl
|   ├── Fx_test.pkl
│   ├── X_train.pkl
│   └── Y_train.pkl
└── readme.md
```

## Pre-requisites

```less
pip3 install sklearn numpy pandas matplotlib
```

## How to run

Run a jupyter notebook. or do

```bash
cd Q1
python3 Q1.py
cd Q2
python3 Q2.py
```

The detailed reports can be found in pdf format.
