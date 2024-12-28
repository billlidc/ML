<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>


# Part 1: Basic Concepts

## Table of Contents

- [Bias-Variance Tradeoff](#bias-variance-tradeoff)
- [Underfitting vs. Overfitting](#underfitting-vs-overfitting)
    - [Cross-Validation](#cross-validation)
    - [Fixing High Bias (Underfitting)](#fixing-high-bias-underfitting)
    - [Fixing High Variance (Overfitting)](#fixing-high-variance-overfitting)
- [Ensemble Methods](#ensemble-methods)
- [Loss Functions](#loss-functions)
- [Confusion Matrix](#confusion-matrix)
    1. Accuracy
    2. Recall
    3. Precision
- [ROC Curves, PR Curves](#roc-curves-pr-curves)

## Bias-Variance Tradeoff

<img src="./res/bias-and-variance.png" alt="" width="400">

Decomposition of **Expected Test Error**:

<div>
$$
\begin{align*}
\underbrace{\mathbb{E}_{x, y, D} \left[ \left( h_D(x) - y \right)^2 \right]}_{\text{Expected Test Error}} 
&= \underbrace{\mathbb{E}_{x, D} \left[ \left( h_D(x) - \bar{h}(x) \right)^2 \right]}_{\text{Variance}} 
+ \underbrace{\mathbb{E}_{x, y} \left[ \left( \bar{y}(x) - y \right)^2 \right]}_{\text{Noise}} 
+ \underbrace{\mathbb{E}_x \left[ \left( \bar{h}(x) - \bar{y}(x) \right)^2 \right]}_{\text{Bias}^2}
\end{align*}
$$
</div>

- **Variance**: How "over-specialized" (overfitting) is your classifier to a particular training set?
- **Bias**: What is the inherent error that you obtain from your classifier even with infinite training data?
    - Inherent to model
- **Noise**: How big is the data-intrinsic noise?
    - Inherent to data

## Underfitting vs. Overfitting

<img src="./res/bias_and_variance_contributing_to_total_error.png" alt="" width="400">

| **Characteristic**           | **Overfitting**                   | **Underfitting**               |
|------------------------------|-----------------------------------|--------------------------------|
| **Training Error**           | Low                               | High                           |
| **Validation/Test Error**    | High                              | High                           |
| **Model Complexity**         | Too complex                       | Too simple                     |
| **Generalization**           | Poor                              | Poor                           |
| **Bias vs. Variance**        | Low bias, high variance           | High bias, low variance        |

## Cross-Validation

### Process
1. Split the dataset into **k-folds** (e.g., 5 or 10)
2. Train the model on $k-1$ folds and validate it on the remaining fold
3. Repeat the process $k$ times, rotating the validation fold each time
4. Compute the average training and validation errors across all folds

### Model Generalization Evaluation
- **Low training error and high validation error** → Overfitting
- **High training error and high validation error** → Underfitting
- **Similar and low training and validation errors** → Good generalization

## Fixing **High Bias** (Underfitting)
1. Increase Model Complexity
2. Decrease Regularization
3. Add Features
4. Increase Training Time
5. Use Non-linear Models

## Fixing **High Variance** (Overfitting)
1. Decrease Model Complexity
2. Increase Regularization
3. Reduce Feature Space
4. Increase Training Data
5. Use Early Stopping
6. Use Ensemble Methods

## Balance: Bias-Variance Tradeoff
The solution often lies in striking a balance between high bias and high variance. You can experiment iteratively with:
- **Model selection**: Trying different algorithms and architectures
- **Hyperparameter tuning**: Adjusting hyperparameters like learning rate, regularization strength, or model depth
- **Feature engineering**: Improving the input data to enhance model performance


## Ensemble Methods

### Characteristics
- **(+) Improved accuracy**/predictive performance
- **(+) Improved robustness** to overfitting and noisy data
- **(+) Flexibility** to use different types of models
- (-) Increased complexity (requires more computational resources)
- (-) Harder to interpret compared to a single model

### 1. Bagging (Bootstrap Aggregating)
- Combines bootstrapping with aggregation
    - **Bootstrapping** is a statistical technique used to generate multiple datasets by randomly **sampling (with replacement)** from the original dataset. Each bootstrapped dataset is the same size as the original dataset but may contain duplicate samples.
- **Reduces variance**

### 2. Boosting
- Training weak learners sequentially to correct errors from the previous one
- **Reduces bias**

### 3. Stacking (Stacked Generalization)
- Improves predictive power by combining predictions from multiple models
- A meta-model is trained on the outputs of base models

### Ensemble Algorithms
1. **Random Forest**: Bagging applied to decision trees
2. **AdaBoost**: Boosting algorithm that combines decision trees or stumps
3. **Gradient Boosting Machines (GBM)**: Sequential training to minimize loss
4. **XGBoost**, **LightGBM**, **CatBoost**: Optimized implementations of gradient boosting with faster training and improved accuracy

## Loss functions



## Confusion Matrix

<img src="./res/confusion-matrix.png" alt="" width="400">

|                  | Predicted Positive | Predicted Negative |
|------------------|--------------------|--------------------|
| **True Positive**| TP                 | FN                 |
| **True Negative**| FP                 | TN                 |

1. **Accuracy**:
   - Formula: $$(TP + TN) / \text{all}$$
   - Measures how many cases (both positive and negative) are correctly classified

2. **Recall (Sensitivity/True Positive Rate)**:
   - Formula: $$TP / (TP + FN)$$
   - Measures how many actual positive cases are correctly classified
   - Sensitive to imbalanced data

3. **Precision**:
   - Formula: $$TP / (TP + FP)$$
   - Measures how many predicted positive cases are actually correct

4. **High Recall, Low Precision**:
   - Many false positives but few false negatives (good for detection)
   
   **Low Recall, High Precision**:
   - Few false positives but many false negatives (good for trustworthiness)

5. **F1 Score**:
   - Formula: $$2 \cdot \frac{\text{recall} \cdot \text{precision}}{\text{recall} + \text{precision}}$$
   - Harmonic mean of recall and precision, useful for imbalanced datasets

6. **Specificity (True Negative Rate)**:
   - Formula: $$TN / (TN + FP)$$
   - Measures the ability to correctly classify negatives

7. Severity of **False Positives vs. False Negatives**:
   - **False Positives** are worse in cases like:
     - Non-contagious diseases (unnecessary treatment)
     - HIV tests (psychological impact)
   - **False Negatives** are worse in cases like:
     - Early treatment importance
     - Quality control defects
     - Software testing (critical errors missed)

## ROC Curves, PR Curves

### ROC (Receiver Operating Characteristic) Curves

<img src="./res/roc-curve.png" alt="" width="250">

- ROC curve plots **recall (TPR)** vs. **1 - specificity (FPR)**
- **AUC (Area Under the Curve)**:
    - AUC near 1 indicates a good model
    - AUC near 0.5 indicates a model performing like random guessing

### Precision-Recall Curves

<img src="./res/pr-curve.png" alt="" width="250">

- PR curves are preferred when the dataset is highly imbalanced or when the focus is on the minority class detection
- Ignores true negatives; Used when classifier specificity is not a concern

- **AUC-PR (Area Under the PR Curve)**:
    - A higher AUC-PR indicates better performance
    - Point 1: **Low Recall, High Precision**
    - Point 2: Perfect model
    - Point 3: **High Recall, Low Precision**
    - Point 4: Trade-off
