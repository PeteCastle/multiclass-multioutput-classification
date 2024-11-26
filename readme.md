# Predicting Industry and Job Roles from Job Descriptions: A Multi-Task Learning Perspective

Learning Team 3
- Francis Mark Cayco
- Marianne Reyes
- Imee Lanie Uy
- Denver Valera

**COSCI 221 - Machine Learning 1** <br>
Master of Science in Data Science 2025

## Introduction
Many prediction tasks demand multiple outputs, not just a single label. Multi-output learning, a subset of supervised learning, is designed to handle these scenarios. Unlike traditional classification, multi-output models can predict multiple interdependent labels simultaneously.
Multi-output learning can be categorized based on the number and nature of target variables, as illustrated in the table below:

| Classification Type | Number of targets | Target cardinality | Valid `type_of_target` |
|---|---|---|---|
| Multiclass classification | 1 | >2 | `'multiclass'` |
| Multilabel classification | >1 | 2 (0 or 1) | `'multilabel-indicator'` |
| Multiclass-multioutput classification | >1 | >2 | `'multiclass-multioutput'` |
| Multioutput regression | >1 | Continuous | `'continuous-multioutput'` |

*Table 1: Types of Multiclass and Multioutput Tasks*

### Problem Statement
Job platforms often provide job descriptions along with specific identifiers like industry and employment status (e.g., full-time, part-time, contractual). The study aims to address the question: Can these identifiers be accurately predicted solely from the job description?

The study will utilize the Job Description dataset from Kaggle. The primary objective is to build a model capable of predicting the correct industry and employment status based solely on the provided job description.

**Multioutput Classification**

To address this multi-output prediction problem, the group will employ a Multi-Output Classification approach as it involves >1 targets and >2 target cardinality. The job descriptions will serve as input features, while the target variables will be the job type and category. By training a multi-output model, the group aims to accurately classify job listings based on their textual content.

![image.png](attachment:image.png)

### Highlights 

1. Complexity 

Multi-output learning models offer a generalized approach to simultaneously predicting multiple target variables. However, interpreting these models can be complex due to the intricate interplay between multiple outputs. As the number of target variables and their interdependencies increase, the model must learn increasingly complex relationships. To effectively capture these relationships and enhance performance, a substantial amount of data is often required. Larger datasets enable the model to identify subtle patterns and correlations. 

Additionally, assessing the independence of target variables is crucial. Multi-output models generally perform better when the targets are interdependent. The ClassifierChain, a multi-output algorithm that sequentially trains binary classifiers, leverages these interdependencies. Each classifier's prediction serves as an input for the next, enabling the model to capture complex relationships and improve overall predictive accuracy. 

 

2. Error Propagation 

Multioutput algorithms are susceptible to errors due to their simultaneous prediction of multiple target variables. An error in one output can diminish the overall accuracy, particularly when outputs have varying importance. Imbalances in a specific target variable can further exacerbate this issue. This is especially pronounced when target variables are interdependent. For instance, misclassifying a job type while correctly identifying the industry category can lead to practical challenges for both employers and job seekers, as these outputs are often used in tandem for decision-making. 

 

3. Evaluation Metrics 

The multiclass multioutput scikit-learn library does not have a built-in scoring metric. This necessitates the creation of custom evaluation metrics to assess the model's performance. One approach is to calculate accuracy for each target variable individually and then combine these scores into a weighted average. Alternatively, comparing the performance of a multi-output model against separate single-output models can provide insights into the trade-offs between complexity and performance. However, it's important to consider the computational cost and training time associated with multi-output models. A more complex model may require significant computational resources and longer training times. Therefore, it's crucial to balance model complexity with practical considerations like computational efficiency and deployment time. 

## Results and Discussion

Based on the results of the exploratory data analysis, the dataset is imbalanced in terms of job types and categories. The distribution of job types is skewed towards "Permanent" roles, while certain job categories are more prevalent than others. This imbalance could affect the model's performance, especially for less common job types or categories. The Chi-Square test of independence revealed that the job type and job category labels are independent of each other.

The results of the Singular Vector Decomposition (SVD) showed that the first 1768 components explain 85% of the variance in the dataset. This reduction in dimensionality can help improve the model's performance by capturing the most important features while reducing noise.

The dataset is trained in two phases: one where the labels are treated independently and another where they are treated as dependent. Results show that the `LogisticRegression` model performed best in both scenarios, with the highest average F1 score of ~71%. Other tree-related models also performed well.  

The fact that classifier chains (which model dependencies between labels) didn't significantly outperform the multioutput classifier (which assumes independence) suggests that the ***relationships between the labels might be weak or not strongly predictive***.  This is consistent with the results of the Chi-Square test of independence, which indicated that the two labels are not significantly associated with each other.

In this case, the added complexity of modeling dependencies might not be necessary. Simpler models like the multioutput classifier could be sufficient for this type of dataset for interpretability purposes. 

## Conclusion
***A Multi-Output Approach to Improving Job Listing Classification***
This study aimed to predict the industry and employment status of job listings based solely on their textual descriptions. To address this multi-output classification problem, we employed two primary models: MultiOutput Classification and ClassificationChain. A key consideration in multi-output classification is the potential dependence between labels. By capturing these relationships, models can achieve improved accuracy and generalization, particularly in applications like job matching and recommendations. 

**Model Performance and Insights**
- **Independence Assumption**: Training the MultiOutput Logistic Regression model under the assumption of label independence yielded strong performance, outperforming K-Nearest Neighbors and Decision Trees in terms of F1-score. 

- **Dependence Assumption**: The ClassifierChain Logistic Regression model, trained assuming label dependence, achieved the highest overall accuracy. 

- **Target Variable Dependence**: While the chi-squared test indicated independence between the target variables, the dominance of the pharmaceutical industry in the dataset might have influenced this result. Further analysis on a more balanced dataset could provide deeper insights into label dependencies. 

- **Computational Efficiency**: Balancing accuracy and computational efficiency is crucial in multi-output models. Techniques like feature selection and dimensionality reduction can help optimize performance. 

- **Multi-Output vs. Single-Output**: Comparing multi-output and single-label models resulted in no significant difference as demonstrated by the KNeighborsClassifier. This shows that `MultiOutputClassifier()` operates similarly to training individual models for each label and then combining their outputs. 

While the limitations of the dataset may not have fully captured the potential advantages of multi-output algorithms, their benefits are evident in scenarios involving interdependent labels, optimal hyperparameter tuning of multiple target variables, and the desire for a unified framework for training and deploying models. 

This study successfully applied MultiOutput Logistic Regression and ClassifierChain Logistic Regression to predict multiple target variables from textual job descriptions. By leveraging the strengths of this approach and exploring future advancements, we can further enhance the accuracy and efficiency of job classification models, benefiting both job seekers and employers. 

## References:

1.  phiyodr. (2024). multilabel-oversampling [Computer software]. GitHub. Retrieved November 26, 2024, from https://github.com/phiyodr/multilabel-oversampling
2.  [Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2020). Classifier chains: A review and perspectives. arXiv.](https://arxiv.org/pdf/1912.13405)
3.  Scikit-learn developers. (2024). 1.12. Multiclass and multioutput algorithms — scikit-learn 1.5.2 documentation. Retrieved November 25, 2024, from https://scikit-learn.org/1.5/modules/multiclass.html

