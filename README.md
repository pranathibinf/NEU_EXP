# Task 1: Gene Annotation Analysis with R
Task 1 involves downloading, parsing, and analyzing comprehensive gene annotation data for the primary assembly from the GENCODE database, specifically using Release 45. The R script provided accomplishes several tasks including downloading the GTF file, creating a TxDb object from it, computing statistics on the number of transcripts per gene, generating a histogram of these numbers, and saving the processed data for future use.

## Dataset
The dataset is available at https://www.gencodegenes.org/human/ use GTF of Comprehensive gene annotation for the primary assembly (chromosomes and scaffolding). Use release Release 45 (GRCh38.p14)

## Dependencies
R (version 3.6.0 or higher recommended)
Bioconductor packages:
rtracklayer
GenomicFeatures

## Running the Code
1) Download and Parse GTF File: The script starts by downloading the GTF file from the GENCODE website (Release 45). It then loads this file and parses it to create a TxDb object.
2) Analyze Transcripts: It computes the mean, minimum, and maximum number of transcripts per gene and generates a histogram of the number of transcripts per gene.
3) Save Processed Data: The script saves the transcripts-to-gene mapping as an .rds file for future analyses.

## Output
1) A histogram and a scattered bar plot titled "Histogram of Number of Transcripts per Gene" showing the distribution of transcripts across genes 
2) A .rds file named transcripts_to_genes.rds containing the S4 object with transcripts-to-gene mappings.
3) Console output detailing the mean, minimum, and maximum number of transcripts per gene.


# Task 2: Boosted Decision Tree Regression with XGBoost
Task 2 involves downloading the dataset, preprocessing its features and target variables, and constructing a boosted decision tree model using XGBoost for a regression task in Python. The dataset contains 1 million samples with 4D inputs and 2D outputs, from which either y1 or y2 is chosen for modeling.

## Dataset
The dataset is available at [  ](http://129.10.224.71/~apaul/data/tests/dataset.csv.gz)
It features 4D input (x1, x2, x3, x4) and 2D output (y1, y2)

## Features and Target Transformation
Features (x1) are scaled using a logarithmic function to normalize their range and improve model performance.
Target (y1 or y2) is transformed using a logarithmic function to handle skewness and enhance model accuracy.

## Requirements
Python 3.6+
Pandas
NumPy
scikit-learn
XGBoost
statsmodels
requests

## Data Preprocessing
1) Data Downloading: Automated from the specified URL.
2) Feature Scaling: Custom scaling function applied to x1.
3) Target Transformation: Custom scaling applied to y1/y2.

## Model Construction
Choice of Target: The model uses y1.
Data Splitting: The dataset is divided into training, validation, and testing sets, with specific splits chosen to balance training data availability and model validation/testing accuracy.
XGBoost Regression: Constructs a boosted decision tree model with careful selection and tuning of hyperparameters for optimal performance.

## Hyperparameter Tuning
The model's performance has been optimized through careful selection of hyperparameters in the XGBoost algorithm. Here's an overview of the chosen parameters and the rationale behind each choice:

1) max_depth: 9
Controls the maximum depth of trees. A depth of 9 was selected to complexly model the relationships in the data while preventing the model from becoming too deep, which could lead to overfitting.

2) eta: 0.1 (Learning Rate)
The learning rate was set at 0.1 to balance the speed of convergence with the model's ability to navigate the solution space effectively, avoiding overshooting the optimal solution.

3) min_child_weight: 10
This parameter helps control overfitting by making the algorithm more conservative. A higher value means the model will be more cautious in making splits that only add significant value, helping to ensure that the model generalizes well.

4) gamma: 0 (Minimum Loss Reduction)
Set to 0 for this model, indicating that no specific gain is required to make a split. This setting allows for flexibility in growing the trees, especially in the early phases of tuning and model exploration.

5) colsample_bytree: 1.0
Indicates that all features are used for each tree, maximizing the potential for each tree to learn from the entirety of the feature set. This choice supports comprehensive model learning from the data.

6) objective: reg:squarederror
Specifies the regression objective for the XGBoost model, targeting minimization of squared errors, which is suitable for continuous outcome prediction.

7) eval_metric: rmse (Root Mean Square Error)
Chosen as the evaluation metric to quantify the model's prediction accuracy, focusing on minimizing the average magnitude of the prediction errors.

These hyperparameters were selected based on their contributions to model performance, as evaluated through cross-validation and testing. The goal was to achieve a balance between learning efficiency, model complexity, and the ability to generalize to unseen data.## Evaluation Metrics
Model performance is evaluated using RMSE, MAE, and R-squared on the testing set to assess accuracy and predictive capability.

