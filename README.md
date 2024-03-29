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

## Evaluation Metrics
Model performance is evaluated using RMSE, MAE, and R-squared on the testing set to assess accuracy and predictive capability.

