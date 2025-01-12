# CS412-TermProject - Çağrı Can Saraçaydın - 30984
# Instagram Influencer Analysis Project: Documentation

## 1. Overview

This is the repository of the CS412: Machine Learning Course Term project. This project consists of two main parts:

- Classification by predicting the account types of the provided instagram accouns(10 categories).
- Regression for the predicting the like counts of the provided posts.

### Key Files and Directories

1. **`main`**
     1. **Data Loading**: Loads provided labeled dataset and additionally my annotated 150 data.
     2. **Feature Extraction**: Uses feature extractor class made for the instagram dataset.
     3. **Model Training**: Training the classification and regression models with hyperparameter.
     4. **Validation & Evaluation**: Splitting the data for validation and printing the important evaluation results for better understanding.
     5. **Prediction Generation**: Creates JSON files for classification and regression as the final output.

2. **`InfluencerFeatureExtractor`**
   - Handles the profile and post related feature extraction.
   - Implements TF-IDF on aggregated captions and robust scaling for numeric columns.

3. **`train_classification_model`** & **`train_regression_model`**
   - Implements methods for Random Forest model training, SMOTE balancing for classification especially for limited categories(Gaming, Art, etc.), and GridSearchCV parameter optimization.

4. **Data Files**
   - **`train-classification.csv`**: Provided classification data.
   - **`annotated_users_*.csv`**: My personal annotation file containing 150 labeled users.
   - **`training-dataset.jsonl.gz`**: Provided main dataset with profiles and recent posts.
   - **Test Files**: `test-classification-round*.dat` and `test-regression-round*.jsonl` for final predictions for each rounds.

5. **Output Files**
   - **`prediction-classification-round*.json`**: Returns round specific predictions for category.
   - **`prediction-regression-round*.json`**: Returns round specific predictions for like_count.

## 2. Methodology

### 2.1 Data Collection and Annotation
- **Step 1**: I started with the provided labeled dataset (`train-classification.csv`) mapping user IDs to specific influencer categories.
- **Step 2**: Then i imported personal annotations from the first step of the project(Google Forms).
- **Step 3**: I merged the original labeled data with these annotated labels. This ensured we had a larger dataset, avoiding conflicts by giving precedence to our new annotations when overlaps occurred.

### 2.2 Data Preprocessing and Feature Extraction
- **Profile Features**: Follower/following/post counts, boolean flags (`is_verified`, `is_business_account`, `is_private`), presence of website/emails, and biography length.
- **Post Features**:
  - Aggregated at the user level for classification: average likes, standard deviation of likes, average comments, hashtags, mentions, emojis, etc.
  - Used directly per post for regression: like_count, comment count, caption length, media type, etc.
- **Text**: TfidfVectorizer on post captions (removing Turkish stopwords, URLs, and non-turkish special characters).
- **Scaling & Transformations**:
  - **Log** transformations on skewed features (like `follower_count` and `avg_likes`).
  - **RobustScaler** for numerical columns to reduce the impact of outliers.

### 2.3 Classification Approach
- **Random Forest Classifier**: Chosen for robust performance on structured/tabular data.
- **SMOTE**: Applied to balance the classes for better performance (e.g., Gaming were underrepresented).
- **GridSearchCV**: Performed hyperparameter tuning across `n_estimators`, `max_features`, `min_samples_leaf`, etc.
- **Validation**: Used a 80-20 train–validation split, ensuring balanced category representation in training and validation subsets.

### 2.4 Regression Approach
- **Random Forest Regressor**:
  - Target variable: `log10(like_count + 1)` to address heavy right-skew.
  - Similar hyperparameter tuning with `GridSearchCV`.
- **Evaluation**: MSE and MAE calculated on the log10 scale; i also record R² for better understanding.

### 2.5 Prediction Generation
- **Classification**: Predicted influencer categories for each username in the test set.  
- **Regression**: Predicted like counts (in log10, then exponentiated to yield the final integer) for each test post.  
- **Output**: JSON files (`prediction-classification-round*.json` and `prediction-regression-round*.json`) for each round submission.

## 3. Main Analysis and Findings

This section provides more details on for each step, along with the findings.

### 3.1 Merging Existing and Annotated Data
1. **Data Loading**:  
   - Loaded `train-classification.csv` containing 2,800 accounts labeled into 10 categories.  
   - Gathered `annotated_users_*.csv` that included custom-labeled users.
2. **Deduplication and Conflict Resolution**:  
   - When a username appeared in both datasets, priority was given to the newly annotated label.  
   - This resulted in a consolidated mapping of 2,775 labeled profiles (filtered for those with available post data).

**Key Finding**: The additional annotated data helped improving minority classes (Gaming, Mom and Children, etc.), making the final class distribution more balanced but still underrepresented.

### 3.2 Exploratory Data Analysis (EDA)
1. **Distribution of Categories**:  
   - Found out skewed distribution: Food and Health & Lifestyle were more frequent, while Gaming and Mom and Children were extremly small.
2. **Profile Statistics**:  
   - Follower counts ranged from very small (hundreds) to very large (millions).  
   - Confirmed the need for log transforms to reduce skew.
3. **Post Observations**:  
   - Many posts had like counts heavily concentrated in lower ranges, with a tail for highly popular accounts.  
   - Comments often correlated with likes, indicating potential corelation between these two variables.

**Key Finding**: EDA confirmed that data imbalance and skewed distributions could significantly affect negatively during the model training. 

### 3.3 Handling Class Imbalance
1. **Minority Classes** (e.g., Gaming) had fewer samples, causing poor recall and precision.  
2. **SMOTE** oversampling was applied during model training to synthesize new samples for underrepresented categories.

**Key Finding**: The use of SMOTE improved balanced accuracy for classification and helped the model better recognize smaller categories.

### 3.4 Feature Engineering and Transformation
1. **Profile and Post Features**: Constructed numerical columns (follower_count, following_count, average likes, etc.).  
2. **Textual Features**:  
   - Processed caption text with custom regex (removing URLs, punctuation, etc.).  
   - Used TfidfVectorizer with up to 5,000 features, unigrams and bigrams, Turkish stopword removal, and n-grams to capture short textual patterns.
3. **Scaling**:  
   - Applied log transformations to control the outliers in follower and like counts.  
   - Deployed a RobustScaler to reduce the effect of outliers in numeric features.

**Key Finding**: Combining numeric features (follower_count, comment_count, etc.) with text-based TF-IDF features provided a better representation of each user and post.

### 3.5 Hyperparameter Tuning and Model Selection
1. **Classification**:  
   - Explored multiple random forest configurations (e.g., `n_estimators` in [100, 200, 300], `max_features` in [‘sqrt’, ‘log2’], etc.).  
   - Chose the best estimator based on balanced accuracy from 5-fold cross-validation.
2. **Regression**:  
   - Similarly tested random forest regressors with `max_depth`, `min_samples_split`, and `max_features` variations.  
   - Selected the combination that maximized R² in cross-validation.

**Key Finding**:  
- For classification, RandomForest with `n_estimators=300`, `max_features='sqrt'`, `min_samples_leaf=1`, `min_samples_split=2` performed best.  
- For regression, a RandomForestRegressor with `n_estimators=300`, `max_features='sqrt'`, `min_samples_split=5`, and `max_depth=None` gave the highest cross-validation R².

### 3.6 Validation and Metrics
1. **Validation Split** (80/20) 
2. **Classification**:  
   - Overall validation accuracy 61.8%.  
   - Relatively high performance for well-represented classes (Food, Tech).  
   - Problems remained with smaller classes (e.g., Gaming) despite SMOTE.
3. **Regression**:  
   - Achieved an R² of 0.9227 on the validation set.  
   - MSE (log10) = 0.0803, indicating the good overal performance for the log-transformed like counts.
4. **Feature Importance**:  
   - Comments count, follower count, and verification status were top features for regression.  
   - Turkish food, travel, or child-related terms significantly influenced classification.

**Key Finding**:  
- The Regression model showed very strong predictive capacity for like counts.  
- The Classification model achieved moderate accuracy.

### 3.7 Final Predictions and Output
   **JSON Generation**: Created the following submission files following it strictly:  
   - `prediction-classification-round*.json`: A dictionary of `{username: "predicted_category"}`.  
   - `prediction-regression-round*.json`: A dictionary of `{post_id: predicted_like_count}`.

**Key Finding**:  
- Strictly following the required JSON format is crucial for succesfull submission.

## 4. Results and Experimental Findings
  Classification: Accuracy  0.6180
  Regression:     MSE (log10)  0.0803


1. **Classification**:
   - Accuracy of 61.8% across 10 classes was good but not great, particularly given significant class imbalance.  
   - Detailed category-level precision/recall: Food and Tech categories show higher metrics, while smaller classes especially the Gaming still suffers from its underrepresentation.

2. **Regression**:
   - R² of 0.9227 means the model explains 92% of the variance in log10 like counts.  
   - Primary factors are numeric in nature (comments_count, follower_count, is_verified, etc.), reflecting core engagement details.

## 5. Team Members

I worked alone on this term project, implemented everything by myself.

Çağrı Can Saraçaydın - 30984