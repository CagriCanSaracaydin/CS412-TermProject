# CS412-TermProject
# Instagram Influencer Analysis Project: Documentation

## 1. Overview of the Repository

This is the repository of the CS412: Machine Learning Course Term project. The project addresses two key tasks:

- **Classification**: Predicting one of ten influencer categories (Food, Tech, Gaming, Fashion, etc.).
- **Regression**: Predicting the like_count for Instagram posts.

### Key Files and Directories

1. **`main`**
     1. **Data Loading**: Combines labeled provided data with additional 150 user annotated data.
     2. **Feature Extraction**: Utilizes a custom feature extractor class.
     3. **Model Training**: Trains both classification and regression models with hyperparameter tuning.
     4. **Validation & Evaluation**: Splits the data for validation and prints out important metrics.
     5. **Prediction Generation**: Creates JSON files for classification and regression as the final output.

2. **`InfluencerFeatureExtractor`**
   - Handles the **profile-level** and **post-level** feature extraction parts.
   - Implements TF-IDF on aggregated captions and robust scaling for numeric columns.

3. **`train_classification_model`** & **`train_regression_model`**
   - Implements methods for **Random Forest** model training, **SMOTE** balancing for classification especially for limited categories(Gaming, Art, etc.), and **GridSearchCV** parameter optimization.

4. **Data Files**
   - **`train-classification.csv`**: Provided classification labels for some users.
   - **`annotated_users_*.csv`**: Personal annotation file containing 150 labeled users.
   - **`training-dataset.jsonl.gz`**: Provided main dataset with profiles and recent posts.
   - **Test Files**: `test-classification-round*.dat` and `test-regression-round*.jsonl` for final predictions for each rounds.

5. **Output Files**
   - **`prediction-classification-round*.json`**: Returns round specific predictions for category.
   - **`prediction-regression-round*.json`**: Returns round specific predictions for like_count.

## 2. Methodology

I followed and implemented the following steps for the important parts of the project.

### 2.1 Data Collection and Annotation
- **Step 1**: I started with the provided labeled dataset (`train-classification.csv`) mapping user IDs to specific influencer categories.
- **Step 2**: Then i imported personal annotations from the first step of the project(via Google Forms).
- **Step 3**: I **merged** the original labeled data with these annotated labels. This ensured we had a **unified set** of user-category mappings, avoiding conflicts by giving precedence to our new annotations when overlaps occurred.

### 2.2 Data Preprocessing and Feature Extraction
- **Profile-Level Features**: Follower/following/post counts, boolean flags (`is_verified`, `is_business_account`, `is_private`), presence of website/emails, and biography length.
- **Post-Level Features**:
  - Aggregated at the user level for classification: average likes, standard deviation of likes, average comments, hashtags, mentions, emojis, etc.
  - Used directly per post for regression: like_count, comment count, caption length, media type, etc.
- **Textual Representation**: TfidfVectorizer on post captions (removing Turkish stopwords, URLs, and non-turkish special characters).
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
- **Evaluation**: MSE and MAE calculated on the **log10 scale**; we also record **R²** for better understanding.

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
   - This resulted in a consolidated mapping of **2,775** labeled profiles (filtered for those with available post data).

**Key Finding**: The additional annotated data helped improving minority classes (Gaming, Mom and Children, etc.), making the final class distribution more balanced but still underrepresented.

### 3.2 Exploratory Data Analysis (EDA)
1. **Distribution of Categories**:  
   - Noticed skewed distribution: Food and Health & Lifestyle were more frequent, while Gaming and Mom and Children were notably smaller.
2. **Profile Statistics**:  
   - Follower counts ranged from very small (hundreds) to very large (millions).  
   - Confirmed the need for log transforms to reduce skew.
3. **Post-Level Observations**:  
   - Many posts had like counts heavily concentrated in lower ranges, with a tail for highly popular accounts.  
   - Comments often correlated with likes, indicating potential corelation between these two variables.

**Key Finding**: EDA confirmed that **data imbalance** and **skewed distributions** could significantly affect negatively during the model training. 

### 3.3 Handling Class Imbalance
1. **Minority Classes** (e.g., Gaming) had far fewer samples, risking poor recall and precision.  
2. **SMOTE** oversampling was applied during model training to synthesize new samples for underrepresented categories.

**Key Finding**: The use of SMOTE improved **balanced accuracy** for classification and helped the model better recognize smaller categories.

### 3.4 Feature Engineering and Transformation
1. **Profile and Post Features**: Constructed numerical columns (follower_count, following_count, average likes, etc.).  
2. **Textual Features**:  
   - Processed caption text with custom regex (removing URLs, punctuation, etc.).  
   - Used TfidfVectorizer with up to 5,000 features, unigrams and bigrams, Turkish stopword removal, and n-grams to capture short textual patterns.
3. **Scaling**:  
   - Applied log transformations to control the outliers in follower and like counts.  
   - Deployed a **RobustScaler** to reduce the effect of outliers in numeric features.

**Key Finding**: Combining numeric features (follower_count, comment_count, etc.) with text-based TF-IDF features provided a better representation of each user and post.

### 3.5 Hyperparameter Tuning and Model Selection
1. **Classification**:  
   - Explored multiple random forest configurations (e.g., `n_estimators` in [100, 200, 300], `max_features` in [‘sqrt’, ‘log2’], etc.).  
   - Chose the **best estimator** based on **balanced accuracy** from 5-fold cross-validation.
2. **Regression**:  
   - Similarly tested random forest regressors with `max_depth`, `min_samples_split`, and `max_features` variations.  
   - Selected the combination that maximized **R²** in cross-validation.

**Key Finding**:  
- For classification, a **non-bootstrapped** RandomForest with `n_estimators=300`, `max_features='sqrt'`, `min_samples_leaf=1`, `min_samples_split=2` performed best.  
- For regression, a **RandomForestRegressor** with `n_estimators=300`, `max_features='sqrt'`, `min_samples_split=5`, and `max_depth=None` gave the highest cross-validation R².

### 3.6 Validation and Metrics
1. **Validation Split** (80/20): Provided an unbiased estimate of performance.  
2. **Classification**:  
   - Overall validation **accuracy 61.8%**.  
   - Relatively high performance for well-represented classes (Food, Tech).  
   - Challenges remained with smaller classes (e.g., Gaming) despite SMOTE.
3. **Regression**:  
   - Achieved an **R² of 0.9227** on the validation set.  
   - **MSE (log10)** = 0.0803, indicating the model reliably predicts the log-transformed like counts.
4. **Feature Importance**:  
   - Comments count, follower count, and verification status were top features for regression.  
   - TF-IDF tokens related to food, travel, or child-related terms significantly influenced classification.

**Key Finding**:  
- The Regression model showed very strong predictive capacity for like counts.  
- The Classification model achieved moderate accuracy but could benefit from more sophisticated text representations.

### 3.7 Final Predictions and Output
   **JSON Generation**: Created the following submission files:  
   - `prediction-classification-round*.json`: A dictionary of `{username: "predicted_category"}`.  
   - `prediction-regression-round*.json`: A dictionary of `{post_id: predicted_like_count}`.

**Key Finding**:  
- Strictly following the **required JSON format** is crucial for succesfull submission.

## 4. Results and Experimental Findings
  Classification: Accuracy  0.6180
  Regression:     MSE (log10)  0.0803


1. **Classification**:
   - Accuracy of 61.8% across 10 classes is respectable, particularly given significant class imbalance.  
   - Detailed category-level precision/recall: Food and Tech categories show higher metrics, while smaller classes like Gaming remain more challenging.

2. **Regression**:
   - R² of 0.9227 means the model explains 92% of the variance in log10 like counts.  
   - Primary factors are numeric in nature (comments_count, follower_count, is_verified, etc.), reflecting core engagement details.

