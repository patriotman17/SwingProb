import pandas as pd
import uuid
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#load in data
year1 = pd.read_csv("year1.csv")
year2 = pd.read_csv("year2.csv")
year3 = pd.read_csv("year3.csv")


year3.head()  # Display the first few rows of the Year 3 data


#%% Data Preprocessing 

# Combine year1 and year2 for training
combined = pd.concat([year1, year2])

# Check for missing values
print(combined.isnull().sum())
print(year3.isnull().sum())


#Filling in pitch type missing values with unknown
combined['pitch_type'] = combined['pitch_type'].fillna('Unknown')
year3['pitch_type'] = year3['pitch_type'].fillna('Unknown')


# Function to generate a unique ID for the missing pitch id's
def generate_unique_id():
    return str(uuid.uuid4())

# Applying function to fill in missing 'pitch_id' values
combined['pitch_id'] = combined['pitch_id'].apply(lambda x: generate_unique_id() if pd.isna(x) else x)
year3['pitch_id'] = year3['pitch_id'].apply(lambda x: generate_unique_id() if pd.isna(x) else x)

# Checking for any duplicate 'pitch_id'
duplicates = combined['pitch_id'].duplicated().sum()
print("Number of duplicates in pitch_id:", duplicates)

# Removing rows where the pitch_type is 'Unknown'
filtered_combined = combined[combined['pitch_type'] != 'Unknown']
filtered_year3 = year3[year3['pitch_type'] != 'Unknown']

# Checking the shape of the new dataset to confirm rows are removed
print("New dataset shape:", filtered_combined.shape)
print("New dataset shape:", filtered_year3.shape)


# Calculate median release_speed for each pitch_type
medians = filtered_combined.groupby('pitch_type')['release_speed'].median()
medians_y3 = filtered_year3.groupby('pitch_type')['release_speed'].median()


# Function to apply that fills missing release_speed with the median of the respective pitch_type
def fill_with_median(row):
    if pd.isna(row['release_speed']):
        return medians[row['pitch_type']]
    return row['release_speed']

def fill_with_median1(row):
    if pd.isna(row['release_speed']):
        return medians_y3[row['pitch_type']]
    return row['release_speed']

# Apply the function
filtered_combined['release_speed'] = filtered_combined.apply(fill_with_median, axis=1)
filtered_year3['release_speed'] = filtered_year3.apply(fill_with_median1, axis=1)


# Function to fill missing values by median within groups
def fill_median_within_group(df, group_cols, fill_cols):
    for col in fill_cols:
        # Compute the median within the specified group
        group_median = df.groupby(group_cols)[col].transform('median')
        # Fill missing values using these medians
        df[col] = df[col].fillna(group_median)
    return df

# Columns to fill
tracking_cols = ['pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'sz_top', 'sz_bot']

# Apply the function
filtered_combined = fill_median_within_group(filtered_combined, ['pitcher', 'pitch_type'], tracking_cols)
filtered_year3 = fill_median_within_group(filtered_year3, ['pitcher', 'pitch_type'], tracking_cols)

# Check how many values remain unfilled (could still be NaN if entire subgroup is NaN)
print(filtered_combined[tracking_cols].isnull().sum())
print(filtered_year3[tracking_cols].isnull().sum())

#removing the one row with unknown in year 3 dataset
filtered_year3 = filtered_year3.dropna(subset=['pfx_z'])

# One-hot encode the 'stand' and 'p_throws' columns
filtered_combined = pd.get_dummies(filtered_combined, columns=['stand', 'p_throws'])
filtered_year3 = pd.get_dummies(filtered_year3, columns=['stand', 'p_throws'])


# Display the first few rows to verify the encoding
print(filtered_combined.head())

#%% Data Exploration

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# histogram of release_speed for each pitch type using FacetGrid
g = sns.FacetGrid(filtered_combined, col="pitch_type", col_wrap=4, height=3)
g.map(sns.histplot, "release_speed", kde=False)


g.set_titles("Pitch Type: {col_name}")
g.set_axis_labels("Release Speed (mph)", "Frequency")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Distribution of Release Speed by Pitch Type')

# Show plot
plt.show()


# Bar chart for pitch_type frequencies
filtered_combined['pitch_type'].value_counts().plot(kind='bar')
plt.title('Frequency of Pitch Types')
plt.xlabel('Pitch Type')
plt.ylabel('Count')
plt.show()

# Aggregate the data to count the frequency of each description for each pitch type
outcome_counts = filtered_combined.groupby(['pitch_type', 'description']).size().reset_index(name='counts')

# Split the dataset based on 'pitch_type'
pitch_types = outcome_counts['pitch_type'].unique()
num_pitch_types = len(pitch_types)
pitch_types_set1 = pitch_types[:num_pitch_types // 2]
pitch_types_set2 = pitch_types[num_pitch_types // 2:]

# Function to create a grid of barplots for pitch outcomes
def plot_outcome_distribution(pitch_types_set, outcome_counts_df, title_suffix=''):
    # Filter the outcome_counts for the specific set of pitch types
    df_subset = outcome_counts_df[outcome_counts_df['pitch_type'].isin(pitch_types_set)]
    
    # Initialize the FacetGrid object
    g = sns.FacetGrid(df_subset, col="pitch_type", col_wrap=4, sharey=False, height=5, aspect=1)
    g.map_dataframe(sns.barplot, x='description', y='counts', palette='bright')
    
    # Improve the visibility of x labels
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_size(9)
        ax.set_xlabel('Outcome', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
    
    # Set the title for each facet
    g.set_titles("{col_name}", size=12)
    
    # Adjust the layout and show the FacetGrid
    g.fig.subplots_adjust(hspace=0.5, wspace=0.3)
    g.fig.suptitle(f'Frequency of Pitch Outcomes by Pitch Type{title_suffix}', size=16)
    plt.show()

# Plot first set of pitch types
plot_outcome_distribution(pitch_types_set1, outcome_counts, title_suffix=' (Set 1)')

# Plot second set of pitch types
plot_outcome_distribution(pitch_types_set2, outcome_counts, title_suffix=' (Set 2)')

# analyzing release speeds vs pitch type
# listing common outcomes of a pitch type
common_outcomes = ['hit_into_play','ball', 'called_strike','swinging_strike' ,'foul', 'foul_bunt','foul_tip', 'in_play']
filtered_data = filtered_combined[filtered_combined['description'].isin(common_outcomes)]

# box plot of release_speed for each outcome
plt.figure(figsize=(12, 6))
sns.boxplot(x='description', y='release_speed', data=filtered_data)
plt.title('Release Speed by Pitch Outcome')
plt.xlabel('Pitch Outcome')
plt.ylabel('Release Speed (mph)')
plt.xticks(rotation=45)
plt.show()


#%% Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

#Encoding the pitch type column
filtered_combined = pd.get_dummies(filtered_combined, columns=['pitch_type'])
filtered_year3 = pd.get_dummies(filtered_year3, columns=['pitch_type'])

# Select features
features = ['release_speed', 'stand_L', 'stand_R', 'p_throws_L', 'p_throws_R', 'balls', 'strikes',
           'pitch_type_CH', 'pitch_type_CS', 'pitch_type_CU', 'pitch_type_EP', 
           'pitch_type_FA', 'pitch_type_FC', 'pitch_type_FF', 'pitch_type_FS', 
           'pitch_type_KC', 'pitch_type_KN', 'pitch_type_PO', 'pitch_type_SC', 
           'pitch_type_SI', 'pitch_type_SL', 'pitch_type_ST', 'pitch_type_SV',
           'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'sz_top', 'sz_bot']

# Create a function to determine if the description implies a swing
def is_swing(description):
    swing_indicators = ['swinging_strike', 'foul', 'hit_into_play','swinging_strike_blocked']
    return int(any(indicator in description.lower() for indicator in swing_indicators))

# Apply the function to the 'description' column to create the target variable
filtered_combined['swing'] = filtered_combined['description'].apply(is_swing)


# Prepare the data
X = filtered_combined[features]  # Features
y = filtered_combined['swing']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter distribution for Random Search
param_dist = {
    'n_estimators': [100, 150],  # Reduced number of trees
    'max_depth': [10, 20],       # Reduced maximum depth
    'min_samples_split': [5, 10],  # Increased minimum number of samples required to split
    'min_samples_leaf': [2, 4],   # Increased minimum number of samples at each leaf
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,   # Reduced number of iterations
    cv=2,        # Reduced cross-validation folds
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'
)

random_search.fit(X_train, y_train)

#checking after the random search
print("Best Parameters:", random_search.best_params_)
print("Best ROC-AUC Score:", random_search.best_score_)


# Use the best estimator to make predictions
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# Evaluate the model on the test data
print("ROC-AUC on Test Data:", roc_auc_score(y_test, y_prob))
print("Accuracy on Test Data:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Applying the best model onto the year 3 data
X_year3 = filtered_year3[features]
filtered_year3['SwingProbability'] = best_rf.predict_proba(X_year3)[:, 1]

#Changing the swing probability column to percentages
filtered_year3['SwingProbability'] = filtered_year3['SwingProbability'] * 100

# Saving the predicted probabilities in the Year 3 dataset and exporting to csv
filtered_year3.to_csv('validation.csv', index=False)

#%% Additional plots for the model
from sklearn.metrics import roc_curve, auc


# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

# Feature Importance Plot
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Distribution of Predicted Probabilities
plt.figure()
sns.histplot(filtered_year3['SwingProbability'], kde=True)
plt.title('Distribution of Predicted Swing Probabilities')
plt.xlabel('Swing Probability (%)')
plt.ylabel('Frequency')
plt.show()
