import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('StudentPerformanceFactors.csv')
df.head(5)
df.shape
df.shape
df.info()
df.describe()
df.isnull().sum()
categorical_columns_with_missing = ['Parental_Education_Level', 'Distance_from_Home', 'Teacher_Quality']

for column in categorical_columns_with_missing:
    mode_value = df[column].mode()[0]  
    df[column].fillna(mode_value, inplace=True)
df.sample(5)
df.duplicated().sum()
df.nunique()
numerical_features = df[['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']]
correlation_matrix = numerical_features.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Purples', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()
fig = px.histogram(
    data_frame=df,
    x='Exam_Score', 
    nbins=20, 
    marginal='box', 
    color_discrete_sequence=['#6c3483'], 
    text_auto=True,
    title='Distribution of Exam Scores'
)

fig.show()

fig = px.histogram(
    data_frame=df,
    x='Hours_Studied', 
    nbins=25, 
    marginal='rug', 
    color_discrete_sequence=['#3498db'], 
    text_auto=True, 
    title='Distribution of Hours Studied'
)

fig.show()
fig = px.histogram(
    data_frame=df,
    x='Attendance', 
    nbins=30, 
    marginal='violin',  
    color='School_Type', 
    color_discrete_sequence=['#1f77b4', '#ff7f0e'], 
    text_auto=True,  
    title='Attendance Distribution by School Type' 
)

fig.show()
fig = px.histogram(
    data_frame=df,
    x='Motivation_Level', 
    nbins=10,  
    marginal='box',  
    color_discrete_sequence=['#f39c12'],
    text_auto=True,  
    title='Motivation Level Distribution'
)

fig.show()
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Parental_Involvement', hue='Gender', palette='Set2')
plt.title('Parental Involvement Distribution')
plt.xlabel('Parental Involvement')
plt.ylabel('Count')
plt.show()
sns.pairplot(numerical_features)
plt.suptitle("Pair Plot of Numerical Features", y=1.02)
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Gender', y='Exam_Score', palette='Set2')
plt.title('Exam Score Distribution by Gender')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='School_Type', y='Exam_Score', palette='Set3')
plt.title('Exam Score Distribution by School Type')
plt.show()
g = sns.FacetGrid(df, col="Gender", row="School_Type", margin_titles=True, height=5)
g.map(sns.histplot, "Exam_Score", kde=True, color="blue")
g.set_axis_labels('Exam Score', 'Frequency')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Exam Score Distribution by Gender and School Type')
plt.show()
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Parental_Involvement', y='Exam_Score', palette='Set2')
plt.title('Exam Score Distribution by Parental Involvement')
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Hours_Studied', y='Exam_Score', hue='Gender', palette='Set1')
plt.title('Hours Studied vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()
correlation_with_target = numerical_features.corr()['Exam_Score'].sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values, palette='coolwarm')
plt.title('Correlation of Features with Exam Score')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.show()
categorical_columns = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
    'Motivation_Level', 'Internet_Access', 'Family_Income','Teacher_Quality', 'School_Type', 
    'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level','Distance_from_Home','Gender'
]

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df.sample(5)
categorical_columns = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
    'Motivation_Level', 'Internet_Access', 'Family_Income','Teacher_Quality', 'School_Type', 
    'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level','Distance_from_Home','Gender'
]

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df.sample(5)
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lf = LinearRegression()
lf.fit(X_train, y_train)

lf_pred = lf.predict(X_test)

lf_r2 = r2_score(y_test, lf_pred)

print(f"R2 Score = {lf_r2}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lf_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal fit')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.legend()
plt.show()
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(lf, X_train, y_train, cv=5)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score', color='blue', marker='o')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Cross-validation score', color='red', marker='o')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.show()
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': lf_pred})
print(comparison_df.head(10))