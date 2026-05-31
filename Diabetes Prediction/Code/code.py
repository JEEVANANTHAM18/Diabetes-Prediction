import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import seaborn as sns



# Load dataset
data = pd.read_csv("D:/ML/Diabetes Prediction/diabetes_prediction_dataset.csv")

# Graph 1: Blood Glucose vs Diabetes
plt.scatter(data["blood_glucose_level"], data["diabetes"])
plt.xlabel("Blood Glucose Level")
plt.ylabel("Diabetes")
plt.title("Blood Glucose vs Diabetes")
plt.show()

# Graph 2: Sankey Diagram (Age Group → Diabetes)

# Create age groups
data["age_group"] = pd.cut(data["age"],
                           bins=[0,30,50,100],
                           labels=["Young","Middle Age","Old"])

# Count flows
young_no = len(data[(data["age_group"]=="Young") & (data["diabetes"]==0)])
young_yes = len(data[(data["age_group"]=="Young") & (data["diabetes"]==1)])

mid_no = len(data[(data["age_group"]=="Middle Age") & (data["diabetes"]==0)])
mid_yes = len(data[(data["age_group"]=="Middle Age") & (data["diabetes"]==1)])

old_no = len(data[(data["age_group"]=="Old") & (data["diabetes"]==0)])
old_yes = len(data[(data["age_group"]=="Old") & (data["diabetes"]==1)])

# Sankey Diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        label=["Youngsters","Middle Age","Old Age","No Diabetes","Diabetes"]
    ),
    link=dict(
        source=[0,0,1,1,2,2],
        target=[3,4,3,4,3,4],
        value=[young_no,young_yes,mid_no,mid_yes,old_no,old_yes],

        # Colors for connecting lines
        color=[
            "lightgrey",   # Young → No Diabetes
            "blue",        # Young → Diabetes
            "lightgrey",   # Middle → No Diabetes
            "red",         # Middle → Diabetes
            "lightgrey",   # Old → No Diabetes
            "green"        # Old → Diabetes
        ]
    )
)])

fig.update_layout(title_text="Age Group → Diabetes Relationship", font_size=10)
fig.show()

# Graph 3: Radar Chart (Spider Chart)

diabetes_mean = data[data["diabetes"] == 1][["blood_glucose_level","bmi","age"]].mean()
no_diabetes_mean = data[data["diabetes"] == 0][["blood_glucose_level","bmi","age"]].mean()

categories = ["Blood Glucose","BMI","Age"]

fig = go.Figure()

# No Diabetes
fig.add_trace(go.Scatterpolar(
    r=[no_diabetes_mean["blood_glucose_level"],
       no_diabetes_mean["bmi"],
       no_diabetes_mean["age"]],
    theta=categories,
    fill='toself',
    name='No Diabetes',
    line=dict(color='blue', width=3),
    fillcolor='rgba(0,0,255,0.25)',
    marker=dict(size=8, color='blue')
))

# Diabetes
fig.add_trace(go.Scatterpolar(
    r=[diabetes_mean["blood_glucose_level"],
       diabetes_mean["bmi"],
       diabetes_mean["age"]],
    theta=categories,
    fill='toself',
    name='Diabetes',
    line=dict(color='red', width=3),
    fillcolor='rgba(255,0,0,0.25)',
    marker=dict(size=8, color='red')
))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title="Health Feature Comparison (Radar Chart)",
    showlegend=True
)

fig.show()
# Graph 5: Violin Plot (Blood Glucose vs Diabetes)

plt.figure(figsize=(8,5))

sns.violinplot(x="diabetes", y="blood_glucose_level", data=data)

plt.xlabel("Diabetes (0 = No, 1 = Yes)")
plt.ylabel("Blood Glucose Level")
plt.title("Distribution of Blood Glucose Level by Diabetes")

plt.show()
# Graph 4: Diabetes Distribution
plt.hist(data["diabetes"], bins=2)
plt.xlabel("Diabetes")
plt.ylabel("Number of Patients")
plt.title("Diabetes Distribution")
plt.show()
# Graph: Bubble Chart (BMI vs Blood Glucose vs Age)

plt.figure(figsize=(8,6))

plt.scatter(
    data["bmi"],                 # X-axis
    data["blood_glucose_level"], # Y-axis
    s=data["age"]*2,             # Bubble size based on age
    c=data["diabetes"],          # Color based on diabetes
    cmap="coolwarm",
    alpha=0.6
)

plt.xlabel("BMI")
plt.ylabel("Blood Glucose Level")
plt.title("Bubble Chart: BMI vs Blood Glucose vs Age")

plt.colorbar(label="Diabetes (0 = No, 1 = Yes)")

plt.show()

# Select features and target
X = data[["blood_glucose_level", "bmi", "age"]]
y = data["diabetes"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------
# Logistic Regression Model
# ---------------------------

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)


# ---------------------------
# Decision Tree Model
# ---------------------------

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_accuracy)

sample = pd.DataFrame([[120, 30, 45]],
columns=["blood_glucose_level","bmi","age"])

sample = scaler.transform(sample)

prediction = lr_model.predict(sample)

if prediction[0] == 1:
    print("Prediction Result: Patient likely has Diabetes")
else:
    print("Prediction Result: Patient likely does NOT have Diabetes")