import pandas as pd
import numpy as np
url="https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
df=pd.read_csv(url)
print(df.head())

#Checking for imbalance in the dataset
print(df['Class'].value_counts())

#Balancing the dataset
from imblearn.over_sampling import SMOTE
x=df.drop(columns=['Class'])
y=df['Class']
smote=SMOTE(random_state=42)
x_resampled,y_resampled=smote.fit_resample(x,y)
df_bal=pd.DataFrame(x_resampled,columns=x.columns)
df_bal['Class']=y_resampled

#Balanced dataset
print(df_bal['Class'].value_counts())

#Sample size detection
import math
import scipy.stats as stats

#Assuming the parameters
Z=1.96  #95% confidence level
p=0.5   #Estimated proportion
E=0.05  #Margin of error
S=2
C=5
N=len(df_bal) #Population size

def calc_sample_size(Z,p,E):
    return math.ceil(((Z**2)*p*(1-p))/(E**2))

n_simple=calc_sample_size(Z,p,E)

E_stratified=E/S
n_stratified=calc_sample_size(Z,p,E_stratified)

E_cluster=E/C
n_cluster=calc_sample_size(Z,p,E_cluster)

n_systematic=int(0.1*N)
k=N//n_systematic  #Sampling interval

n_bootstrap=N

print(f"Sample Size(Simple Random Sampling): {n_simple}")
print(f"Sample Size(Stratified Sampling, {S} strata): {n_stratified}")
print(f"Sample Size(Cluster Sampling, {C} clusters): {n_cluster}")
print("Systematic Sample Size:", n_systematic)
print("Bootstrap Sample Size:", n_bootstrap)

from sklearn.model_selection import train_test_split

#Simple Random Sampling
simple_random_sample=df_bal.sample(n=n_simple,random_state=42)

#Stratified Sampling
stratified_sample, _=train_test_split(df_bal,train_size=math.ceil(n_stratified/2),stratify=df_bal["Class"],random_state=42)

#Systematic Sampling
start=np.random.randint(0,k)
systematic_sample=df_bal.iloc[start::k]

#Cluster Sampling
df_bal['Cluster']=np.random.randint(0,C,size=N)
cluster_sample=df_bal[df_bal['Cluster']==np.random.randint(0, C)].sample(n=math.ceil(n_cluster/3),random_state=42,replace=True)

#Bootstrap Sampling
bootstrap_sample=df_bal.sample(n=N,replace=True)

print("Simple Random:",len(simple_random_sample))
print("Stratified:",len(stratified_sample))
print("Systematic:",len(systematic_sample))
print("Cluster:",len(cluster_sample))
print("Bootstrap:",len(bootstrap_sample))

#Applying models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

models={
    "Logistic Regression":LogisticRegression(),
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "XGBoost":XGBClassifier(use_label_encoder=False,eval_metric='logloss'),
    "SVM":SVC()
}

def train_and_evaluate(sample,name):
    X_t=sample.drop(columns=["Class", "Cluster"],errors="ignore")
    y_t=sample["Class"]
    X_train,X_test,y_train,y_test=train_test_split(X_t,y_t,test_size=0.2,random_state=42)

    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    results={}
    for model_name,model in models.items():
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        acc=accuracy_score(y_test,y_pred)
        results[model_name]=acc
    print(f"\nAccuracy Scores for {name} Sampling:")
    for model,acc in results.items():
        print(f"{model}:{acc:.4f}")
    return results

res={}
res["Simple Random"]=train_and_evaluate(simple_random_sample,"Simple Random")
res["Stratified"]=train_and_evaluate(stratified_sample,"Stratified")
res["Systematic"]=train_and_evaluate(systematic_sample,"Systematic")
res["Cluster"]=train_and_evaluate(cluster_sample,"Cluster")
res["Bootstrap"]=train_and_evaluate(bootstrap_sample,"Bootstrap")

#Comparing results
from tabulate import tabulate
df_results=pd.DataFrame(res).T
print(tabulate(df_results,headers="keys"))

#Best sampling techniques per model
best_sampling_per_model={}
for model in models.keys():
    best_sampling=max(res,key=lambda s:res[s][model])
    best_sampling_per_model[model]=best_sampling

print("\nBest Sampling Technique for Each Model:")
for model,sampling in best_sampling_per_model.items():
    print(f"{model}:{sampling}")
