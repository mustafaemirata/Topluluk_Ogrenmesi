#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 7 21:58:45 2025

@author: emir
"""

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor # BaggingRegressor için base_estimator olarak kullanılır
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Bagging Regressor": BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42), # base_estimator yerine estimator kullanıldı ve random_state eklendi
        n_estimators=100, # ağaç sayısı
        max_features=0.8, # Her bir temel eğiticinin kullanacağı özelliklerin oranı
        max_samples=0.8,  # Her bir temel eğiticinin kullanacağı örneklerin oranı
        random_state=42   # Randomlik için sabitleme
    ),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    ),
    "Extra Trees Regressor": ExtraTreesRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
}
# training & testing
results={}
predictions={}

for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse=mean_squared_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)
    results[name]={"MSE":mse,"R2":r2}
    predictions[name]=y_pred

result_df=pd.DataFrame(results).T
print(result_df.head())    
#r2 ne kadar yüksekse o kadar iyi.
#mse ne kadar düşükse o kadar iyi
#hata en yüksek extra treelerde çıkmış  

#visualize

#tahmin vs gerçek değerler
plt.Figure()
for i,(name,y_pred) in enumerate(predictions.items(),1): 
    plt.subplot(1,3,i)  
    plt.scatter(y_test,y_pred,alpha=0.5,label=name)
    plt.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()], "r--",lw=2) 
    plt.title(f"{name} Gerçek vs Tahmin")
    plt.xlabel("Gerçek")
    plt.ylabel("Tahmin")
    plt.legend()
plt.tight_layout()
plt.show()


#hatalar
plt.figure()
for i,(name,y_pred) in enumerate(predictions.items(),1): 


    plt.subplot(1,3,i)  
    plt.scatter(y_pred,residuals,alpha=0.5,label=name)
    plt.axhline(y=0,color="r",linestyle="--")
    plt.title(f"{name} Gerçek vs Tahmin")
    plt.xlabel("Gerçek")
    plt.ylabel("Tahmin")
    plt.legend()
plt.tight_layout()
plt.show()

# feature importance - EN onemlisini kesfetme.BU model için Medinc en önemlisi.

feature_names=housing.feature_names
plt.figure()
for i,(name,y_pred) in enumerate(models.items(),1):
    if hasattr(model, "feature_importances_"):
        importance=model.feature_importances_
        sorted_idx=np.argsort(importance)[::-1]
        plt.subplot(1,3,i)
        plt.bar(range(X.shape[1]),importance[sorted_idx],label=name)
        plt.xticks(range(X.shape[1]),np.array(feature_names)[sorted_idx],rotation=45)
        plt.title(f"{name} Feature Importance")
        plt.xlabel("Features")
        plt.ylabel("Importance Featuures")
        plt.legend()
plt.tight_layout()
plt.show()























