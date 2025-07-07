# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 22:31:30 2025

@author: mustafa
"""

#import libraries
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#load dataset breast cancer
cancer=load_breast_cancer()
X=cancer.data
y=cancer.target

#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#define extra trees
et_model=ExtraTreesClassifier(
    n_estimators=100, #ağaç sayısı
    max_depth=10, #max derinlik
    min_samples_split=5, #bir düğümü bölmek için min önrek sayısı
    random_state=42
    
    
    )

#training
et_model.fit(X_train,y_train)

#testing
y_pred=et_model.predict(X_test)

#evaluation: accuracy,report,confusion matrix
print(f"Accuracy{accuracy_score(y_pred,y_test)}") #random forest'a göre %2 daha başarılı
print(classification_report(y_pred,y_test))


#visualize feature imoortance
feature_importance=et_model.feature_importances_  #değeri büyük olan daha önemli
sorted_index=np.argsort(feature_importance) [::-1] 
features=cancer.feature_namesss

plt.figure()
plt.bar(range(X.shape[1]),feature_importance[sorted_index],align="center")
plt.xticks(range(X.shape[1]),features[sorted_index],rotation=90)
plt.title("Features Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()







