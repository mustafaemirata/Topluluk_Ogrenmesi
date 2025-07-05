# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 21:59:01 2025

@author: mustafa
"""
#iport libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

#load dataset: breat cancer veri seti
cancer=load_breast_cancer()
X= cancer.data #features, tumor boyut, sekil, alan
y=cancer.target #hedef değişken, 0= kotu (malignant),1=iyi (benign)

#veriyi train ve test verisi olarak ayır
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#create random forest model
rf_model=RandomForestClassifier(
    n_estimators=100, #agac sayisi
    max_depth=10, #max derinlik
    min_samples_split=5, #bir düğümü bölmek için min örnek sayısı
    bootstrap=False,
    random_state=42
    
    
    )


#training
rf_model.fit(X_train,y_train)

#testing
y_pred=rf_model.predict(X_test)

#evaluation:accuracy 
accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy:{accuracy}")

print(classification_report(y_test, y_pred))









