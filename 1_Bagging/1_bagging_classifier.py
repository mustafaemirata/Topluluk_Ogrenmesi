from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data # Özellikler
y = iris.target # Hedef değişken

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Temel model: Karar Ağacı
base_model = DecisionTreeClassifier(random_state=42)

# Bagging modeli oluşturma
# 'base_estimator' yerine 'estimator' kullanıldı ve 'n_estimator' yerine 'n_estimators' kullanıldı
bagging_model = BaggingClassifier(
    estimator=base_model,        # Temel model (DecisionTreeClassifier)
    n_estimators=7,             # Kullanılacak temel model sayısı
    max_samples=0.9,             # Her modelin kullanacağı örnek oranı
    max_features=0.9,            # Her modelin kullanacağı özellik oranı
    bootstrap=False,              # Örneklerin tekrar seçilmesine izin verdik
    random_state=42
)

#model training

bagging_model.fit(X_train,y_train)

#model testing
y_pred= bagging_model.predict(X_test)

#model accuracy
accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy:{accuracy}")