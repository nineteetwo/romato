import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import shap

def veri_yukle_ve_hazirla(filepath):
    """
    GEO veya benzeri text/csv dosyalarından veriyi yükleyip 
    temel hazırlıkları yapar.
    """
    # Not: Buraya kendi parser kodunu veya pd.read_csv kodunu ekleyebilirsin.
    # Örnek olarak elimizde 'ra' isimli bir DataFrame olduğunu varsayıyoruz.
    print(f"Veri yükleniyor: {filepath}")
    
    # Simüle edilmiş boş DataFrame (Kendi verinle değiştirmelisin)
    ra = pd.DataFrame() 
    return ra

def main():
    # 1. Veriyi Yükle
    filepath = "GSE93272_series_matrix.txt"
    ra = veri_yukle_ve_hazirla(filepath)
    
    # GERÇEK VERİ VARSA AŞAĞIDAKİ İŞLEMLER ÇALIŞACAKTIR
    if not ra.empty:
        print(f"✓ Tablo boyutu: {ra.shape}")
        
        # 2. Vektörize Veri Tipi Dönüşümü (Döngü yerine tek satır - Çok daha hızlı)
        sayisal = ["age", "cdai", "haq", "d.vas", "pain.vas", "kowabari", "tjc66.68"]
        mevcut_sayisal = [col for col in sayisal if col in ra.columns]
        if mevcut_sayisal:
            ra[mevcut_sayisal] = ra[mevcut_sayisal].apply(pd.to_numeric, errors="coerce")

        # 3. Cinsiyeti modele özellik (feature) olarak ekleme (Az veriyi bölmemek için)
        if "gender" in ra.columns:
            ra["gender_num"] = ra["gender"].map({"M": 1, "F": 0})
            ra = ra.drop("gender", axis=1)

        # 4. Hedef (y) ve Özellikleri (X) ayırma
        hedef_kolon = "hedef" # Kendi hedef kolonunun adını yaz
        if hedef_kolon in ra.columns:
            y = ra[hedef_kolon]
            X = ra.drop([hedef_kolon], axis=1) # ID gibi gereksiz kolonları da buradan düşür

            # DİKKAT: Eksik verileri (NA) doldurmak için SimpleImputer KULLANMIYORUZ.
            # XGBoost eksik verilerle algoritmik olarak kendisi başa çıkabilir.

            # 5. Veriyi Bölme
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. XGBoost ve RandomizedSearchCV (Hızlı Hiperparametre Optimizasyonu)
            # tree_method="hist" CPU üzerinde eğitimi inanılmaz hızlandırır
            model = xgb.XGBClassifier(
                random_state=42, 
                eval_metric="logloss", 
                tree_method="hist",
                enable_categorical=True # Kategorik veri kalırsa hata vermemesi için
            )

            param_distributions = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0]
            }

            print("Hiperparametre araması başlıyor (RandomizedSearchCV)...")
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=15, # 15 farklı kombinasyon deneyecek
                scoring='roc_auc',
                cv=5,
                verbose=1,
                random_state=42,
                n_jobs=-1 # Bilgisayarındaki tüm çekirdekleri kullanır
            )

            random_search.fit(X_train, y_train)

            # 7. Sonuçlar ve Değerlendirme
            best_model = random_search.best_estimator_
            print(f"\nEn İyi Parametreler: {random_search.best_params_}")
            
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            print("\n--- Sınıflandırma Raporu ---")
            print(classification_report(y_test, y_pred))
            print(f"ROC AUC Skoru: {roc_auc_score(y_test, y_pred_proba):.4f}")

            # 8. SHAP ile Açıklanabilirlik (Modeli nelerin etkilediğini görme)
            print("\nSHAP grafiği oluşturuluyor...")
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test)

if __name__ == "__main__":
    main()