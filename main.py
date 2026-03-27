import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import shap
import warnings

# Uyarıları bastıralım
warnings.filterwarnings("ignore")

def veri_yukle_ve_hazirla(filepath):
    """
    GEO Series Matrix dosyalarını okuyup,
    Gen ekspresyonu ve Klinik verileri birleştiren fonksiyon.
    """
    print(f"Veri yükleniyor: {filepath}")
    
    # 1. Meta verileri ve veriyi ayırma
    # GEO dosyaları '!' ile başlayan satırlarda metadata taşır.
    metadata = {}
    data_lines = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("!"):
                    # Meta veriyi kaydet (örn: !Sample_title = "Hasta 1")
                    if "=" in line:
                        key, value = line.split("=", 1)
                        metadata[key.strip()] = value.strip()
                elif line.startswith('"') or line.startswith('ID'):
                    # Veri satırlarını sakla
                    data_lines.append(line)
    except FileNotFoundError:
        print("HATA: Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return pd.DataFrame()

    # 2. Veriyi DataFrame'e çevirme
    # StringIO kullanarak string'i dosya gibi okutuyoruz
    from io import StringIO
    str_data = "".join(data_lines)
    
    # GEO matrisleri genelde tab ile ayrılmıştır
    df = pd.read_csv(StringIO(str_data), sep="\t", index_col=0)
    
    # Temizlik: Sütunlardaki çift tırnakları temizleme (varsa)
    df.columns = [col.replace('"', '') for col in df.columns]

    # --- DİKKAT: BURASI VERİ SETİNE ÖZELDİR ---
    # GSE93272 gen ekspresyon verisidir. 'cdai', 'age' gibi klinik veriler 
    # genellikle sütun isimlerinde veya ayrı bir metadata satırında gizlidir.
    # Eğer bu dosyada klinik veriler sütunlarda yoksa, bu kod sadece genleri çeker.
    
    # Örnek: Eğer klinik veriler ayrı bir dataframe'deyse burada birleştirme (merge) yapılmalıdır.
    # Şimdilik yüklenen tabloyu döndürüyoruz.
    print(f"Yüklenen tablo boyutu (Gen x Örnek): {df.shape}")
    
    # Transpoze alıyoruz çünkü Genellikle satırlar genlerdir, XGBoost için sütunlar gen olmalıdır.
    # Yani: Satırlar = Hastalar (Örnekler), Sütunlar = Genler
    df = df.transpose()
    
    return df

def main():
    filepath = "GSE93272_series_matrix.txt" 
    ra = veri_yukle_ve_hazirla(filepath)
    
    if ra.empty:
        print("Veri yüklenemediği için işlem durduruldu.")
        return

    print(f"✓ İşlenmiş Tablo boyutu: {ra.shape}")

    # ---------------------------------------------------------
    # KRİTİK ADIM: HEDEF DEĞİŞKEN (TARGET) OLUŞTURMA
    # GEO verisinde hedef genelde sütun isimlerinde veya index'te gizlidir.
    # Örnek: Eğer index isimleri "Control_1", "RA_1" gibi ise:
    # ---------------------------------------------------------
    
    # Basit bir örnek target oluşturma mantığı (Kendi verinize uyarlayın):
    # Eğer index isminde "Control" geçiyorsa 0, diğerleri 1 olsun
    ra['hedef'] = ra.index.str.contains('Control|Healthy', case=False).astype(int)
    
    # Eğer elinizde 'cdai' gibi klinik veriler varsa bunlar yukarıdaki transpoze işleminde sütun olmalı.
    # Eğer yoksa model sadece gen ekspresyonlarına göre çalışır.
    
    sayisal_kolonlar = ["age", "cdai", "haq", "d.vas", "pain.vas"] 
    mevcut_sayisal = [col for col in sayisal_kolonlar if col in ra.columns]
    
    if mevcut_sayisal:
        print(f"Sayısal klinik veriler bulundu: {mevcut_sayisal}")
        ra[mevcut_sayisal] = ra[mevcut_sayisal].apply(pd.to_numeric, errors="coerce")

    # Cinsiyet dönüşümü
    if "gender" in ra.columns:
        ra["gender_num"] = ra["gender"].map({"M": 1, "F": 0, "Male": 1, "Female": 0})
        ra = ra.drop("gender", axis=1)

    # Özellikler (X) ve Hedef (y)
    hedef_kolon = "hedef" 
    
    if hedef_kolon not in ra.columns:
        print("HATA: Hedef kolon bulunamadı. Lütfen yukarıdaki 'hedef oluşturma' kısmını verinize göre düzenleyin.")
        return

    y = ra[hedef_kolon]
    X = ra.drop([hedef_kolon], axis=1)

    # Object tipindeki sütunları (varsa) temizleyelim veya kategori yapalım
    # XGBoost 1.5+ versiyonu kategorik veriyi destekler ama sayısal daha güvenlidir
    X = X.select_dtypes(include=[np.number]) 

    print(f"Model girdi boyutu (X): {X.shape}")
    print(f"Hedef dağılımı:\n{y.value_counts()}")

    # Veriyi Bölme
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model ve Hiperparametre
    model = xgb.XGBClassifier(
        random_state=42, 
        eval_metric="logloss", 
        tree_method="hist",
        use_label_encoder=False
    )

    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }

    print("\nHiperparametre araması başlıyor...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=10, # Veri büyükse sayıyı düşük tutmak hızlandırır
        scoring='roc_auc',
        cv=3, # Küçük veride CV sayısını 3 yapmak güvenlidir
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    print(f"\nEn İyi Parametreler: {random_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Skoru: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # SHAP Analizi
    print("\nSHAP grafiği oluşturuluyor...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    
    # Grafiği göstermek için matplotlib backend kullanılabilir
    import matplotlib.pyplot as plt
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()