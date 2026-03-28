import pandas as pd
import numpy as np
import xgboost as xgb
import os
import shap
import warnings
import matplotlib.pyplot as plt
import urllib.request
import gzip
import io
import re

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

# --- SABİTLER ---
NASA_GENES = [
    'FOS', 'ATM', 'BRCA2', 'ERCC1', 'XRCC1', 'DDB2', 'LIG4', 'MSH6', 
    'BAG1', 'GSTM3', 'CYP1A1', 'CYP2F1', 'RAG1', 'RAG2', 'TCP1', 'TOP1'
]

# --- FONKSİYONLAR ---
def get_geo_mapping(gpl_id):
    if not gpl_id: return {}
    gpl_id = gpl_id.strip().replace('"', '')
    print(f"[*] {gpl_id} platform sözlüğü indiriliyor...")
    
    try:
        gpl_num = int(gpl_id.replace('GPL', ''))
        folder = "GPLnnn" if gpl_num < 1000 else f"GPL{str(gpl_num)[:-3]}nnn" 
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{folder}/{gpl_id}/annot/{gpl_id}.annot.gz"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response) as unzipped:
                file_content = unzipped.read().decode('utf-8')
                
        lines = file_content.split('\n')
        start_idx = next(i for i, line in enumerate(lines) if line.startswith('ID\t') or line.startswith('"ID"\t'))
                
        df_annot = pd.read_csv(io.StringIO('\n'.join(lines[start_idx:])), sep='\t', dtype=str, low_memory=False)
        df_annot.columns = [str(c).replace('"', '').strip() for c in df_annot.columns]
        
        id_col = [c for c in df_annot.columns if c.lower() in ['id', 'probe_id']][0]
        gene_col = [c for c in df_annot.columns if 'symbol' in c.lower()][0]
        
        df_annot = df_annot.dropna(subset=[gene_col])
        df_annot[gene_col] = df_annot[gene_col].apply(lambda x: str(x).split('///')[0].strip())
        
        print(f"  ✓ Haritalama hazır.")
        return dict(zip(df_annot[id_col], df_annot[gene_col]))
    except Exception as e:
        print(f"  [!] Sözlük okunamadı: {e}")
        return {}

def veri_yukle_ve_hazirla(filepath):
    print(f"[*] {filepath} veri seti işleniyor...")
    
    hasta_meta = {}
    gpl_id = None
    gsm_ids = []
    titles = []
    data_lines = []
    
    opener = gzip.open(filepath, "rt", encoding='utf-8') if filepath.endswith('.gz') else open(filepath, "r", encoding='utf-8')
    
    with opener as f:
        for line in f:
            if line.startswith("!Series_platform_id"):
                gpl_id = line.strip().split("\t")[1].strip('"')
            elif line.startswith("!Sample_geo_accession"):
                gsm_ids = [g.strip('"') for g in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [v.strip('"') for v in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [v.strip('"').lower() for v in line.strip().split("\t")[1:]]
                if ':' not in vals[0]: continue
                
                char_name = vals[0].split(':')[0].strip()
                for gsm, val in zip(gsm_ids, vals):
                    if gsm not in hasta_meta: hasta_meta[gsm] = {}
                    if ':' in val:
                        hasta_meta[gsm][char_name] = val.split(":", 1)[1].strip()

            elif line.startswith('"ID_REF"') or (not line.startswith("!") and len(gsm_ids) > 0):
                data_lines.append(line)

    # Hasta ID ve Ziyaret Çıkarma
    for gsm, title in zip(gsm_ids, titles):
        match = re.search(r'\(([^_]+)_([^\)]+)\)', title)
        if match:
            hasta_meta[gsm]['hasta_id'] = match.group(1)
            hasta_meta[gsm]['ziyaret'] = pd.to_numeric(match.group(2), errors='coerce')
        else:
            hasta_meta[gsm]['hasta_id'] = gsm
            hasta_meta[gsm]['ziyaret'] = 1
            
        if 'healthy' in title.lower() or 'control' in title.lower() or title.startswith('HC'):
            hasta_meta[gsm]['is_healthy'] = True
        else:
            hasta_meta[gsm]['is_healthy'] = False

    str_data = "".join(data_lines)
    df_genes = pd.read_csv(io.StringIO(str_data), sep="\t", index_col=0)
    df_genes.index = df_genes.index.astype(str).str.replace('"', '').str.strip()
    
    mapping = get_geo_mapping(gpl_id)
    if mapping:
        df_genes.index = df_genes.index.map(mapping)
        df_genes = df_genes[df_genes.index.notnull()]
        df_genes = df_genes.groupby(df_genes.index).mean()

    df_genes = df_genes.transpose()
    
    mevcut_nasa_genleri = [gen for gen in NASA_GENES if gen in df_genes.columns]
    df_genes = df_genes[mevcut_nasa_genleri]
    print(f"  ✓ {len(mevcut_nasa_genleri)}/{len(NASA_GENES)} NASA geni bulundu.")

    df_meta = pd.DataFrame.from_dict(hasta_meta, orient='index')
    final_df = df_meta.join(df_genes, how='inner')
    
    return final_df

def hedef_olustur(df):
    print("\n[*] Veri yapısı kuruluyor...")
    
    # 1. Sağlıklı Kontrolleri Çıkar
    df = df[df['is_healthy'] == False]
    
    # 2. CDAI Sütununu Bul
    cdai_col = next((col for col in df.columns if 'cdai' in col.lower()), None)
    if not cdai_col:
        print("[!] HATA: CDAI skoru bulunamadı.")
        return pd.DataFrame()

    df['cdai_numeric'] = pd.to_numeric(df[cdai_col], errors='coerce')
    df = df.dropna(subset=['cdai_numeric', 'hasta_id', 'ziyaret'])
    
    # 3. TOPLAM ZİYARET SAYISINI HESAPLA (Hastanın veri setinde toplam kaç kaydı var?)
    df['toplam_ziyaret'] = df.groupby('hasta_id')['hasta_id'].transform('count')
    
    # 4. HEDEF: CDAI > 10 ise 1 (Yanıt yok / Aktif hastalık), değilse 0
    df['hedef'] = (df['cdai_numeric'] > 10.0).astype(int)
    
    print(f"  ✓ {len(df)} RA hastası ölçümü başarıyla oluşturuldu.")
    return df

def alt_grup_degerlendir(y_true, y_pred, y_prob, grup_adi):
    """Belirli bir alt grup için metrikleri hesaplar ve yazdırır."""
    if len(y_true) == 0:
        return
        
    print(f"\n{'='*60}")
    print(f" GRUP: {grup_adi} (Test Setindeki Örnek Sayısı: {len(y_true)})")
    print(f"{'='*60}")
    
    try:
        roc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC:   {roc:.2f}")
    except:
        print("ROC-AUC:   Hesaplanamadı (Gruptaki test hastalarının hepsi aynı hedef sınıfında)")
        
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2f}")
    print("\nSınıflandırma Raporu:\n", classification_report(y_true, y_pred, zero_division=0))

def main():
    dosya_yolu = "/home/doksaniki/romato/GEO dataset/GSE93272_series_matrix.txt" 
    
    if not os.path.exists(dosya_yolu):
        print(f"[!] HATA: Dosya bulunamadı: {dosya_yolu}")
        return

    df_ham = veri_yukle_ve_hazirla(dosya_yolu)
    if df_ham.empty: return

    df = hedef_olustur(df_ham)
    if df.empty: return

    # Orijinal veriyi alt grup değerlendirmesi için sakla
    X_raw = df.copy()

    # --- VERİ SIZINTISINI ÖNLEME ---
    # Model hastanın gelecekte kaç kez kliniğe geleceğini bilmemeli! 'toplam_ziyaret' ve 'ziyaret' siliniyor.
    drop_cols = ['hedef', 'response', 'efficacy', 'cdai', 'cdai_numeric', 'is_healthy', 'toplam_ziyaret', 'ziyaret'] 
    X_pre = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Tüm string/kategorik verileri One-Hot yap ve sayısala çevir
    X_encoded = pd.get_dummies(X_pre.drop(columns=['hasta_id']), drop_first=True)
    X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
    X_encoded = X_encoded.dropna(axis=1, how='all').fillna(X_encoded.median())
    
    y = df['hedef']
    hasta_gruplari = df['hasta_id'] # Aynı hasta eğitim ve teste aynı anda girmesin diye grupluyoruz

    print(f"\n--- Modelleme Aşaması ---")
    print(f"Toplam Örnek: {X_encoded.shape[0]}, Özellik (Feature): {X_encoded.shape[1]}")
    
    # GroupShuffleSplit ile Hastaya Göre Bölme
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_encoded, y, groups=hasta_gruplari))
    
    X_train = X_encoded.iloc[train_idx]
    X_test = X_encoded.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    # XGBoost Modeli
    print("\n[*] XGBoost modeli eğitiliyor...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42, 
        eval_metric="logloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # ---------------------------------------------------------
    # 1. GENEL DEĞERLENDİRME TABLOSU
    # ---------------------------------------------------------
    alt_grup_degerlendir(y_test, y_pred, y_prob, "TÜM HASTALAR (GENEL PERFORMANS)")
    
    # ---------------------------------------------------------
    # 2. CİNSİYETE GÖRE DEĞERLENDİRME
    # ---------------------------------------------------------
    gender_col = next((col for col in X_raw.columns if 'gender' in col.lower() or 'sex' in col.lower()), None)
    if gender_col:
        test_cinsiyetler = X_raw[gender_col].iloc[test_idx].astype(str).str.lower().str.strip()
        
        mask_kadin = test_cinsiyetler.isin(['female', 'f', 'kadın', 'kadin'])
        mask_erkek = test_cinsiyetler.isin(['male', 'm', 'erkek'])
        
        alt_grup_degerlendir(y_test[mask_kadin], y_pred[mask_kadin], y_prob[mask_kadin], "KADIN HASTALAR")
        alt_grup_degerlendir(y_test[mask_erkek], y_pred[mask_erkek], y_prob[mask_erkek], "ERKEK HASTALAR")

    # ---------------------------------------------------------
    # 3. HASTANIN TOPLAM ZİYARET SAYISINA GÖRE DEĞERLENDİRME
    # ---------------------------------------------------------
    if 'toplam_ziyaret' in X_raw.columns:
        test_toplam_ziyaretler = X_raw['toplam_ziyaret'].iloc[test_idx]
        ziyaret_tipleri = sorted(test_toplam_ziyaretler.unique())
        
        for v in ziyaret_tipleri:
            mask_v = (test_toplam_ziyaretler == v)
            alt_grup_degerlendir(y_test[mask_v], y_pred[mask_v], y_prob[mask_v], f"KLİNİĞİ TOPLAM {int(v)} KEZ ZİYARET EDEN HASTALAR")

    # --- SHAP ANALİZİ ---
    print("\n[*] SHAP Grafiği Çiziliyor...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("COSMO-AI: NASA Genlerinin ve Kliniğin Genel Etkisi")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"SHAP hatası: {e}")

if __name__ == "__main__":
    main()
