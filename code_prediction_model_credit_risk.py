# LOAN DATA 2007 - 2014 ID/X Partners
## IMPORT DATASET
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df_loan = pd.read_csv('loan_data_2007_2014.csv')
pd.set_option('display.max_columns', None)
df_loan.head(3)
## INFORMASI AWAL
### Informasi data awal
df_loan.info()
#### Temuan
Hasil temuan pada .info di atas, banyak kolom yang sama sekali tidak memiliki nilai (0 non-null). <br>
Saya memutuskan untuk menghapusnya karena memang tidak akan terpakai. kemudian saya menghapus kolom Unnamed: 0 karena itu merupakan kolom nomor tanpa judul dari file CSV
#### Tindakan
# daftar kolom yang akan di hapus
kolom_dihapus = ['Unnamed: 0', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m']

# Hapus Kolom pada variabel kolom_dihapus
df_loan.drop(columns=kolom_dihapus, inplace=True)
df_loan.info()
### Informasi Missing Values
# Hitung jumlah missing values untuk setiap kolom
jumlah_missing = df_loan.isnull().sum()

# Hitung persentase missing values untuk setiap kolom
persentase_missing = round((jumlah_missing / len(df_loan)) * 100, 2)

# Gabungkan hasil dalam satu DataFrame
missing_summary = pd.DataFrame({
    'Jumlah Missing Values': jumlah_missing,
    'Persentase Missing Values (%)': persentase_missing
})

# Filter hanya kolom yang memiliki missing values
missing_summary_filtered = missing_summary[missing_summary['Jumlah Missing Values'] > 0]

print(missing_summary_filtered)
#### Temuan
Dari 57 Kolom, 22 Kolom memiliki Missing Values (Tidak memiliki Nilai). Bahkan ada 3 kolom yang memiliki Missing Values lebih dari 70%. <br>
Penanganan akan dilakukan nanti pada tahap data preprocessing.
### Deskripsi Singkat
Jumlah kolom yang ada pada dataset adalah 75 kolom dan memiliki 466285 baris data. Saya menemukan bahwa kolom target adalah kolom loan_status (status pinjaman) atau bisa jadi jika jumlah unik valuesnya lebih dari 2 maka akan ditambahkan kolom baru yang menjadi target yang dipisakan antara yang diberi pinjaman (1) dan tidak diberi pinjaman (0) dari kolom loan_status. Dari hasil analis informasi awal, ternyata sejumlah 17 kolom tidak memiliki nilai sama sekali. Maka sudah bisa dipastikan kolom tersebut tidak akan terpakai, saya melakukan tindakan secara langsung dengan menghapus kolom-kolom tersebut. Melihat dari Missing Values juga ternyata ada 3 kolom yang memiliki persentase kehilangan data sebanyak 70%. Hal ini tentunya ada kemungkinan kolom-kolom tersebut akan dihapus.
## STATISTIK DESKRIPTIF
### Memisahkan Kolom Kategorikal dan Numerical
# Memisahkan kolom numerikal dan kategorikal
kolom_numerikal = df_loan.select_dtypes(include=['number']).columns
kolom_kategorikal = df_loan.select_dtypes(include=['object']).columns

print("Kolom Numerikal:", kolom_numerikal)
print("Kolom Kategorikal:", kolom_kategorikal)
### Statistika Deskriptif
pd.set_option('display.float_format', '{:.2f}'.format)
#### Kolom Numerikal
df_loan[kolom_numerikal].describe().transpose()
#### Kolom Kategorikal
df_loan[kolom_kategorikal].describe().transpose()
#### Nilai kolom Kategorikal
for col in kolom_kategorikal:
    print(f'''Value count kolom {col}:''')
    print(df_loan[col].value_counts())
    print()

## EXPLORATORY DATA ANALISYS (EDA)
### Box Plot
plt.figure(figsize=(10, 10))

for i in range(0, len(kolom_numerikal)):
    plt.subplot(5, 7, i+1)
    sns.boxplot(y=df_loan[kolom_numerikal[i]], color='blue', orient='v')
    plt.tight_layout()
#### Temuan
Banyak sekali Outliers dari tampilan Boxplot di atas. Hanya beberapa kolom saja yang tidak memiliki outliers yang kemungkinan data tersebut berdistribusi normal. policy_code merupakan jumlah nomor emergency polisi, maka hanya memiliki jumlah 1 untuk setiap id.
### Distribution Plot
plt.figure(figsize=(12, 10))
for i in range(0, len(kolom_numerikal)):
    plt.subplot(7, 5, i+1)
    sns.distplot(df_loan[kolom_numerikal[i]], color='gray')
    plt.tight_layout()
#### Temuan
Beberapa kolom tidak terlihat distribusinya karena data hanya berkumpul di satu nilai dan sisanya menyebar. Hal ini terjadi karena Outliers yang terlalu banyak pada kolom tersebut. Ada potensi kolom kolom tersebut akan dihapus.
### Heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(df_loan[kolom_numerikal].corr(), cmap='viridis', annot=True, fmt='.2f')
#### Temuan
Banyak Feature yang redundan, seperti id dengan member_id, loan_amount dengan funded_amount, dan lain sebagainya. Perbandingan hubungan positif dengan negatif, lebih banyak feature yang memiliki hubungan positif, salah satu feature yang memiliki hubungan positif dengan nilai yang tinggi adalah total_payment dengan loan_amount yaitu 74. Sedangkan salah satu hubungan negatif yang paling tinggi adalah delinq_2yrs dengan mths_since_last_delinq yaitu -57.
## HAPUS KOLOM DAN LANJUTAN EDA
### Hapus Kolom
# daftar kolom yang akan di hapus
kolom_dihapus2 = ['id', 'member_id', 'desc', 'mths_since_last_delinq', 'mths_since_last_record', 'next_pymnt_d', 'mths_since_last_major_derog', 'url', 'emp_title', 'policy_code', 'application_type', 'pymnt_plan', 'funded_amnt', 'funded_amnt_inv', 'installment', 'out_prncp_inv', 'total_pymnt_inv', 'total_rec_prncp']

# Hapus Kolom pada variabel kolom_dihapus
df_loan.drop(columns=kolom_dihapus2, inplace=True)
#### Alasan Menghapus Kolom Tersebut
Alasan kolom tersebut di hapus adalah sebagai berikut : <br>
* Kolom kategorikal yang memiliki jumlah unik value yang terlalu banyak
* kolom dengan presentase nilai null lebih dari 40%
* Kolom yang redundan 
* kolom yang memiliki 1 nilai
* kolom yang hampir memiliki 1 nilai

Untuk informasi lebih detail mengenai kolom - kolom yang dihapus dapat diakses [disini](https://docs.google.com/document/d/1YxvRDpsjv0MpIfZA9DiXLX4mqpKdVMIyo-aFocB8zho/edit?usp=sharing)
df_loan.info()
### Analisis Kategorikal
kolom_kategorikal = df_loan.select_dtypes(include=['object']).columns

# Membuat bar chart untuk setiap kolom kategorikal
for i, col in enumerate(kolom_kategorikal):
    plt.figure(figsize=(10, 6))
    df_loan[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.xlabel("")
    plt.ylabel("Frequency")
    plt.show()
#### Temuan
Beberapa kolom terlalu banyak nilai Unik dan juga adanya ketidak sesuaian dalam tipe yang seharusnya adalah numerical. <br>
kolom title mengalami error. 
#### Tindakan
Tindakan
Kolom term akan diambil angkanya saja yaitu 36 dan 60 <br>
kolom emp_length diambil angkanya saja <br>
kolom issue_d akan di pihan antara angka tanggal dan bulan <br>
kolom zip_code akan di ambil angka depannya saja <br>
kolom earlist akan di pisah antara angka dengan bulan <br>
kolom last_payment_id di pisah antara angka dengan bulan <br>
kolom last_credit_pull di pisah antara angka dengan bulan
## HANDLING MISSING VALUES
# Hitung jumlah missing values untuk setiap kolom
jumlah_missing = df_loan.isnull().sum()

# Hitung persentase missing values untuk setiap kolom
persentase_missing = round((jumlah_missing / len(df_loan)) * 100, 2)

# Gabungkan hasil dalam satu DataFrame
missing_summary = pd.DataFrame({
    'Jumlah Missing Values': jumlah_missing,
    'Persentase Missing Values (%)': persentase_missing
})

# Filter hanya kolom yang memiliki missing values
missing_summary_filtered = missing_summary[missing_summary['Jumlah Missing Values'] > 0]

print(missing_summary_filtered)
# Menghapus baris yang memiliki nilai Null
df_loan.dropna(subset=['emp_length', 'annual_inc', 'title', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_util', 'total_acc', 'last_pymnt_d', 'last_credit_pull_d', 'collections_12_mths_ex_med', 'acc_now_delinq'], inplace=True)
Alasan penghapusan baris di atas karen nilai missing values yang sedikit. 14 kolom tersebut memiliki missing values di bawah 10%
# Mengganti nilai NaN dengan median pada kolom1 dan kolom2
df_loan[['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']] = df_loan[['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']].apply(lambda x: x.fillna(x.median()))
Alasan mengganti dengan nilai Median adalah sebagai berikut : <br>
* Kolom bertipe Numeric
* Berdistribusi Skewed
* Nilai Missing Values lebih dari 10%
print('Cek Missing Values Kembali')
df_loan.isna().sum()
## FEATURE ENGINEERING
### Menghapus simbol + dan kata years pada kolom emp_lenght
# Menghapus kata "years" pada kolom tertentu
df_loan['emp_length'] = df_loan['emp_length'].str.replace('years', '')
# Menghapus kata "+" pada kolom tertentu
df_loan['emp_length'] = df_loan['emp_length'].str.replace('+', '')
# Menghapus kata "<" pada kolom tertentu
df_loan['emp_length'] = df_loan['emp_length'].str.replace('<', '')

# Menghapus kata "year" pada kolom tertentu
df_loan['emp_length'] = df_loan['emp_length'].str.replace('year', '')
df_loan['emp_length'] = df_loan['emp_length'].astype(int)
df_loan.head()
### Memisahkan kolom Tanggal dan Bulan
# Kolom issue_d

# Parse kolom sebagai datetime
df_loan['issue_d'] = pd.to_datetime(df_loan['issue_d'], format='%b-%y')

# Ekstrak komponen bulan dan tanggal
df_loan['bulan_issue_d'] = df_loan['issue_d'].dt.month
df_loan['tanggal_issue_d'] = df_loan['issue_d'].dt.day

# Hapus kolom asli
df_loan.drop('issue_d', axis=1, inplace=True)
# last_credit_pull_d

# Parse kolom sebagai datetime
df_loan['last_credit_pull_d'] = pd.to_datetime(df_loan['last_credit_pull_d'], format='%b-%y')

# Ekstrak komponen bulan dan tanggal
df_loan['bulan_last_credit_pull_d'] = df_loan['last_credit_pull_d'].dt.month
df_loan['tanggal_last_credit_pull_d'] = df_loan['last_credit_pull_d'].dt.day

# Hapus kolom asli
df_loan.drop('last_credit_pull_d', axis=1, inplace=True)
# last_pymnt_d

# Parse kolom sebagai datetime
df_loan['last_pymnt_d'] = pd.to_datetime(df_loan['last_pymnt_d'], format='%b-%y')

# Ekstrak komponen bulan dan tanggal
df_loan['bulan_last_pymnt_d'] = df_loan['last_pymnt_d'].dt.month
df_loan['tanggal_last_pymnt_d'] = df_loan['last_pymnt_d'].dt.day

# Hapus kolom asli
df_loan.drop('last_pymnt_d', axis=1, inplace=True)
# last_pymnt_d

# Parse kolom sebagai datetime
df_loan['earliest_cr_line'] = pd.to_datetime(df_loan['earliest_cr_line'], format='%b-%y')

# Ekstrak komponen bulan dan tanggal
df_loan['bulan_earliest_cr_line'] = df_loan['earliest_cr_line'].dt.month
df_loan['tanggal_earliest_cr_line'] = df_loan['earliest_cr_line'].dt.day

# Hapus kolom asli
df_loan.drop('earliest_cr_line', axis=1, inplace=True)
df_loan.head()
### Menghapus kata months pada kolom term
# Menghapus kata "months" pada kolom tertentu
df_loan['term'] = df_loan['term'].str.replace('months', '')
df_loan['term'] = df_loan['term'].astype(int)
df_loan.head()
### Menghapus kata xx pada zip_code
df_loan['zip_code'] = df_loan['zip_code'].str.replace('xx', '')
df_loan['zip_code'] = df_loan['zip_code'].astype(int)
df_loan.head()
### Membuat kolom target yaitu risk
# Create a new column "risk" based on the values in "loan_status"
df_loan['risk'] = df_loan['loan_status'].apply(lambda x: 0 if x in ['Current', 'Fully Paid', 'In Grace Period'] else 1)
# Hapus kolom loan_status
df_loan = df_loan.drop('loan_status', axis=1)
df_loan.head()
Alasan membuat kolom target dengan judul risk karena tujuan dari pembuatan Machine Learning ini untuk mengembangkan dan mengetahui prediksi apakah user dengan pola data tertentu beresiko untuk mengganggu arus kas perusahaan atau tidak.

Pengambilan nilai pada kolom risk adalah dengan melihat isi dari nilai loan_status. Mengapa demikian? karena status pemimjaman mempresentasikan apakah user membayarnya tepat waktu, terlambat, atau bahkan tidak membayar sama sekali. Hal ini tentunya menjadi dasar user yang terlambat membayar atau bahkan tidak membayar dianggap beresiko maka diberi nilai 1. User yang membayar tepat waktu diberi nilai 0 yang berarti tidak beresiko.
## DATA PREPROCESSING
### Penanganan Outliers dengan Log Transformation
target_col = 'risk'  # Ganti dengan nama kolom target Anda
numerical_cols = df_loan.select_dtypes(include=['number']).columns.drop(target_col)
categorical_cols = df_loan.select_dtypes(include=['object', 'category']).columns
# Cek nilai negatif dan nol
df_loan[numerical_cols] = df_loan[numerical_cols].apply(lambda x: x + 1 if (x <= 0).any() else x)
# Log Transformation Kolom Numerical
df_loan[numerical_cols] = df_loan[numerical_cols].apply(np.log1p)
transformed_df = pd.concat([df_loan[categorical_cols], df_loan[numerical_cols], df_loan[target_col]], axis=1)
transformed_df.head()
### Encoding Kolom Kategorikal
#### Menghapus kolom kategorikal (Hapus Tambahan)
# daftar kolom yang akan di hapus
kolom_dihapus3 = ['title', 'sub_grade', 'addr_state']

# Hapus Kolom pada variabel kolom_dihapus
df_loan.drop(columns=kolom_dihapus3, inplace=True)
# memisahkan kolom kategorikal nominal dan ordinal
nominal = ['home_ownership', 'verification_status', 'loan_status', 'purpose', 'initial_list_status']
ordinal = ['grade']
# One Hot Encoding untuk kolom kategorikal Nominal
df_encoded = pd.get_dummies(df_loan, columns=['home_ownership', 'verification_status', 'purpose', 'initial_list_status'])
from sklearn.preprocessing import LabelEncoder
# Label Encoding untuk kolom kategorikal Ordinal
label_encoder = LabelEncoder()

df_encoded['grade_encode'] = label_encoder.fit_transform(df_encoded['grade'])

df_encoded = df_encoded.drop('grade', axis=1)
df_encoded.head()
#### Memindahkan Target ke paling Kanan
# Misalkan df adalah DataFrame-mu dan 'risk' adalah nama kolom target
kolom_target = 'risk'

# Memindahkan kolom target ke paling kanan
kolom_lain = [kolom for kolom in df_encoded.columns if kolom != kolom_target]
df_encoded = df_encoded[kolom_lain + [kolom_target]]
df_encoded.head()
#### Mengubah nilai Boolean menjadi 1 0
# Mengubah semua kolom boolean menjadi integer (1 dan 0)
df_encoded = df_encoded.applymap(lambda x: int(x) if isinstance(x, bool) else x)
df_encoded.head()
### Scaling
from sklearn.preprocessing import MinMaxScaler
# Identifikasi kolom boolean
boolean_cols = df_encoded.columns[df_encoded.isin([0, 1]).all()]

# Pilih kolom non-boolean untuk di-scaling
non_boolean_cols = df_encoded.columns.difference(boolean_cols)

# Inisialisasi MinMaxScaler
scaler = MinMaxScaler()

# Lakukan scaling hanya pada kolom non-boolean
df_encoded[non_boolean_cols] = scaler.fit_transform(df_encoded[non_boolean_cols])
df_encoded.head()
## MEMISAHKAN TRAIN DAN TEST SET
from sklearn.model_selection import train_test_split
X = df_encoded.drop('risk', axis=1)  # Fitur (input)
y = df_encoded['risk']  # Target (output)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:\n", X_train)
print("y_train:\n", y_train)
print("X_test:\n", X_test)
print("y_test:\n", y_test)
## CEK DAN HANDLING IMBALANCED CALSS
# Visualisasi dengan bar plot
sns.countplot(x=y_train, data=df_encoded)
plt.title('Distribusi Kelas')
plt.show()
from imblearn.over_sampling import SMOTE

# Menginisialisasi SMOTE
smote = SMOTE(random_state=42)

# Melakukan oversampling pada train set
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# Visualisasi dengan bar plot
sns.countplot(x=y_train_resampled, data=df_encoded)
plt.title('Distribusi Kelas')
plt.show()
## MODELING
### Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
# Inisialisasi model Logistic Regression
logreg = LogisticRegression(random_state=42)

# Melatih model dengan data pelatihan
logreg.fit(X_train_resampled, y_train_resampled)
# Melakukan prediksi pada data pengujian
y_pred = logreg.predict(X_test)
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {roc_auc:.4f}")
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
### Random Forest
from sklearn.ensemble import RandomForestClassifier
# Inisialisasi model
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# Latih model
model.fit(X_train_resampled, y_train_resampled)
# Prediksi pada data uji
y_pred = y_pred = model.predict(X_test)

# Evaluasi model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
from sklearn.model_selection import cross_val_score
# 3. Inisialisasi model Random Forest dengan pengaturan untuk menangani overfitting
model = RandomForestClassifier(
    n_estimators=100,          # Jumlah pohon
    max_depth=10,              # Batasi kedalaman pohon
    min_samples_split=4,       # Jumlah minimum sampel untuk split
    min_samples_leaf=2,        # Jumlah minimum sampel per daun
    max_features='sqrt',       # Pengambilan sampel fitur
    random_state=42            # Untuk reproduktifitas hasil
)

# 4. Melatih model
model.fit(X_train_resampled, y_train_resampled)

# 5. Prediksi pada data uji
y_pred = model.predict(X_test)

# 6. Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Akurasi: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{cm}")

# 7. Cross-validation untuk memastikan tidak overfitting
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")
Interpretasi: Mean CV Accuracy 0.97 menunjukkan bahwa rata-rata model Anda memiliki akurasi 97% ketika diuji dengan data yang berbeda dalam proses cross-validation. Ini menunjukkan model Anda memiliki performa yang konsisten dan kemungkinan besar akan bekerja dengan baik pada data yang belum pernah dilihat sebelumnya.
## FEATURE IMPORTANCE
### Logistic Regression
# Melihat koeficient
coefficients = logreg.coef_[0]

# Menggabungkan koefisien dengan nama fitur
feature_importance = pd.Series(coefficients, index=X_train_resampled.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print(feature_importance)
# Membuat visualisasi
plt.figure(figsize=(10, 12))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance (Logistic Regression)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()
### Random Forest
# Melihat koeficient
feature_importance_rf = pd.Series(model.feature_importances_, index=X_train_resampled.columns)

# Mengurutkan feature importance dari yang terbesar hingga terkecil
feature_importance_rf = feature_importance_rf.sort_values(ascending=False)

print(feature_importance_rf)
# Membuat visualisasi
plt.figure(figsize=(10, 12))
sns.barplot(x=feature_importance_rf, y=feature_importance_rf.index)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()
