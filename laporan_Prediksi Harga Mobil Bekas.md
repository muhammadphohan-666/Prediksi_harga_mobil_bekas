# Laporan Proyek Machine Learning - Prediksi Harga Mobil Bekas

## Domain Proyek

Pasar mobil bekas memiliki variasi harga yang besar karena harga kendaraan dipengaruhi oleh banyak faktor, seperti umur mobil, jarak tempuh, merek, model, tipe bahan bakar, transmisi, kondisi kerusakan, dan karakteristik kendaraan lainnya. Bagi calon pembeli, menentukan apakah suatu mobil bekas memiliki harga yang wajar dapat menjadi sulit karena informasi yang tersedia sering kali tidak lengkap. Bagi penjual, kesalahan menentukan harga dapat menyebabkan mobil terlalu lama terjual atau justru dijual terlalu murah.

Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi harga mobil bekas berdasarkan atribut kendaraan. Dengan pendekatan regresi, model diharapkan dapat membantu memberikan estimasi harga yang lebih objektif berdasarkan pola historis pada data.

sumber data :
- The Devastator. "Uncovering Factors That Affect Used Car Prices." Kaggle. https://www.kaggle.com/datasets/thedevastator/uncovering-factors-that-affect-used-car-prices

## Business Understanding

### Problem Statements

Permasalahan yang ingin diselesaikan dalam proyek ini adalah sebagai berikut.

- Bagaimana memprediksi harga mobil bekas berdasarkan karakteristik kendaraan?
- Fitur apa saja yang dapat digunakan untuk membantu model memahami variasi harga mobil bekas?
- Model regresi mana yang memberikan performa lebih baik untuk memprediksi harga mobil bekas?

### Goals

Tujuan dari proyek ini adalah sebagai berikut.

- Membangun model machine learning untuk memprediksi harga mobil bekas.
- Melakukan data understanding dan data preparation agar data siap digunakan untuk modeling.
- Membandingkan performa dua algoritma regresi, yaitu Random Forest Regressor dan Decision Tree Regressor.
- Memilih model terbaik berdasarkan metrik evaluasi regresi, yaitu MAE, RMSE, dan R2.

### Solution Statements

Untuk mencapai tujuan tersebut, proyek ini menggunakan dua pendekatan model regresi.

- Solution 1: Decision Tree Regressor digunakan sebagai pendekatan pohon keputusan tunggal. Model ini mempelajari aturan pemisahan data berdasarkan fitur yang paling informatif terhadap target, sehingga dapat digunakan sebagai baseline pembanding untuk melihat performa model tree-based yang lebih sederhana.
- Solution 2: Random Forest Regressor digunakan sebagai pendekatan ensemble berbasis banyak pohon keputusan. Model ini membangun beberapa Decision Tree dari subset data dan subset fitur yang berbeda, lalu menggabungkan hasil prediksi seluruh pohon sebagai prediksi akhir.

Kedua model dievaluasi menggunakan metrik yang sama, yaitu MAE, RMSE, dan R2. Model terbaik dipilih berdasarkan performa pada data test, terutama nilai MAE dan RMSE yang lebih rendah serta R2 yang lebih tinggi.

## Data Understanding

Dataset yang digunakan adalah dataset `autos.csv` dari Kaggle: "Uncovering Factors That Affect Used Car Prices" oleh The Devastator.

Tautan dataset: https://www.kaggle.com/datasets/thedevastator/uncovering-factors-that-affect-used-car-prices

Dataset awal memiliki:

- 371.528 baris data.
- 21 kolom.
- Target prediksi berupa kolom `price`.
- Masalah yang diselesaikan berupa regresi karena target berbentuk nilai numerik kontinu.

Setelah proses pembersihan data dan outlier filtering, jumlah data yang digunakan untuk modeling menjadi:

- 319.772 baris.
- 12 kolom setelah kolom tidak relevan dihapus.
- 11 fitur untuk prediksi.
- 1 target, yaitu `price`.

### Variabel pada Dataset

Berikut ringkasan variabel awal pada dataset.

- `index`: indeks data.
- `dateCrawled`: tanggal data iklan dikumpulkan.
- `name`: nama atau judul iklan mobil.
- `seller`: jenis penjual.
- `offerType`: tipe penawaran.
- `price`: harga mobil, digunakan sebagai target.
- `abtest`: kategori eksperimen A/B.
- `vehicleType`: tipe kendaraan, seperti sedan, coupe, SUV, dan lainnya.
- `yearOfRegistration`: tahun registrasi kendaraan.
- `gearbox`: tipe transmisi.
- `powerPS`: tenaga kendaraan dalam PS.
- `model`: model kendaraan.
- `kilometer`: jarak tempuh kendaraan.
- `monthOfRegistration`: bulan registrasi kendaraan.
- `fuelType`: jenis bahan bakar.
- `brand`: merek kendaraan.
- `notRepairedDamage`: status kerusakan yang belum diperbaiki.
- `dateCreated`: tanggal iklan dibuat.
- `nrOfPictures`: jumlah gambar.
- `postalCode`: kode pos lokasi.
- `lastSeen`: tanggal terakhir iklan terlihat.

### Exploratory Data Analysis

Beberapa tahapan EDA yang dilakukan adalah sebagai berikut.

- Melihat lima data teratas menggunakan `head()`.
- Melihat jumlah baris dan kolom menggunakan `shape`.
- Melihat tipe data dan jumlah non-null menggunakan `info()`.
- Mengecek missing value menggunakan `isnull().sum()`.
- Melihat statistik deskriptif menggunakan `describe().T`.
- Mengecek duplikasi menggunakan `duplicated().sum()`.
- Melihat data dengan `price = 0`.
- Membuat boxplot untuk melihat outlier, terutama pada `yearOfRegistration` dan kolom numerik.
- Membuat histogram untuk melihat distribusi data numerik.

Berdasarkan EDA, ditemukan beberapa kondisi data yang perlu ditangani.

- Kolom `price` memiliki nilai 0 dan nilai ekstrem.
- Kolom `yearOfRegistration` memiliki nilai tidak realistis seperti 1000 dan 9999.
- Kolom `powerPS` memiliki nilai 0 dan nilai ekstrem.
- Kolom `monthOfRegistration` memiliki nilai 0, padahal bulan valid adalah 1 sampai 12.
- Beberapa kolom kategorikal memiliki missing value, seperti `vehicleType`, `gearbox`, `model`, `fuelType`, dan `notRepairedDamage`.
- Kolom `nrOfPictures` tidak informatif karena nilainya 0.

## Data Preparation

Tahapan data preparation dilakukan secara berurutan sebagai berikut.

### Menghapus Harga Tidak Valid

Data dengan `price = 0` dihapus karena harga mobil tidak mungkin bernilai 0. Nilai tersebut dianggap sebagai data tidak valid.

### Mengubah Kolom Tanggal

Kolom `dateCrawled`, `dateCreated`, dan `lastSeen` diubah menjadi tipe datetime. Tujuannya agar kolom tanggal dapat dianalisis dan digunakan untuk membuat fitur baru.

### Mengubah Nilai 0 Menjadi Missing Value

Nilai 0 pada `powerPS` dan `monthOfRegistration` diubah menjadi NaN.

Alasannya:

- `powerPS = 0` tidak merepresentasikan tenaga kendaraan yang valid.
- `monthOfRegistration = 0` bukan bulan yang valid.

Nilai NaN ini tidak langsung dihapus karena akan ditangani menggunakan imputasi setelah data dibagi menjadi training dan testing.

### Menangani Outlier

Setelah nilai 0 pada `powerPS` diubah menjadi NaN, kolom `powerPS` ditangani menggunakan metode IQR. Tahap ini dilakukan karena `powerPS` memiliki nilai ekstrem yang tidak realistis untuk tenaga kendaraan.

Nilai batas bawah dan batas atas dihitung menggunakan kuartil pertama, kuartil ketiga, dan IQR. Data dengan `powerPS` di luar batas IQR dihapus, sedangkan nilai NaN tetap dipertahankan karena akan ditangani pada tahap imputasi setelah train-test split.

### Feature Engineering:

Fitur baru `car_age` dibuat dengan rumus:

```text
car_age = tahun dateCreated - yearOfRegistration
```

Fitur ini digunakan karena umur mobil merupakan salah satu faktor penting yang memengaruhi harga mobil bekas.

### Filtering Tahun Registrasi, Umur Mobil, dan Harga

Setelah fitur `car_age` dibuat, dilakukan filtering lanjutan pada `yearOfRegistration`, `car_age`, dan `price`.

Tahap filtering yang dilakukan adalah sebagai berikut.

- `yearOfRegistration` difilter agar berada pada rentang yang masuk akal, yaitu minimal 1990 dan maksimal sesuai tahun data dikumpulkan.
- `car_age` difilter agar tidak bernilai negatif.
- `price` difilter menggunakan batas minimal 100 dan batas atas quantile 0.99 untuk mengurangi pengaruh harga ekstrem.


### Menghapus Kolom Tidak Relevan

Beberapa kolom dihapus karena tidak digunakan dalam modeling atau kurang informatif.

- `index` dihapus karena hanya berupa indeks.
- `nrOfPictures` dihapus karena nilainya tidak informatif.
- `seller`, `offerType`, dan `abtest` dihapus karena tidak banyak memberikan informasi untuk prediksi.
- `name` dihapus karena berupa teks bebas.
- `dateCrawled`, `dateCreated`, dan `lastSeen` dihapus karena informasi tanggal penting sudah digunakan untuk membuat `car_age`.
- `postalCode` dihapus karena berupa kode wilayah.

### Train-Test Split

Data dibagi menjadi data training dan data testing dengan proporsi 80:20.

- Data training: 255.817 baris.
- Data testing: 63.955 baris.

Pembagian data dilakukan sebelum imputasi agar proses imputasi tidak menyebabkan data leakage.

### Imputasi Missing Value

Imputasi dilakukan setelah train-test split.

- Kolom numerik diisi menggunakan median.
- Kolom kategorikal diisi menggunakan modus atau nilai yang paling sering muncul.

Imputer di-fit hanya pada data training, lalu diterapkan ke data training dan testing. Hal ini dilakukan untuk mencegah informasi dari data test bocor ke proses training.

Hasil imputasi pada data training menunjukkan bahwa missing value berhasil dihilangkan pada kolom yang sebelumnya memiliki NaN.

- `vehicleType`: 14.546 missing menjadi 0.
- `gearbox`: 10.932 missing menjadi 0.
- `powerPS`: 24.151 missing menjadi 0.
- `model`: 10.813 missing menjadi 0.
- `monthOfRegistration`: 21.303 missing menjadi 0.
- `fuelType`: 17.253 missing menjadi 0.
- `notRepairedDamage`: 44.514 missing menjadi 0.

### Encoding Kategorikal

Kolom kategorikal diubah menjadi bentuk numerik menggunakan One Hot Encoding. Parameter `handle_unknown="ignore"` digunakan agar model tidak error jika terdapat kategori baru pada data test.

### Normalisasi atau Standardisasi

Normalisasi atau standardisasi tidak diterapkan karena model yang digunakan adalah Decision Tree dan Random Forest. Kedua model tersebut berbasis pohon keputusan dan tidak sensitif terhadap skala fitur numerik.

## Modeling

Proyek ini menggunakan dua algoritma regresi.

### Decision Tree Regressor

Decision Tree Regressor adalah model berbasis pohon keputusan yang membagi data berdasarkan aturan tertentu untuk memprediksi nilai target.

Parameter utama yang digunakan:

- `random_state=42`

Kelebihan Decision Tree:

- Memiliki struktur pohon keputusan yang jelas.
- Dapat menangkap hubungan non-linear antar fitur.
- Tidak membutuhkan standardisasi fitur.

Keterbatasan Decision Tree:

- Performa model sangat dipengaruhi oleh struktur dan kedalaman pohon.
- Sebagai model tunggal, Decision Tree tidak memiliki mekanisme agregasi prediksi seperti model ensemble.

### Random Forest Regressor

Random Forest Regressor adalah model ensemble berbasis banyak Decision Tree. Secara arsitektur, setiap pohon dibangun dari variasi subset data dan fitur, kemudian hasil prediksi dari seluruh pohon digabungkan untuk menghasilkan prediksi akhir.

Parameter utama yang digunakan:

- `n_estimators=100`
- `random_state=42`
- `n_jobs=-1`

Kelebihan Random Forest:

- Menggunakan pendekatan ensemble dengan banyak pohon keputusan.
- Menghasilkan prediksi akhir melalui agregasi prediksi dari seluruh pohon.
- Cocok untuk data tabular dengan kombinasi fitur numerik dan kategorikal.
- Tidak membutuhkan standardisasi fitur.

Kekurangan Random Forest:

- Waktu training lebih lama dibandingkan Decision Tree.
- Model lebih sulit diinterpretasikan dibandingkan satu Decision Tree.

## Evaluation

Karena proyek ini merupakan masalah regresi, metrik evaluasi yang digunakan adalah MAE, RMSE, dan R2.

### Mean Absolute Error

MAE mengukur rata-rata selisih absolut antara harga aktual dan harga prediksi.

```text
MAE = (1 / n) * sum(|y_actual - y_pred|)
```

Semakin kecil nilai MAE, semakin baik performa model. MAE mudah dijelaskan karena satuannya sama dengan target, yaitu harga.

### Root Mean Squared Error

RMSE mengukur akar dari rata-rata kuadrat error.

```text
RMSE = sqrt((1 / n) * sum((y_actual - y_pred)^2))
```

RMSE memberikan penalti lebih besar pada kesalahan prediksi yang besar. Semakin kecil nilai RMSE, semakin baik performa model.

### R2 Score

R2 mengukur seberapa baik model menjelaskan variasi pada target.

```text
R2 = 1 - (SS_res / SS_tot)
```

Nilai R2 semakin mendekati 1 berarti model semakin baik dalam menjelaskan variasi harga.

### Hasil Evaluasi

Hasil evaluasi Random Forest Regressor:

- Train RMSE: 790.89
- Train MAE: 453.68
- Train R2: 0.9778
- Test RMSE: 1632.42
- Test MAE: 938.88
- Test R2: 0.9049

Hasil evaluasi Decision Tree Regressor:

- Train RMSE: 583.98
- Train MAE: 216.71
- Train R2: 0.9879
- Test RMSE: 2024.95
- Test MAE: 1109.10
- Test R2: 0.8537

Berdasarkan hasil evaluasi, Random Forest Regressor dipilih sebagai model terbaik karena memiliki performa test yang lebih baik.

Alasan pemilihan Random Forest:

- Test MAE Random Forest lebih rendah daripada Decision Tree.
- Test RMSE Random Forest lebih rendah daripada Decision Tree.
- Test R2 Random Forest lebih tinggi daripada Decision Tree.
- Gap antara performa training dan testing lebih terkendali dibandingkan Decision Tree.

Decision Tree memiliki performa training yang sangat tinggi, tetapi performa test lebih rendah dibandingkan Random Forest. Perbedaan ini menunjukkan bahwa Random Forest memberikan hasil yang lebih konsisten pada data test melalui pendekatan ensemble yang menggabungkan banyak pohon keputusan.

## Kesimpulan

Proyek ini berhasil membangun model machine learning untuk memprediksi harga mobil bekas menggunakan pendekatan regresi. Setelah melalui proses data understanding, data preparation, modeling, dan evaluation, diperoleh bahwa Random Forest Regressor merupakan model terbaik dibandingkan Decision Tree Regressor.

Model terbaik menghasilkan:

- Test MAE: 938.88
- Test RMSE: 1632.42
- Test R2: 0.9049

Nilai MAE menunjukkan bahwa rata-rata kesalahan prediksi harga mobil pada data test adalah sekitar 938.88. Nilai R2 sebesar 0.9049 menunjukkan bahwa model mampu menjelaskan sebagian besar variasi harga mobil bekas pada dataset.

Meskipun hasil model sudah cukup baik, scatter plot actual vs predicted menunjukkan bahwa prediksi masih memiliki sebaran error, terutama pada harga yang lebih tinggi. Hal ini wajar karena harga mobil bekas juga dipengaruhi faktor lain yang tidak tersedia dalam dataset, seperti kondisi fisik detail, riwayat servis, lokasi pasar, dan negosiasi harga.


