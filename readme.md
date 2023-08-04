# About

### WORK IN PROGRESS

Program API face recognition dirancang oleh tim Face Recognition Magang Lab ISR.

Program ini akan mengirim gambar dari sisi client ke sisi server dengan response berupa JSON berisikan koordinat prediksi keberadaan wajah, hasil prediksi emosi, hasil prediksi umur, dan hasil prediksi gender.

Prediksi wajah menggunakan Face Recognition yang disediakan dari dlib. Sepenuhnya dapat dilihat dari github [Face Recognition](https://github.com/ageitgey/face_recognition).

Program ini menggunakan bahasa **Python** dengan framework **Django**.

## Setup

Untuk memulai pastikan workspace anda telah memiliki Python dengan pip. Pastikan pip telah terpasang, apabila belum jalankan perintah di bawah:

```bash
python get-pip.py
```

Pasang cmake dengan mengunduh distribusi resmi dari [websitenya](https://cmake.org/download/), pastikan anda mengunduh distribusi yang sesuai dengan perangkat anda.

### MacOS / Linux

Sebagai user MacOS atau Linux anda bisa secara langsung menggunakan terminal atau sejenisnya dalam pemasangan library pendukung.

```bash
pip install -r requirements.txt
```

### Windows

Sayangnya dlib tidak memiliki distribusi resmi untuk perangkat Windows, namun tetap dapat dipasang melalui beberapa cara lainnya.

[Baca di sini untuk mengetahui cara pemasangan Face Recognition pada perangkat Windows.](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)

Setelah berhasil memasang dlib, anda bisa menjalankan perintah yang sama untuk memasang library pendukung sisanya:

```bash
pip install -r requirements.txt
```

## Running

Repository ini mengandung sisi server dan sisi client yang dapat dijalankan pada perangkat yang sama. Untuk menjalankan prediksi gambar, jalankan terlebih dahulu server dengan menggunakan perintah di bawah:

```bash
python manage.py runserver 8100
```

Angka **8100** menunjukkan port yang digunakan. Setelah server dijalankan, anda hanya tinggal menjalankan client_test.py seperti berikut:

```bash
python client_test.py
```

Anda dapat mengubah gambar yang diprediksi dengan menambahkan gambar yang anda ingin prediksi ke dalam direktori */images/* dan mengubah variabel *image_path* pada baris 7 di client_test.py.
