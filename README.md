## Cuneyt Ozdemir Solo Pipeline

Bu pipeline tek komutta su adimlari yapar:
- `linkCNN.txt` dosyasindaki linkleri indirir
- sahne degisimlerini otomatik bulur
- `c.webp` referans yuzune gore sadece tek kisi ve hedef kisi olan sahneleri ayiklar

### Hazirlik

0. Python surumu: `3.11.x` (onerilen) veya `3.12.x`.
1. `linkCNN.txt` olustur ve her satira bir video URL koy.
2. Referans fotografin `c.webp` olarak proje kokunde kalsin.
3. Gereksinimleri yukle:

```bash
pip install -r requirements.txt
```

Bu pipeline `face_recognition -> dlib` kullandigi icin Windows'ta Visual Studio C++ Build Tools gerekir.
En stabil ortam icin yeni venv'i 3.11 ile olustur:

```bash
py -3.11 -m venv venv311
.\venv311\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Eger `dlib` kurulumunda hata alirsan:
1. Visual Studio Build Tools kur (`Desktop development with C++` workload secili olmali).
2. CMake'in PATH'te oldugunu dogrula: `cmake --version`
3. Sonra tekrar:

```bash
pip install --no-cache-dir dlib
pip install -r requirements.txt
```

### Calistirma

```bash
python scripts/pipeline_cuneyt_solo.py
```

### Cikti

- Indirilen videolar: `data/raw_videos`
- Filtrelenmis klipler: `data/cuneyt_solo_clips`

### Opsiyonel Parametreler

```bash
python scripts/pipeline_cuneyt_solo.py ^
  --links-file linkCNN.txt ^
  --reference-image c.webp ^
  --scene-threshold 30 ^
  --sample-count 3 ^
  --face-tolerance 0.47 ^
  --min-duration 1.2
```
