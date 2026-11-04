# osmanlica-ai

Osmanlı Türkçesi (Arap harfli) metin ve matbû/el yazması görüntüleri günümüz Türkçesine aktaran uçtan uca yapay zekâ boru hattı.

## Proje Mimarisi

Sistem 3 temel aşamadan oluşur:

1. **OCR (Optik Karakter Tanıma):** Matbû veya rik'a el yazması Osmanlıca sayfa ve belge görüntülerinden Arap harfli metnin tespiti ve tanınması.
2. **Çeviriyazı (Transliterasyon):** Arap harfli Osmanlıca metnin fonetik ve imla kurallarına uygun olarak Latin harfli transkripsiyon alfabesine aktarılması.
3. **Sadeleştirme (Modernleştirme):** Tarihsel ve ağdalı Osmanlıca kelime ve terkiplerin günümüz anlaşılır Türkiye Türkçesine çevrilmesi.

## Kurulum

Proje, Python 3.12 ve CUDA destekli PyTorch ortamında çalışacak şekilde yapılandırılmıştır. Hızlı paket yönetimi için `uv` kullanılır.

### 1. Sanal Ortam Oluşturma

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\activate
```

### 2. PyTorch (CUDA 12.8) Kurulumu

NVIDIA GPU hızlandırması (RTX 3060 vb.) için CUDA tekeri:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 3. Bağımlılıkların Yüklenmesi

```powershell
uv pip install -r requirements.txt
```

### 4. Ortam ve Donanım Doğrulaması

Ortamın, PyTorch CUDA desteğinin ve GPU matris çarpımının sağlıklı çalıştığını test etmek için:

```powershell
.venv\Scripts\python scripts/check_env.py
```

## Testlerin Çalıştırılması

Birim ve entegrasyon testlerini çalıştırmak için:

```powershell
.venv\Scripts\python -m pytest -q
```
