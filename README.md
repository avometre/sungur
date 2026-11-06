# Sungur

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

## Durum (Kasım 2026)

Altın test seti: Osmanlıca Vikikaynak ↔ Türkçe Vikikaynak sayfalarından hizalanmış 898 satır (eğitimden tamamen ayrı tutulur).

| Sistem | Test CER (sade) | chrF |
|---|---|---|
| Kural tabanlı (B0) | %32.2 | 28.1 |
| Sözlük tabanlı (B1) | %26.7 | 40.8 |
| **Sungur çeviriyazı v1 (ByT5-small)** | **%15.7** | **66.2** |

- OCR: [Azra-1-Mini](https://huggingface.co/OttomanNLP/Azra-1-Mini-0.8b) + satır bölütleme (matbaa baskısında iyi, taş baskıda zayıf).
- Sadeleştirme: henüz başlamadı.
- Yol haritası ve ayrıntılar: [PLAN.md](PLAN.md). Görev metinleri: [prompts/](prompts/).

## Lisans

- **Kod:** Apache-2.0 ([LICENSE](LICENSE)).
- **Veri ve modeller bu depoda yok.** Eğitim verisinin bir kısmı ticari olmayan lisanslıdır
  (OttomanNLP yer adı sözlüğü ve basılı sayfa OCR'ı: CC BY-NC 4.0). Bu yüzden yayımlanacak modeller
  **CC BY-NC 4.0** ile yayımlanacaktır.
- LATOC kitap metinleri yalnızca yerel araştırma için kullanılır, yeniden dağıtılmaz.
- Vikikaynak metinleri CC BY-SA 4.0'dır.
