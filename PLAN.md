# Sungur — Osmanlıca → Türkçe Yol Haritası

Modeller: sungur-ceviriyazi-* (Arap → Latin), sungur-ters-* (Latin → Arap, veri üretimi için),
ileride sungur-ocr-* ve sungur-sade-*.

Boru hattı: Görüntü → [OCR] → Arap harfli Osmanlıca → [Çeviriyazı] → Latin → [Sadeleştirme] → Bugünkü Türkçe

Çalışma düzeni: Claude görev metnini yazar (`prompts/NN_*.md`), Gemini kodu yazar, Claude inceler ve test eder.
ÖNEMLİ (Gemini için): Bu dosyayı ve prompts/ klasörünü değiştirme, geri alma (git checkout/restore/stash yok).
Commit yaparken bu dosyada değişiklik varsa onu da commit'e ekle.

## Güncel skorlar (altın TEST, sade mod)
| Sistem | CER | chrF | Not |
|---|---|---|---|
| B0 kural | %32.2 | 28.1 | |
| B1 sözlük | %26.7 | 40.8 | eski en iyi |
| B2 bilirkesi | %40.4 | 18.3 | DeepSeek API sarmalayıcısı |
| S1 sungur-small-v1 | %22.8 | 63.9 | tekrar döngüsü hatası |
| **S1b** (aynı model, kod çözme düzeltmesi) | **%15.7** | **66.2** | dev %13.7, makro %15.3 |
OCR (Azra-1-Mini, O0): yapay satır CER %14.1; Claude'un elle yazdığı 3 gerçek matbaa satırında boşluksuz CER %3–16.

## A. Veri ve ölçüm
- [x] 00  Kurulum + normalize.py (hemze/izafet korunur)
- [x] 01  Toplayıcılar: wiktionary (9.9 bin), gazetteer (22 bin, CC BY-NC), wikisource (2.5 bin satır)
- [x] 02  Temizlik + Vikikaynak (tr) hizalama v1
- [x] 03  Hizalama v2 + sızıntı koruması (leakage.py)
- [x] 04  Altın set eşiği 0.75 → 1.318 çift (dev 420 / test 898). Claude 200 test çiftini elle inceledi.
- [x] 05  LATOC düzeltildi (16 kitap, 64 bin cümle). Vikikaynak: 2.262 sayfa, 44 bin cümle. Altın: birebir 537 / serbest 781.
- [x] 06  Temizlik + ölçüm altyapısı + temel çözümler (B0–B2). Kelime eğitim verisi 27.367 çift (%67'si CC BY-NC yer adı).
- [x] 10  Geniş veri taraması (commit 2d4cd1a): tr Vikisözlük 6.470 çift (4.913 yeni), Wikidata 163, Vikipedi 111,
          Akis 8 bin el yazısı satır (görüntü + Latin; uçtan uca TEST için ayrıldı), CHURRO 237 sayfa (OCR testi),
          ota-bible 806 ayet (Türkçesi serbest çeviri), sadeleştirme çifti 462, printed_ocr 60 kitap ~240 bin satır makine OCR'ı.
          Claude incelemesi: başlığa göre eser eşleştirme başarısız (LATOC'ta yalnız Heft Peyker inandırıcı; Vikikaynak–IA listesi
          neredeyse tamamen yanlış). Bu iki eşleşme dosyası kullanılmayacak.
- [ ] 11  İçerik tabanlı eşleştirme (prompts/11_icerik_eslestirme.md): printed_ocr → S1b → Vikikaynak/LATOC'ta n-gram arama →
          eser bulunursa satır hizalama (iskelet ≥ 0.75) → gerçek (gürültülü) cümle çiftleri. Kelime verisi v2.
          Sentetik veriye gidiş-dönüş skoru (rt_cer).

## B. Çeviriyazı (Arap → Latin)
- [x] 07  Sungur v1 (commit d4b24d2): morph 57 bin çift, ters model val CER %19.75, geri çeviri dev CER %17.8 (07b sık kelime
          listesi + çıktı filtresi sonrası), sentetik vk 32.4 bin + latoc 56.2 bin cümle. İleri model ByT5-small, Colab A100,
          5 epoch, 65 dk (gereksiz gradient checkpointing açıktı; sonraki eğitimlerde kapalı).
- [x] 08  Kod çözme düzeltmesi (commit 276a3b7): girdiye bağlı uzunluk sınırı + tekrar tespiti → S1b. Bozuk satır 28 → 3.
- [ ] 12  Çeviriyazı v2 (11'den sonra karar verilecek):
          a) Gerçek OCR çiftleri (11) + kelime v2 + rt_cer ile filtrelenmiş sentetik veri
          b) Kısa satır verisi ("Dedi.", "Gülter:" gibi): durmayı öğretmek için
          c) Kelime v2 ile ters modeli yeniden eğitip geri çeviriyi yenilemek
          d) ByT5-base (Colab A100, ~2.5–3 saat, ~35 kredi)
          Bilinen zayıflıklar: Arapça kökenli kelimeler, k/g (ک) ayrımı, özel adlar ("Asmânlı").

## C. OCR
- [x] 09  Temel çözüm (commit 290ed11): klasik + Kraken bölütleme, Azra-1-Mini-0.8b, yapay satır ölçümü, run_ocr.py.
- [x] 09b Düzeltmeler (commit 2eaeb92): font glif doğrulaması, sütun bölme, WSL toplu çıkarım batch 16 → 5.5 satır/sn
          (1.000 sayfa ≈ 1 saat, 3060). Matbaada iyi, taş baskıda zayıf. Gerçek sayfa OCR altın seti yok (CHURRO el yazması).
- [ ] 13  sungur-ocr: Azra üzerine ince ayar (yapay satırlar + gerçek satırlar + öz-eğitim). Önce gerçek matbaa OCR test seti.
          Claude'un uçtan uca denemesi (sadbegefendiaret00ahme/mid_0115, 16 satır): Azra keşideli (uzatılmış) harflerde
          fazladan harf uyduruyor (پیمـانه → پیمهانه, میـخانه → میهخانه, نهـاندر → نمهاندر) → yapay satırlara rastgele keşide ekle.
          Sayfa numarası süslemesi "١١١" → "1110". Çeviriyazıda sık kelimeler bile hatalı ("tarafından" → "tırfından",
          "bilemediğim" → "bile medikem", "fikrinde" → "Fekrende").

## D. Sadeleştirme (Latin Osmanlıca → bugünkü Türkçe)
- [ ] 14  Veri: Vikikaynak sadeleştirme çiftleri (462), Ottoman2Turkish sözlüğü (10 bin, yalnız yerel), Vikisözlük anlamları,
          ota-bible (serbest çeviri), dil modeliyle üretilecek gümüş veri (kullanım şartları kontrol edilecek)
- [ ] 15  QLoRA (3–4B model), Colab
- [ ] 16  Değerlendirme (chrF + gözle kontrol)

## E. Ürün
- [ ] 17  pipeline.py + Gradio arayüzü (görüntü → Arap → Latin → sade)
- [ ] 18  Model kartları. Gazetteer ve printed_ocr CC BY-NC → model ticari olmayan lisansla yayımlanmalı.

## Kaynak / kredi
- Colab: 100 kredinin ~15'i harcandı (07c, A100 ~1.3 saat).

## Benzer çalışmalar (araştırma: 2026-11-05)
- Osmanlica.com (İÜC, Mina Arge): aynı 3 adım. Kapalı ve ticari; OCR %96, çeviriyazı %75 olarak bildirilmiş.
- Transleyt (Aicoor): ticari; OCR + çeviriyazı + sadeleştirme. Bağımsız ölçüm yok.
- OttomanNLP: Azra-1-Mini (açık OCR, temelimiz), Akis/CHURRO/printed-pages veri setleri. Çeviriyazı modeli yayımlanmadı.
- bilirkesi/osmanlica-v1: altın setimizde B2 olarak ölçüldü (%40.4).
- Akademik: OtoL (kural tabanlı, 2021), Dölek & Kurt CRNN OCR (2022), LREC 2026 IJMES dil modelleri.
- Boşluk: açık, uçtan uca ve herkese açık bir test setiyle ölçülmüş sistem yok. Bizim farkımız bu.

## Notlar
- Dil modeli çıktısıyla model eğitme: Gemini/Claude kullanım şartları gümüş veri aşamasından önce kontrol edilecek.
- Kürtçe (Sorani) sayfalar Vikikaynak'ın Osmanlıca kategorisine karışmış; 02'de ayıklanıyor.
- Yasak kaynaklar: Redhouse, Kubbealtı, Nişanyan, lugatim, erisale/Risale-i Nur (telif 2030'a kadar), ücretli e-kitap, tez tam metinleri.
- LATOC kitap metinleri yalnız yerel; yeniden dağıtılmaz.
