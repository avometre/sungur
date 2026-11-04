# Osmanlıca → Türkçe — Yol Haritası

Boru hattı: Görüntü → [OCR] → Arap harfli Osmanlıca → [Çeviriyazı] → Latin → [Sadeleştirme] → Bugünkü Türkçe

Çalışma düzeni: Claude görev metnini yazar (`prompts/NN_*.md`), Gemini kodu yazar, Claude inceler ve test eder.

## A. Veri
- [x] 00  Kurulum + normalize.py
- [x] 01  Toplayıcılar: wiktionary (9.9 bin), gazetteer (22 bin, CC BY-NC), wikisource (2.5 bin satır)
- [x] 02  Temizlik + yer adı okunuş seçimi + Vikikaynak (tr) hizalama v1 (15 sayfa, 380 güvenilir çift)
- [x] 03  Hizalama v2: 33 sayfa, 1.343 çift (dev 7 sayfa / test 26 sayfa) + sızıntı koruması
- [ ] 04  Altın set eşiği 0.75 (ölçümle belirlendi) + LATOC + Internet Archive kataloğu
- Not: Altın setteki Latin metinlerin bir kısmı sadeleştirilmiş baskılardan alınmış (ör. "uzaklarda" → "uzaktan").
  Bu yüzden test sonuçları sayfa bazında da raporlanacak; en güvenilir ölçü chrF olacak.

## B. Çeviriyazı (Arap → Latin)
- [ ] 05  Değerlendirme altyapısı (CER, WER, chrF) + kural/sözlük tabanlı basit çözüm
- [ ] 06  Gümüş veri: güçlü bir dil modeliyle satır çevirisi, iskelet benzerliğiyle filtreleme
- [ ] 07  ByT5-base v1 (yerel GPU)
- [ ] 08  Ters model (Latin → Arap) ile LATOC'un Arap harfine çevrilmesi (geri çeviri) → büyük yapay veri
- [ ] 09  ByT5 v2 (gerçek + gümüş + geri çeviri verisi)

## C. OCR
- [ ] 10  Satır bölütleme + Azra-1-Mini temel çözümü, HF OCR setinde CER
- [ ] 11  Yapay satır görüntüsü üretici (Osmanlıca fontlar + bozulma efektleri)
- [ ] 12  Gerekirse Azra'ya LoRA ince ayarı

## D. Sadeleştirme (Latin Osmanlıca → bugünkü Türkçe)
- [ ] 13  Veri: Wiktionary `turkce` alanı (kelime düzeyi) + LATOC cümlelerinin dil modeliyle sadeleştirilmesi
- [ ] 14  QLoRA (3–4B model), yerel GPU veya Colab
- [ ] 15  Değerlendirme (chrF + gözle kontrol)

## E. Ürün
- [ ] 16  pipeline.py + Gradio arayüzü
- [ ] 17  Model kartları. Gazetteer CC BY-NC olduğu için model ticari olmayan lisansla yayımlanmalı.

## Notlar
- Dil modeli çıktısıyla model eğitme: Gemini/Claude kullanım şartları gümüş veri aşamasından önce kontrol edilecek.
- Kürtçe (Sorani) sayfalar Vikikaynak'ın Osmanlıca kategorisine karışmış; 02'de ayıklanıyor.
