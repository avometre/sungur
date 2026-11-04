# Osmanlıca → Türkçe — Yol Haritası

Boru hattı: Görüntü → [OCR] → Arap harfli Osmanlıca → [Çeviriyazı] → Latin → [Sadeleştirme] → Bugünkü Türkçe

Çalışma düzeni: Claude görev metnini yazar (`prompts/NN_*.md`), Gemini kodu yazar, Claude inceler ve test eder.

## A. Veri
- [x] 00  Kurulum + normalize.py
- [x] 01  Toplayıcılar: wiktionary (9.9 bin), gazetteer (22 bin, CC BY-NC), wikisource (2.5 bin satır)
- [x] 02  Temizlik + yer adı okunuş seçimi + Vikikaynak (tr) hizalama v1 (15 sayfa, 380 güvenilir çift)
- [x] 03  Hizalama v2: 33 sayfa, 1.343 çift (dev 7 sayfa / test 26 sayfa) + sızıntı koruması
- [x] 04  Altın set eşiği 0.75 → 1.318 çift (dev 420 / test 898). LATOC 17/143 kitap indirildi.
          IA kataloğu: 2.856 eser, ~2.000'i kamu malı; IA'nın kendi OCR'ı kullanılamaz durumda.
          Claude 200 test çiftini elle inceledi: yanlış hizalama yok. Sayfalar birebir/serbest diye etiketlendi.
- [x] 05  LATOC düzeltildi (16 kitap, 64 bin cümle, bozuk harf <%0.1). Vikikaynak: 2.262 sayfa, 44 bin cümle.
          Altın set: birebir 537 / serbest 781 çift.
- [ ] 06  Temizlik (Çağatayca/Karahanlıca metinler, sadeleştirilmiş Atatürk konuşmaları, mısra no) + ölçüm
          altyapısı + temel çözümler B0 kural / B1 sözlük / B2 bilirkesi / B3 LLM (iki yön) + kelime düzeyi eğitim verisi
- Kritik eksik: cümle düzeyinde gerçek eğitim çifti yok (hizalanan çiftlerin hepsi test setinde).
  Yapay veri yöntemi GÖREV 06'nın ölçümüne göre seçilecek:
    a) LLM ters yönde (Latin→Arap) iyiyse: Vikikaynak/LATOC Latin cümleleri → LLM → Arap (geri çeviri)
    b) değilse: kelime düzeyi Latin→Arap ByT5 + ek kuralları ile kelime kelime geri çeviri
    c) ek olarak: Wikisource'ın eşleşmemiş Arap satırları → LLM ile ileri yönde gümüş veri
- Karar: Hedef çıktı üslubu = Vikikaynak tarzı geç dönem çeviriyazı (ör. "devlet-i aliyye", şapkalar korunur).
  LATOC (15.–16. yy, IJMES, "oldı/itdi") farklı dönem ve üslupta. Eğitimde "<ijmes>" üslup etiketiyle ayrılacak,
  ana ağırlık Vikikaynak'ta olacak.
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

## Benzer çalışmalar (araştırma: 2026-11-05)
- Osmanlica.com (İÜC, Mina Arge): bizimle aynı 3 adım. Kapalı ve ticari; OCR %96, çeviriyazı %75 olarak bildirilmiş.
- Transleyt (Aicoor): ticari; OCR + çeviriyazı + sadeleştirme. Bağımsız ölçüm yok.
- OttomanNLP / Azra-1-Mini: açık OCR, bizim OCR temelimiz. Çeviriyazı modeli henüz yayımlanmadı.
- bilirkesi/osmanlica-v1 (HF, MIT): çeviriyazı paketi; mimarisi ve başarısı belirtilmemiş.
  → GÖREV 06'da altın setimizde karşılaştırma için ölçülecek.
- Akademik: OtoL (kural tabanlı, 2021), Dölek & Kurt CRNN OCR (2022), LREC 2026 IJMES dil modelleri.
- Boşluk: açık, uçtan uca ve herkese açık bir test setiyle ölçülmüş sistem yok. Bizim farkımız bu olacak.

## Notlar
- Dil modeli çıktısıyla model eğitme: Gemini/Claude kullanım şartları gümüş veri aşamasından önce kontrol edilecek.
- Kürtçe (Sorani) sayfalar Vikikaynak'ın Osmanlıca kategorisine karışmış; 02'de ayıklanıyor.
