# Results Summary — Module C (9 Experiments)

> Bu belge, 3 zaman-frekans gösterimi (STFT, CWD, WVD) × 3 mimari (Custom-CNN, ResNet-50, ViT-Small)
> kombinasyonundan oluşan 9 deneyin nihai test sonuçlarını, karşılaştırma figürlerini ve
> yorumlarını kaydeder. Doğrudan makalenin **Results** ve **Discussion** bölümlerine kaynak olacaktır.
>
> Üretim tarihi: 2026-07-13 · Test seti: 6.000 örnek (sınıf başına ~750, SNR-stratified)
> · Kaynak veriler: `experiments/results/<tf>_<arch>/test_metrics.json`
> · Figür/tablo üretimi: `analysis/compare_all_experiments.py`

---

## 1. Ana Sonuç Tablosu

| TF | Mimari | Test Acc | Macro-F1 | -10 dB | -8 dB | -6 dB | ≥+6 dB |
|---|---|---|---|---|---|---|---|
| STFT | Custom-CNN | **98.22** | 98.22 | 86.5 | 95.0 | 97.6 | 99.9 |
| STFT | ResNet-50 | 98.00 | 98.00 | 83.5 | 95.3 | 98.4 | 99.8 |
| STFT | ViT-Small | 97.45 | 97.45 | 84.3 | 90.8 | 96.8 | 99.6 |
| CWD | Custom-CNN | 97.57 | 97.57 | 83.2 | 92.3 | 95.2 | 99.7 |
| CWD | ResNet-50 | 97.55 | 97.55 | 83.8 | 91.3 | 97.1 | 99.6 |
| CWD | ViT-Small | 97.15 | 97.15 | 85.4 | 90.8 | 94.7 | 99.4 |
| WVD | Custom-CNN | 89.57 | 89.61 | 26.5 | 53.8 | 77.6 | 99.5 |
| WVD | ResNet-50 | 88.43 | 88.47 | 22.4 | 46.7 | 75.5 | 99.2 |
| WVD | ViT-Small | 88.85 | 88.88 | 25.4 | 51.7 | 75.5 | 99.2 |

Değerler yüzde (%). `≥+6 dB` = +6…+20 dB arası 8 SNR seviyesinin ortalaması. Ham tablo: `analysis/summary_table.csv`.

**Gösterim ailesi ortalamaları (3 mimari üzerinden):**

| TF | Overall (min–max) | -10 dB | -8 dB | ≥+6 dB |
|---|---|---|---|---|
| STFT | 97.89 (97.45–98.22) | 84.8 | 93.7 | 99.74 |
| CWD | 97.42 (97.15–97.57) | 84.1 | 91.5 | 99.56 |
| WVD | 88.95 (88.43–89.57) | 24.8 | 50.7 | 99.30 |

---

## 2. Figürler

- **`analysis/tf_family_snr_robustness.png`** — Ana figür. 3 TF ailesi, mimari ortalaması (kalın çizgi)
  ± mimariler arası min/max bandı. WVD'nin düşük SNR'daki çöküşünü tek bakışta anlatır.
- **`analysis/all9_snr_robustness.png`** — Detay figür. 9 modelin tamamı; renk = TF gösterimi,
  çizgi stili/marker = mimari. Ek/supplementary materyal için.
- Deney-bazlı figürler (her `results/<tf>_<arch>/` altında): `confusion_matrix.png`,
  `per_snr_accuracy.png`, `class_snr_accuracy.png`.

---

## 3. Bulgular

### 3.1 Gösterim sıralaması: STFT ≳ CWD ≫ WVD
Genel test doğruluğunda STFT (~97.9%) ve CWD (~97.4%) başa baş; aralarındaki ~0.5 puanlık fark
mimari varyansının içinde kalıyor. WVD ise ~89% ile **~9 puan** geride. Ancak bu genel ortalama
asıl hikâyeyi gizliyor — fark tamamen düşük SNR rejiminden geliyor (§3.2).

### 3.2 WVD'nin düşük-SNR çöküşü (makalenin ana bulgusu)
STFT-ortalaması ile WVD-ortalaması arasındaki fark, SNR'a göre keskin biçimde değişiyor:

| SNR | STFT | WVD | Fark |
|---|---|---|---|
| -10 dB | 84.8 | 24.8 | **60.0** |
| -8 dB | 93.7 | 50.7 | 42.9 |
| -6 dB | 97.6 | 76.2 | 21.4 |
| -4 dB | 97.1 | 87.9 | 9.3 |
| -2 dB | 98.5 | 95.3 | 3.2 |
| 0 dB | 98.1 | 96.2 | 1.9 |
| +2 dB | 99.2 | 98.6 | 0.6 |
| ≥+4 dB | ~99.6 | ~99.3 | <1 |

WVD -10 dB'de **%24.8** — 8 sınıf için şans seviyesi (%12.5) çok az üzerinde, yani neredeyse
tesadüfî. +2 dB ve üzerinde üç gösterim de pratikte ayırt edilemez (~%99). Kesişim/yakınsama
noktası ≈ 0…+2 dB.

### 3.3 Yorum — cross-term interference (Discussion metni taslağı)
WVD ikinci-dereceden (quadratic) bir zaman-frekans dağılımıdır ve doğası gereği **cross-term**
üretir. Sinyaller mono-component olsa bile (her örnek tek radar sınıfı), eklenen AWGN
gürültü×sinyal ve gürültü×gürültü çapraz terimleri yaratır; bu terimler enerjileri sinyalle
ölçeklendiği için tüm zaman-frekans düzlemine yayılır ve düşük SNR'da otantik sinyal imzasını
boğar. STFT (pencereleme ile lineer, cross-term üretmez) ve CWD (Choi-Williams çekirdeği
çapraz terimleri σ=1.0 ile bastırır) bu bozulmaya çok daha dayanıklıdır. Yüksek SNR'da (≥+2 dB)
gürültü kaynaklı çapraz terimler ihmal edilebilir hale gelir ve WVD'nin yüksek zaman-frekans
çözünürlüğü avantajı geri gelir — bu yüzden üç gösterim yüksek SNR'da yakınsar.

**Sonuç cümlesi:** Elektronik harp gibi düşük-SNR ortamlarında ham WVD'nin yüksek çözünürlüğü,
gürültü kaynaklı cross-term kirlenmesi tarafından fazlasıyla dengelenmektedir; STFT ve CWD
operasyonel olarak anlamlı SNR aralığında belirgin şekilde daha dayanıklıdır.

### 3.4 Mimari ikincil bir faktör
Her gösterim ailesi içinde mimariler arası fark ≤1 puan: Custom-CNN ≈ ResNet-50 ≳ ViT-Small.
Hafif Custom-CNN, çok daha ağır ResNet-50/ViT-Small ile başa baş — hatta STFT'de en yüksek tek
skor (%98.22) ona ait. Bu, bu problem için kaynak-verimli küçük bir CNN'in yeterli olduğu
yönünde sağlam bir yan bulgu (kaynak/parametre kıyaslaması makaleye ek katkı).

### 3.5 Sınıf-bazlı gözlemler
Genelde en zor sınıflar LFM (en düşük precision, ~0.93–0.96; NLFM/discretized-chirp aileleriyle
karışıyor) ve SteppedFH (WVD'de recall belirgin düşük). Frank, Polyphase, CW tüm koşullarda en
kolay (F1 ≈ 0.99). Ayrıntı: her deneyin `confusion_matrix.png` ve `class_snr_accuracy.png`
dosyaları.

---

## 4. WVD Düşük-SNR Confusion Analizi

> Kaynak: `analysis/wvd_lowsnr_confusion.py` (per-sample `labels`/`preds`/`snr`,
> `eval_arrays.npz`). Figür: `analysis/wvd_lowsnr_confusion_custom_cnn.png`.
> **Önemli kısıt:** Bu analiz yalnızca WVD için yapılabildi — STFT/CWD checkpoint'leri
> (`.pth`, gitignore'lı ve büyük) diskten silinmiş; onların per-SNR confusion'ı ancak
> yeniden eğitimle elde edilebilir (bkz. §5 TODO). WVD zaten çöken gösterim olduğu için
> asıl ilgi çekici vaka budur.

WVD'nin hataları **SNR'a göre nitelik değiştiriyor** — tek bir sabit karışım çifti yok:

**(a) -10 dB — gürültü/cross-term kaynaklı dağınık karışım.** Doğruluk ~%25. Hiçbir çift
baskın değil; en yüksek çiftler bile ~%22–34 ve tahminler **frekans-çevik "çekim" sınıflarına**
(Costas, SteppedFH, NLFM sütunları) geniş biçimde yayılıyor. Bu, gürültü kaynaklı cross-term'lerin
WVD düzlemini geniş-bantlı desenlerle doldurmasının doğrudan imzası — model bunu frekans-atlamalı
sınıflara benzetiyor. Sınıf-bazlı recall (3 mimari pooled):

| Sınıf | -10 dB recall | Sınıf | -10 dB recall |
|---|---|---|---|
| LFM | **9.6%** (en kötü) | Costas | 28.7% |
| Barker | 19.4% | SteppedFH | 30.2% |
| NLFM | 21.1% | Polyphase | 30.6% |
| CW | 24.4% | Frank | **34.0%** (en iyi) |

**(b) -8 → -6 dB — yapısal karışıma geçiş.** Gürültü çekildikçe hatalar **gerçekten benzer
çiftlere** toplanıyor: LFM↔NLFM (chirp ailesi; -8 dB'de %27, -6 dB'de %15/%13) ve
Frank↔Polyphase (matris-faz ailesi; ~%11–24). -6 dB'de matris neredeyse köşegen (acc ~%76).

**Hipotez edilen çiftlerin doğrulanması:** Checklist'te öngörülen iki çift de görülüyor ama
**farklı SNR rejimlerinde**: Costas↔SteppedFH bir **-10 dB (gürültü) olgusu** (SteppedFH→Costas
%24, Costas→SteppedFH %23), LFM↔NLFM ise bir **yapısal (-8/-6 dB) olgu**. Yani "düşük SNR'da model
neyi karıştırır" sorusunun cevabı SNR-bağımlı: önce gürültü-çekim, sonra aile-içi benzerlik.

---

## 5. Sıradaki Adımlar
- [ ] Ana figür ve tabloyu makale Results bölümüne yerleştir; caption'ları yaz. (Not: henüz `paper/`
      dizini yok — makale iskeleti oluşturulmalı.)
- [x] LFM↔NLFM ve Costas↔SteppedFH karışımlarını düşük SNR confusion matrix'leri üzerinden incele.
      → §4 (WVD için tamamlandı).
- [ ] STFT/CWD için per-SNR confusion: checkpoint'ler silinmiş; ya yeniden eğit ya da
      eğitim çıktısı olarak preds'i kalıcı sakla. (§4 kısıtı.)
- [ ] Parametre/FLOP tablosu ekle (§3.4 kaynak-verimlilik argümanını sayısallaştır).
- [ ] (Opsiyonel) İstatistiksel anlamlılık: tekrarlı seed veya bootstrap CI ile TF farklarının
      güven aralığı.
