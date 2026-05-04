# Project Decisions Log — Radar Pulse Classification

Bu dosya proje boyunca alınan teknik ve stratejik kararları kayıt altına alır. Her giriş şu yapıyı izler:

```
## YYYY-MM-DD — [Karar Başlığı]
- **Karar:** Ne yapıldı/yapılacak
- **Gerekçe:** Neden bu seçildi
- **Alternatifler:** Ne göz önünde bulunduruldu, neden elendi
- **Sonuç/Etki:** Bu karar neyi etkiliyor
```

> **Not:** Bu dosya makalenin "Methods" ve "Implementation Details" bölümleri için doğrudan kaynak olacak. Detaylı tut.

---

## 2026-05-03 — Proje Konusu ve Akademik Çerçeve

- **Karar:** Düşük SNR koşullarında derin öğrenme tabanlı radar darbe sinyali sınıflandırması. Spektrogram, CWD, WVD gösterimlerinin karşılaştırılması.
- **Gerekçe:** Literatürde çoğu çalışma yüksek SNR'da yüksek doğruluk gösteriyor; gerçek elektronik harp ortamında düşük SNR koşulları hakim. Gürültü dayanıklılığı analizi, makaleyi "yayınlanabilir katkı" seviyesine taşıyor.
- **Alternatifler:** Otonom sürü simülasyonu, balistik yörünge optimizasyonu, görüntü işleme tabanlı hedef tespiti, uydu görüntülerinden stratejik analiz, şifreli taktik mesajlaşma.
- **Sonuç/Etki:** Tüm modül seçimleri ve teknoloji yığını bu karara göre şekillendi.

---

## 2026-05-03 — Hedef Sınıf Sayısı: 8

- **Karar:** 8 farklı radar sinyal sınıfı kullanılacak (LFM, NLFM, Barker, Frank, polifaz P1-P4, Costas, CW, FH/Stepped).
- **Gerekçe:** Akademik literatürdeki standart aralık 6-12 sınıf. 8 sınıf hem yeterli çeşitlilik sağlar hem de RTX 3050 + 4 GB VRAM ile makul sürede eğitilebilir.
- **Alternatifler:** 4-5 sınıf (yetersiz), 12+ sınıf (eğitim süresi ve VRAM zorlu).
- **Sonuç/Etki:** Veri üretim modülünün kapsamı, model output dimension'ları, confusion matrix boyutu.

---

## 2026-05-03 — SNR Aralığı: -10 dB ile +20 dB

- **Karar:** Eğitim ve test için SNR aralığı -10 dB ile +20 dB arası (genelde 2 dB adımlarla).
- **Gerekçe:** Gerçek EH ortamlarında alt sınır ~ -10 dB civarı; +20 dB üst sınır temiz sinyal davranışını yansıtır. Bu aralık literatürde en yaygın referans.
- **Alternatifler:** -20 dB başlangıç (çok aşırı, anlamsız sonuçlar), 0 dB başlangıç (gerçekçi değil).
- **Sonuç/Etki:** Veri üretim parametrelerinin temeli; SNR robustness eğrilerinin x-ekseni.

---

## 2026-05-03 — Geliştirme Ortamı: Hibrit (Lokal + Cloud)

- **Karar:** Lokal makinede kod geliştirme + MATLAB veri üretimi; Kaggle'da ResNet eğitimi; Colab Pro'da ViT/Swin eğitimi.
- **Gerekçe:** RTX 3050 Laptop GPU (4 GB VRAM) prototipleme için yeterli ama büyük modeller için kısıtlı. Kaggle bedava T4 sunuyor, Colab Pro ucuz A100 erişimi sağlıyor. AWS bilinçli olarak dışlandı (saatlik maliyet, akademik bütçe).
- **Alternatifler:** Tamamen lokal (yavaş, ViT eğitilemez), tamamen AWS (pahalı), tek bulut sağlayıcı (esnek değil).
- **Sonuç/Etki:** Veri formatı (Kaggle/Colab'a yüklenebilir boyutta tutulmalı), config yönetimi (bulut/lokal ayrımı).

---

## 2026-05-03 — Python 3.11.9 (3.14 yerine)

- **Karar:** Proje Python 3.11.9 venv'i üzerinde çalışacak.
- **Gerekçe:** Sistemde Python 3.14.3 vardı, ancak PyTorch ve `tftb` gibi kritik kütüphaneler henüz 3.14 için resmi wheel yayınlamadı. 3.11 sweet spot: tüm bilimsel kütüphaneler tam destekliyor.
- **Alternatifler:** Python 3.10 (eski), 3.12 (çoğu kütüphane destekliyor ama 3.11 daha test edilmiş), 3.14 (uyumsuzluk riski).
- **Sonuç/Etki:** `requirements.txt` 3.11 hedeflenerek pin'lendi. CI/CD ileride kurulursa Python 3.11 image'ı kullanılacak.

---

## 2026-05-03 — Deep Learning Framework: PyTorch (TensorFlow yerine)

- **Karar:** PyTorch 2.6.0 + CUDA 12.4 (cu124 wheel).
- **Gerekçe:** Akademik araştırmada PyTorch hakim, `timm` kütüphanesi ile ViT/Swin gibi modern mimarilere kolay erişim, `grad-cam` paketi PyTorch için olgun. RTX 3050 driver'ı CUDA 12.5 destekliyor → cu124 ileri uyumlu.
- **Alternatifler:** TensorFlow/Keras (deployment için iyi ama araştırma topluluğu PyTorch'a kaydı), JAX (henüz olgun değil).
- **Sonuç/Etki:** Tüm model implementasyonları PyTorch syntax'ı kullanacak; ONNX export ile gerekirse deployment yapılabilir.

---

## 2026-05-03 — Repository Visibility: Private Başla, Yayın Sonrası Public

- **Karar:** GitHub repo private olarak oluşturuldu, makale yayınlandıktan sonra public yapılacak.
- **Gerekçe:** Bazı dergiler kod public ise "prior publication" sayabiliyor; submission sürecinde anonimlik gereksinimleri olabilir.
- **Alternatifler:** Baştan public (akademik şeffaflık ama submission riski), hep private (akademik etki yok).
- **Sonuç/Etki:** Repo URL'i CV'de henüz görünmeyecek; arXiv preprint yüklendikten sonra public yapılıp link paylaşılabilir.

---

## 2026-05-03 — Lisans: MIT

- **Karar:** MIT License.
- **Gerekçe:** Akademik dünyanın de facto standardı; en serbest izin koşulları; başkalarının kodu kullanması için engel yok.
- **Alternatifler:** Apache 2.0 (patent koruması var, ama radar pulse classification patentlenebilir bir şey değil), GPL (kopyala-yapıştır kısıtlı, akademide tercih edilmez).
- **Sonuç/Etki:** Kod istenirse şirketler tarafından da kullanılabilir; atıf etiketi (citation) README'de zorunlu kılındı.

---

## 2026-05-04 — Sample Rate: 100 MHz

- **Karar:** Sentetik radar darbe sinyalleri 100 MHz örnekleme oranında üretilecek (Ts = 10 ns).
- **Gerekçe:** Literatürdeki de facto standart; tipik chirp bandwidth aralığımız (5-20 MHz) için Nyquist'in çok üzerinde marj sağlar. 200 MHz seçeneği sinyal vektörünü 2× büyütür, WVD/CWD hesabını ~4× yavaşlatır (O(N²) bileşen) ve 4 GB VRAM'de batch size'ı kısıtlar.
- **Alternatifler:** 200 MHz (yüksek bandwidth LFM için anlamlı ama bizim senaryoda gereksiz), 50 MHz (bazı dar bantlı sinyaller için yeterli ama LFM bandwidth marjı dar).
- **Sonuç/Etki:** Tüm MATLAB sinyal üretim fonksiyonlarının `fs` parametresi 100e6. TF gösterim boyutları, FFT noktaları bu temel üzerine kurulacak.

---

## 2026-05-04 — Pulse Width: 1-20 µs, Sabit Sinyal Uzunluğu 2048 Örnek

- **Karar:** Pulse width [1 µs, 20 µs] aralığında uniform rastgele seçilecek. Tüm sinyaller sabit **2048 örnek** uzunluğa zero-padding ile getirilecek (≈20.48 µs @ 100 MHz).
- **Gerekçe:** 2048 = 2^11, FFT-dostu. 1-20 µs aralığı tipik EH senaryolarını kapsar (search radar 1-100 µs, tracking 0.1-1 µs, pulse compression 10-50 µs). Sabit uzunluk batch processing için zorunlu; padding pozisyonu da rastgeleleştirilebilir (data augmentation).
- **Alternatifler:** Variable length + RNN (karmaşık), 1024 örnek (uzun darbeler kesilir), 4096 örnek (gereksiz büyük).
- **Sonuç/Etki:** MATLAB üretim fonksiyonlarında `pulse_width = rand_in([1e-6, 20e-6])`, sonra `signal = [zeros, pulse, zeros]` ile 2048'e tamamlanacak. Padding stratejisi (sol/sağ/ortalanmış/rastgele) Modül A başında netleşecek.

---

## 2026-05-04 — Sınıf Başına 5000 Örnek, SNR Rastgele Atanacak

- **Karar:** Her sınıftan 5000 örnek üretilecek (toplam 40.000). Her örneğe SNR değeri `[-10, -8, ..., +20] dB` setinden rastgele atanacak (her örnek tek bir SNR'de).
- **Gerekçe:** 5000 örnek RTX 3050'de eğitilebilir, Kaggle dataset limitine (20 GB free) sığar. SNR'i örnek başına rastgele atamak, "her örneği her SNR'de üret" yaklaşımına göre çok daha az veri ile eşdeğer genelleme verir (continuous augmentation mantığı). Test/eval'de SNR-stratified analiz için her SNR seviyesinde yeterli (≈ 5000/16 ≈ 312 örnek/sınıf/SNR) örnek garantilenir.
- **Alternatifler:** 10000/sınıf (80.000 toplam → ~24 GB ham TF görüntü, Kaggle limiti zorlanır), her örneği her SNR'de üret (16× veri patlaması).
- **Sonuç/Etki:** Veri üretim döngüsü iki katmanlı: (1) sınıf başına 5000 temiz sinyal üret, (2) her birine `randi([1,16])` ile SNR seçilip AWGN eklenir.

---

## 2026-05-04 — SNR Adımı: 2 dB (16 Nokta)

- **Karar:** SNR aralığı `[-10, -8, -6, ..., +18, +20]` dB → 16 ayrık nokta.
- **Gerekçe:** Literatürde 2 dB de facto standart (örn. Wei et al. 2019, Liu & Zhang 2020). 1 dB adımı 31 nokta verir → robustness eğrisi daha pürüzsüz ama hesap maliyeti ve veri yönetimi yarı yarıya artar; bilgi kazanımı marjinal.
- **Alternatifler:** 1 dB (gereksiz ince), 5 dB (eğri pürüzlü), 3 dB (literatürle uyumsuz).
- **Sonuç/Etki:** SNR robustness eğrilerinin x-ekseni bu 16 nokta. Confusion matrix'ler SNR-stratified olarak da hesaplanabilir.

---

## 2026-05-04 — Train/Val/Test Bölünmesi: 70/15/15

- **Karar:** Veri seti 70% train / 15% validation / 15% test olarak bölünecek. Bölünme sınıf-stratified ve SNR-stratified yapılacak.
- **Gerekçe:** 40.000 örnek için 28k/6k/6k. Val 6000 ile early stopping ve hyperparameter tuning güvenli; ViT gibi büyük modellerde val loss gürültüsü yönetilebilir. Test 6000 → her sınıfta ~750 örnek, güvenilir confusion matrix.
- **Alternatifler:** 80/10/10 (val 4000'e düşer, ViT için riskli), 60/20/20 (train daraldıkça large model performansı düşer), k-fold CV (akademik olarak güzel ama 9 deney × k-fold = pratik değil).
- **Sonuç/Etki:** Bölünme MATLAB tarafında değil Python tarafında yapılacak (esneklik için). `sklearn.model_selection.train_test_split` ile `stratify=labels` kullanılacak; SNR dengesini de korumak için iki aşamalı stratification.

---

## 2026-05-04 — Class Balance: Tam Dengeli

- **Karar:** Her sınıftan tam olarak 5000 örnek (perfect balance).
- **Gerekçe:** Akademik karşılaştırma için kontrollü deney ortamı zorunlu. Class imbalance ayrı bir araştırma sorusu (re-sampling, focal loss, class-weighted CE) ve makaleyi dağıtır. Reviewer'lar baseline olarak dengeli sınıflar bekler. Real-world imbalance ileride ek bir bölüm olarak eklenebilir, ama bu opsiyonel.
- **Alternatifler:** Gerçekçi orantısız (örn. CW daha sık) — gerçek operasyonel veriye benzer ama akademik baseline'ı zorlaştırır.
- **Sonuç/Etki:** Loss fonksiyonu standart cross-entropy (ağırlıksız). Confusion matrix doğrudan yorumlanabilir.

---

## 2026-05-04 — Dosya Formatı: HDF5 (.h5)

- **Karar:** Sentetik veri tek HDF5 dosyasında saklanacak. MATLAB tarafında `save('-v7.3')` (otomatik HDF5), Python tarafında `h5py` veya `mat73` ile okuma.
- **Gerekçe:** Hierarchical yapı (signals/labels/snr/metadata aynı dosyada), chunking + compression desteği, MATLAB v7.3 zaten HDF5 → çift sistem doğal köprülenir. `.npy` tek array sınırlı, `.mat` (eski sürüm) 2 GB sınırı var, çoklu küçük dosya filesystem overhead yaratır.
- **Alternatifler:** `.npy/.npz` (metadata yönetimi zor), `.mat` v7 (2 GB limit), `.parquet` (tabular, sinyal için doğal değil), `.tfrecord` (TF özel).
- **Sonuç/Etki:** Dosya yapısı:
  ```
  dataset.h5
  ├── /signals       (N, 2048) complex64
  ├── /labels        (N,) uint8
  ├── /snr_db        (N,) float32
  ├── /pulse_widths  (N,) float32   # µs cinsinden
  ├── /class_names   (8,) string
  └── /metadata      (sample_rate, generation_date, seed, version)
  ```
  Tahmini boyut: 40.000 × 2048 × 8 byte (complex64) ≈ 655 MB ham + metadata. Compression ile ~300-400 MB.

---

## 2026-05-04 — Random Seed: Global 42, Katmanlı

- **Karar:** Master seed = 42, modül başına alt-seed yönetimi:
  - MATLAB veri üretimi: `rng(42, 'twister')`
  - NumPy: `np.random.seed(42)`
  - PyTorch: `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`
  - cuDNN: `torch.backends.cudnn.deterministic = True` + `benchmark = False`
  - DataLoader: `worker_init_fn` ile her worker'a `seed + worker_id`
- **Gerekçe:** Tam reproducibility akademik makale için kritik. Reviewer "kodu çalıştırdım, sayılar tutmadı" derse problem. Katmanlı yapı, bir modülün seed'inin diğerini etkilememesini sağlar.
- **Alternatifler:** Random seed (reproducibility yok), tek global seed (modüller arası leak), seed yok (kabul edilemez).
- **Sonuç/Etki:** `configs/seed.yaml` dosyası oluşturulacak; tüm script'ler buradan okuyacak. cuDNN deterministik mod %5-10 yavaşlama getirir; kabul edilebilir trade-off.

---

## 2026-05-04 — NLFM Varyantları: Quadratic + Sinusoidal Karışımı

- **Karar:** NLFM sınıfı tek bir matematiksel form değil, iki varyanttan rastgele seçilecek:
  - **Quadratic NLFM** (%60): $f(t) = f_0 + k_1 t + k_2 t^2$
  - **Sinusoidal NLFM** (%40): $f(t) = f_c + (B/2) \cdot \sin(2\pi t / T)$
- **Gerekçe:** Tek bir varyant kullanmak modeli matematiksel forma overfit edebilir. NLFM literatüründe çok sayıda alt-varyant var; CNN'in "lineer olmayan FM ailesi" örüntüsünü öğrenmesi için iki farklı eğri tipi yeterli çeşitlilik sağlar. Quadratic en yaygın baseline; sinusoidal görsel olarak çok farklı (TF'de S-eğrisi vs parabol). Karışım ayrıca gerçek EH koşullarına daha yakın — rakip emitter'lar farklı NLFM şekilleri kullanır.
- **Alternatifler:** Sadece quadratic (basit ama monolitik), 4+ varyant (cubic, tangent/Price, polynomial) — başlangıçta gereksiz karmaşıklık; eğitim sonrası NLFM accuracy düşükse genişletilir. Variant'ı alt sınıf olarak ayırmak — 8 sınıf hedefini bozar.
- **Sonuç/Etki:** `generate_nlfm.m` her çağrıda rastgele varyant seçer, kullanılan varyant `params.variant` alanına yazılır. Confusion matrix üretimi sırasında alt-varyant breakdown opsiyonel olarak çıkarılabilir (debug için).

---

## YYYY-MM-DD — [Sonraki Karar Buraya]

<!-- Şablon:
- **Karar:**
- **Gerekçe:**
- **Alternatifler:**
- **Sonuç/Etki:**
-->

---

## ❓ Açık Sorular (Modül A İlerlerken Karar Verilecek)

- [x] ~~Sample rate: 100 MHz mi 200 MHz mi?~~ → **100 MHz** (2026-05-04)
- [x] ~~Pulse width aralığı~~ → **1-20 µs, 2048 örnek sabit uzunluk** (2026-05-04)
- [x] ~~Sınıf başına örnek sayısı~~ → **5000, SNR rastgele** (2026-05-04)
- [x] ~~SNR adım büyüklüğü~~ → **2 dB, 16 nokta** (2026-05-04)
- [x] ~~Train/Val/Test bölünmesi~~ → **70/15/15** (2026-05-04)
- [x] ~~Class balance~~ → **Tam dengeli** (2026-05-04)
- [x] ~~Dosya formatı~~ → **HDF5 (.h5)** (2026-05-04)
- [x] ~~Random seed yönetimi~~ → **Global 42, katmanlı** (2026-05-04)

### Modül A için Yeni Açık Sorular:
- [x] ~~Padding stratejisi~~ → **Random** (LFM testinde doğrulandı, 2026-05-04)
- [ ] **P1-P4 sınıfı:** Tek birleşik sınıf mı, 4 ayrı alt sınıf mı? (8 sınıf hedefini etkiler)
- [ ] **Barker kod uzunlukları:** Sadece B13 mü, yoksa B7+B11+B13 karışımı mı?
- [ ] **Costas dizi uzunluğu:** Sabit (örn. N=7) mi, değişken mi?
- [x] ~~Frekans aralığı (carrier)~~ → **Complex baseband, fc ∈ [1, 20] MHz, %5 guard band** (config'de tanımlı, 2026-05-04)
- [x] ~~AWGN'in eklendiği nokta~~ → **Padding sonrası tam frame'e, SNR aktif bölge gücüne göre** (2026-05-04)