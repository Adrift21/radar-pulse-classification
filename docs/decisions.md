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

## 2026-05-04 — Barker Kod Karışımı: B7 + B11 + B13, Rectangular Chip

- **Karar:** Barker sınıfı için her örnekte rastgele B7, B11 veya B13 kodu seçilecek (eşit olasılıkla, %33-%33-%33). Chip şekli rectangular (filtresiz, anlık faz atlamaları). Chip süresi $T_c = T/N$ (önce pulse width $T$ seçilir, kod uzunluğu $N$'e bölünür).
- **Gerekçe:** B7/B11/B13 üçlüsü literatürde "the Barker codes" olarak bilinen kanonik küme; B2-B5 çok kısa ve gerçek sistemlerde nadir. Eşit olasılık dağılımı modelin tek bir kod uzunluğuna overfit olmasını engeller. Rectangular chip akademik baseline; sınıflar arası adillik için tercih edildi (LFM/NLFM'de de transmitter filter modellenmedi). $T_c = T/N$ ilişkisi `cfg.pulse_width_s` aralığını LFM/NLFM ile tutarlı tutar.
- **Alternatifler:** Sadece B13 (literatür yaygın ama tek-noktalı), tüm Barker kodları B2-B13 (akademik tamlık ama B2-B5 gerçekçi değil), raised-cosine veya Gaussian filtered chip (gerçekçi ama "hangi filter, neden?" sorularını açar), $T_c$ önce seçip $T = N \cdot T_c$ (pulse width dağılımı koda bağımlı olur, confusion matrix'te yan etki).
- **Sonuç/Etki:** `generate_barker.m` her çağrıda rastgele kod seçer, kullanılan kod `params.code_name` (B7/B11/B13) alanına yazılır. Modül B'de TF gösterim çözünürlüğü Barker'ın CW'den ayrılması için kritik — chip geçişlerinin spektrumda görünür olması gerekir.

---

## 2026-05-04 — Frank Polyphase: N ∈ {4,6,8}, fc=0, Genişletilmiş Pulse Width

- **Karar:** Frank polyphase kod sınıfı için üç parametre:
  - **Matris boyutu N**: {4, 6, 8} kümesinden eşit olasılıkla rastgele (toplam chip sayısı $N^2$ → 16/36/64)
  - **Carrier frekansı**: $f_c = 0$ (saf complex baseband, ekstra carrier modülasyonu yok)
  - **Pulse width**: Frank'a özel [4, 20] µs aralığı (LFM/NLFM/Barker [1, 20] korunur)
- **Gerekçe:** N karışımı Barker'daki 3-kod yaklaşımı ile **tutarlı**; modelin tek bir matris boyutuna overfit olmasını engeller. N=5,7 (asal) literatürde nadir, eklenmedi. fc=0 seçimi: Frank'ın karakteristik TF imzası faz matrisinden gelir, carrier eklemek imzayı maskeler ve sınıflar arası ayırt ediciliği azaltır. Genişletilmiş pulse width ($T \geq 4$ µs): N=8 (64 chip) ile T=1 µs olsaydı $T_c = 15.6$ ns → 100 MHz örneklemede chip başına sadece 1.56 örnek olurdu (yetersiz). T=4 µs alt sınırı en kötü durumda chip başına 6.25 örnek garantiler.
- **Alternatifler:** Sabit N=8 (en yaygın baseline ama monolitik), N ∈ {4,5,6,7,8} (asal değerler eklenir, Frank için literatürde nadir), fc rastgele (Barker'daki gibi, ama imza maskeleme problemi), [1,20] µs aralığı (chip undersampling sorunu).
- **Sonuç/Etki:** `cfg.frank_pulse_width_s = [4e-6, 20e-6]` yeni bir config alanı. Diğer sınıflar `cfg.pulse_width_s`'i kullanmaya devam eder. `generate_frank.m` her çağrıda rastgele N seçer; `params.N`, `params.num_chips`, `params.phase_matrix` alanlarına yazılır. Frank-özel pulse width nedeniyle confusion matrix'te Frank örneklerinin pulse width dağılımı diğer sınıflardan biraz farklı olacak — bu kabul edilebilir çünkü model sınıflandırma yapıyor, pulse width regresyonu değil.

---

## 2026-05-04 — P1-P4 Polyphase: Tüm Alt-Kodlar, fc=0, Ortak Pulse Width

- **Karar:** Polyphase sınıfı için dört alt-kod eşit olasılıkla rastgele seçilecek (her biri %25):
  - **P1 (Lewis-Kretschmer):** $\phi_{m,n} = -\frac{\pi}{N}(N - (2n-1))((n-1)N + (m-1))$
  - **P2 (Lewis-Kretschmer, palindromik):** $\phi_{m,n} = -\frac{\pi}{2N}(2m-1-N)(2n-1-N)$
  - **P3 (LFM yaklaşımı):** $\phi_k = \frac{\pi (k-1)^2}{N_c}, \, k=1..N_c$
  - **P4 (LFM yaklaşımı, lineer offset):** $\phi_k = \frac{\pi (k-1)^2}{N_c} - \pi(k-1)$
- N matris boyutu Frank ile aynı: {4, 6, 8} (P2 için çift olma şartı zaten sağlanıyor). $N_c = N^2$. Carrier $f_c = 0$. Pulse width Frank ile paylaşılır.
- **Konfigürasyon değişikliği:** `cfg.frank_pulse_width_s` → `cfg.polyphase_pulse_width_s` olarak yeniden adlandırılır (Frank dahil tüm polyphase ailesi paylaşır). `generate_frank.m` küçük bir güncelleme alır.
- **Gerekçe:** Dört alt-kod akademik tamlık için zorunlu — reviewer "neden P2 yok?" sorusunu kapatır. P3 ve P4 aslında **discretized LFM** karakterli (literatürde "stepped approximation of chirp"); bu, modelin "P3/P4 vs LFM" ayrımını öğrenmesini zorlaştırır → makalenin "fine-grained classification" hikâyesini güçlendirir. P1 ve P2 ise Frank'a yakın (matris tabanlı discrete fazlar), bu da "polyphase ailesi içinde Frank'tan ayırt etme" zorluğu yaratır. fc=0 Frank ile tutarlı (faz imzasını maskelememek için). Ortak pulse width aralığı [4,20] µs Polyphase ailesi için chip oversampling garanti eder.
- **Alternatifler:** Sadece P3+P4 (basit ama kapsayıcılık eksik), P1+P3+P4 (P2'yi atla, ama akademik tamlık zarar görür), P kodlarını ayrı sınıflar yap (8 sınıf hedefini bozar; orijinal plan birleşik), fc rastgele (Frank ile tutarsız).
- **Sonuç/Etki:** `generate_polyphase.m` her çağrıda rastgele alt-kod seçer; `params.subcode` (P1/P2/P3/P4), `params.N`, `params.num_chips` alanlarına yazılır. Confusion matrix'te alt-kod breakdown'u opsiyonel olarak çıkarılabilir (debug için). Toplam 8 sınıf hedefi korunur.

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
- [x] ~~P1-P4 sınıfı~~ → **Tek birleşik sınıf, P1+P2+P3+P4 karışımı (%25 eşit)** (2026-05-04)
- [x] ~~Barker kod uzunlukları~~ → **B7 + B11 + B13 karışımı, rectangular chip** (2026-05-04)
- [ ] **Costas dizi uzunluğu:** Sabit (örn. N=7) mi, değişken mi?
- [x] ~~Frekans aralığı (carrier)~~ → **Complex baseband, fc ∈ [1, 20] MHz, %5 guard band** (config'de tanımlı, 2026-05-04)
- [x] ~~AWGN'in eklendiği nokta~~ → **Padding sonrası tam frame'e, SNR aktif bölge gücüne göre** (2026-05-04)