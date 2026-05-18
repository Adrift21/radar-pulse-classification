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

## 2026-05-04 — Costas Frequency Hopping: N ∈ {5,6,7,8}, Sembolik Diziler, Simetrik Baseband

- **Karar:** Costas sınıfı için beş parametre:
  - **N (kod uzunluğu)**: {5, 6, 7, 8} kümesinden eşit olasılıkla rastgele seçilir
  - **Costas dizisi**: Her N için 2 önceden tanımlı kanonik dizi, çağrı başına rastgele seçilir (toplam 8 dizi):
    - N=5: `[3,1,4,2,5]`, `[2,4,1,5,3]`
    - N=6: `[4,1,6,3,5,2]`, `[5,2,1,3,6,4]`
    - N=7: `[4,7,1,6,5,2,3]`, `[3,2,5,7,4,1,6]`
    - N=8: `[3,5,8,7,2,1,4,6]`, `[5,7,2,8,3,1,4,6]`
  - **Frekans adımı Δf**: [2, 5] MHz aralığında uniform rastgele
  - **Pulse width**: Generic [1, 20] µs (LFM/NLFM/Barker ile ortak `cfg.pulse_width_s`)
  - **Frekans yerleşimi**: Simetrik baseband, $f_k = (\pi(k) - (N+1)/2) \cdot \Delta f$ → frekanslar 0 etrafında ortalanır
- **Gerekçe:** N ∈ {5,6,7,8} aralığı literatürde standart Costas radarı (N=3,4 çok kısa, N≥9 araştırma odaklı çeşitlilik fazla varyans yaratır). Her N için 2 dizi yeterli çeşitlilik sağlar; tüm bilinen Costas dizilerini (örn. N=7 için 200 dizi) kullanmak gereksiz, model "Costas örüntüsü" kavramını az örnekle de öğrenir. Δf [2,5] MHz aralığı: N=8 ile en kötü 40 MHz toplam bandwidth (100 MHz fs için %5 guard band'i koruyarak yeterli), N=5 ile en küçük 10 MHz (chip blokları görsel olarak ayırt edilebilir). Generic pulse width: N max 8 olduğundan T=1 µs ile bile $T_c$=125 ns → 12.5 örnek/chip, polyphase'in N²=64 chip'inden çok rahat. Simetrik baseband: Frank/Polyphase'deki fc=0 mantığı ile tutarlı, TF imzası net görünür.
- **Alternatifler:** Tek N (sabit N=7 yaygın ama monolitik), tüm bilinen Costas dizilerini kullan (gereksiz çeşitlilik), Welch-Costas otomatik üretim (N kümesi p-1 olanlarla sınırlı, esnek değil), tek frekans adımı Δf sabit (daha az çeşitlilik), asimetrik frekans yerleşimi $f_k = (\pi(k)-1) \cdot \Delta f$ (taban DC'de, simetrisiz).
- **Sonuç/Etki:** `generate_costas.m` her çağrıda rastgele N + dizi + Δf seçer; `params.N`, `params.sequence`, `params.delta_f_hz`, `params.frequencies_hz` alanlarına yazılır. TF gösterimde (Modül B) N adet kısa "blok" zaman-frekans düzleminde dağılır — Costas'ın imza işareti. Bu desen modelin diğer 5 sınıftan ayırt etmesi için zayıf görünebilir ama düşük SNR'de blokların tespiti zorlaşır → SNR robustness analizinde Costas'ın eğrisi farklı davranabilir, bu da makaleye bilgi katar.

---

## 2026-05-04 — CW (Continuous Wave): Generic Pulse Width, Rastgele fc, Rastgele Initial Phase

- **Karar:** CW sınıfı için üç parametre:
  - **Pulse width**: Generic `cfg.pulse_width_s` aralığı [1, 20] µs (LFM/NLFM/Barker ile aynı)
  - **Carrier frekansı $f_c$**: Rastgele $[-f_{max}, +f_{max}]$, $f_{max} = f_s/2 - 0.05 f_s$ (LFM ile aynı %5 guard band)
  - **Başlangıç fazı $\phi_0$**: Rastgele $U[0, 2\pi)$
  - Sinyal modeli: $s(t) = e^{j(2\pi f_c t + \phi_0)}$ (tek frekanslı complex baseband)
- **Gerekçe:** Generic pulse width: CW örnekleri sadece pulse width ile diğer sınıflardan ayırt edilmesin (adil karşılaştırma). Rastgele fc: Frank/Polyphase fc=0 kullanıyor; CW de fc=0 olsaydı N=4 Frank'ın ilk satırıyla DC civarında karışabilirdi. Geniş fc aralığı modelin "spektrogramda tek yatay çizgi" örüntüsünü frekans-bağımsız öğrenmesini sağlar. Rastgele $\phi_0$: gerçek transmitter'lar koherent değil; padding zaten rastgele konumda, faz da rastgele olunca model padding/phase ipuçlarına overfit olmaz.
- **Alternatifler:** [10,20] µs (CW genelde uzun ama diğer sınıflarla pulse width ipucu yaratır), tüm frame doluluğu (padding stratejisi tutarsızlığı), sabit fc=0 (Frank ile karışma riski), sabit $\phi_0=0$ (deterministik ama gerçekçi değil), dar fc aralığı (gereksiz kısıt).
- **Sonuç/Etki:** `generate_cw.m` en kısa generator (~50 satır). `params.f_carrier_hz` ve `params.initial_phase` alanlarına yazılır. TF imzası: spektrogramda **tek yatay parlak çizgi**, tüm darbe boyunca sabit frekans. Costas ile ayırt etme: Costas N farklı yatay seviyede kısa bloklar, CW tek seviye uzun bant. -10 dB'de tek çizgi gürültüye gömülür → SNR robustness eğrisinde CW yüksek SNR'da kolay, düşük SNR'da diğerleri kadar zor.

---

## 2026-05-04 — Stepped/FH (Stepped Frequency): Costas-Tutarlı Parametreler, Monotonik Sıralama

- **Karar:** Stepped Frequency sınıfı için altı parametre:
  - **N (chip sayısı)**: {5, 6, 7, 8} eşit olasılıkla (Costas ile aynı set)
  - **Yön (direction)**: %50 up, %50 down (LFM up/down ile tutarlı)
  - **Frekans adımı Δf**: Rastgele [2, 5] MHz (Costas ile aynı)
  - **Başlangıç frekansı $f_{start}$**: Rastgele, ancak son frekans $f_{start} + (N-1)\Delta f$ Nyquist guard band'i içinde kalacak şekilde
  - **Pulse width**: Generic [1, 20] µs (LFM/CW/Costas ile aynı)
  - **Phase continuity**: Costas'taki gibi chip geçişlerinde sürekli faz, sinc artifaktları önlenir
  - Sinyal modeli: $f_k = f_{start} + (k-1)\Delta f$ (up) veya $f_k = f_{start} - (k-1)\Delta f$ (down), $k=1..N$
- **Gerekçe:** Stepped frequency aslında **discretized LFM** karakterli (sürekli LFM yerine basamaklı yaklaşım). Costas ile aynı parametre seti hem ailesel tutarlılık hem TF imzasının doğrudan karşılaştırılabilirliği sağlar. **Costas ile farkı:** Costas frekansları rastgele permütasyondadır (dağınık bloklar), Stepped frekansları monotonik sıralı (basamak/merdiven). Bu farklılık modelin "Costas vs Stepped" ayrımını yapması için sınıflar arası net bir görsel ipucu sağlar. Rastgele $f_{start}$ + monotonik step → gerçek FH radar davranışına yakın (belirli RF bandını tarama). Phase continuity Costas ile mutlak tutarlılık için.
- **Alternatifler:** Sabit yön (sadece up — gerçekçi değil), simetrik baseband (Costas ile birebir aynı, Stepped'ı ayırt edici özelliği zayıflatır), discrete $f_{start}$ kümesi (esnek değil), naif phase switching (sinc artifaktları, görsel kalite düşer), N kümesini farklılaştırma (gereksiz tutarsızlık).
- **Sonuç/Etki:** `generate_stepped_fh.m` Costas generator'üne mimari olarak benzer ama frekans ataması farklı. `params.N`, `params.direction`, `params.delta_f_hz`, `params.f_start_hz`, `params.frequencies_hz` alanlarına yazılır. TF imzası: spektrogramda **monotonik basamak/merdiven** (yukarı veya aşağı). LFM ile karışma riski var (her ikisi de monoton frekans değişimi) — ama LFM sürekli, Stepped basamaklı; CNN/ViT bu farkı TF gösterimin çözünürlüğüne göre öğrenecek. Modül A'nın 8. ve son sınıfı; bu kararla Modül A açık soruları kapanır.

---

## 2026-05-04 — Ana Üretim Döngüsü: Sıralı, Per-Sample Seed, AWGN Ayrı, HDF5 Metadata

- **Karar:** `main_generate_dataset.m` ana döngüsü için beş tasarım kararı:
  1. **Üretim sırası**: Sıralı (önce 5000 LFM → 5000 NLFM → ... → 5000 Stepped). Sınıflar arası shuffle Python tarafında train/val/test split sırasında yapılacak.
  2. **Per-sample seed**: Her örnek üretilmeden önce `rng(master_seed + global_sample_idx, 'twister')` çağrılır. Her sinyal bağımsız reproducible — "sample #1234'ü tek başına yeniden üret" mümkün.
  3. **Progress reporting**: `cfg.progress_every = 500` ile her 500 örnekte ilerleme yazdır (sınıf adı, sample sayacı, tahmini kalan süre).
  4. **AWGN ayrı tutulur**: HDF5'e **temiz sinyaller + intended SNR değerleri** kaydedilir. Gerçek AWGN Python tarafında okuma sırasında eklenir. Avantaj: Aynı temiz sinyalden farklı epoch'larda farklı gürültü seedleri ile augmentation, Modül B/C esnek SNR stratejisi seçebilir, disk alanı azalmaz (temiz sinyal aynı boyut).
  5. **HDF5 dataset şeması**: Minimum + temel metadata.
  
  ```
  dataset.h5
  ├── /signals        (40000, 2048) complex64    [clean signals]
  ├── /labels         (40000,)      uint8        [class index 0-7]
  ├── /snr_db         (40000,)      float32      [intended SNR per sample]
  ├── /pulse_widths_us (40000,)     float32      [actual pulse width]
  ├── /class_names    (8,)          string       [class label strings]
  └── attributes:
      ├── sample_rate_hz       (100e6)
      ├── signal_length        (2048)
      ├── master_seed          (42)
      ├── generation_date      (ISO timestamp)
      ├── dataset_version      ('0.1.0')
      ├── samples_per_class    (5000)
      └── num_classes          (8)
  ```
- **Gerekçe:** Sıralı üretim debugging için kolay, ayrıca train/test split kütüphaneleri (sklearn) shuffle parametresiyle bu sorunu zaten çözüyor. Per-sample seed: akademik makalelerde reviewer "tek bir yanlış sınıflandırılan örneği inceleyelim" diyebilir, bu durumda örnek tek başına üretilebilmeli. AWGN ayrı tutmak: aynı temiz sinyalden farklı augmentation'lar üretmek modelin genelleşmesine yardım eder; ayrıca Modül B'de TF gösterimini hesaplamadan önce gürültüyü ekleyip eklememe esnekliği kalır. HDF5 metadata: temel attribute'lar yeterli, full per-sample params struct (random direction, B, fc, vb.) gereksiz çünkü per-sample seed ile üretici fonksiyon tekrar çağrılarak elde edilebilir — disk alanı verimli.
- **Alternatifler:** Karışık üretim (interleaved — implementasyon karmaşık, debugging zor), tek global seed (bağımsız örnek üretimi imkansız), AWGN ana döngüde (esnek değil), tüm params struct'ı kaydet (40000 × ~10 alan, GB'larca yer), her sınıfa ayrı dosya (split logic karmaşık), küçük chunk'lar halinde HDF5 yazma (fancy ama gereksiz).
- **Sonuç/Etki:** İki yeni dosya: `main_generate_dataset.m` (ana giriş noktası, ~80 satır) + `utils/save_dataset_h5.m` (HDF5 export, ~70 satır). Dataset üretimi tek seferlik ~5-15 dakika sürer (donanıma bağlı), `synthetic_samples/dataset.h5` ~280-400 MB civarında olur (40000 × 2048 × 8 byte complex64 + metadata, sıkıştırılmamış). Modül B Python tarafında `h5py` veya `mat73` ile okuyacak.

---

---

## 2026-05-17 — CWD Kütüphane Seçimi: Custom NumPy (tftb v0.2.0'da CWD Yok)

- **Karar:** Choi-Williams Distribution (CWD) için **custom NumPy implementasyonu** yazıldı (`preprocessing/time_frequency/cwd.py`). tftb kütüphanesi WVD için kullanılmaya devam edecek (Phase 2'nin sonraki adımı), ancak CWD için kullanılmıyor.
- **Gerekçe:** Phase 2 öncesinde tftb v0.2.0'ın `ChoiWilliamsDistribution` sınıfını içerdiği varsayılmıştı (web search ve eski sürüm dokümantasyonlarına dayanarak). Bugün doğrudan paket içeriği incelendi: `tftb.processing` `__init__.py`'sinde `ChoiWilliamsDistribution` yok, `cohen` alt-modülünde de yok. Yalnızca `linear.py`'de bir yorum/docstring referansı bulundu — implementasyon değil. Custom impl matematiksel olarak küçük (~200 satır), `tftb.processing.WignerVilleDistribution` ile sigma→∞ limitinde **Pearson = 1.0** korelasyon doğrulaması yapıldı (bit-for-bit aynı total energy: 7.117e+06, aynı max: 1023.0). Bu, kütüphane karşılaştırmasından daha güçlü bir matematiksel doğrulama sağlar. Akademik açıdan da defensible: "we implemented CWD from first principles based on Choi & Williams (1989), verified against tftb's WignerVilleDistribution in the σ→∞ limit" cümlesi reviewer için temiz argüman.
- **Alternatifler:** (1) tftb'yi fork'la, CWD'yi ekle — fazla maintenance yükü, akademik makale için gereksiz; (2) eski sürüm tftb (0.1.4) — Python 3.11 ve numpy 2.x uyumsuzluk riski, paket zaten "scikit-signal" organization'a taşındığında 0.1.x'ten kopmuş; (3) Başka bir kütüphane (`pytftb`, `tfa`) — daha az maintained, daha riskli; (4) Hibrit: tftb primary, custom referans — orijinal plandı, paket içeriği keşfedildikten sonra geçersiz oldu.
- **Sonuç/Etki:** `preprocessing/time_frequency/cwd.py` time-lag formülasyonunda CWD'yi hesaplar (Eq. 1-3 docstring'de açıklı). `requirements.txt`'teki `tftb>=0.1.4` constraint'i WVD için hâlâ gerekli, sadece minimum versiyon `>=0.2.0`'a yükseltilebilir. `scripts/smoke_test_tftb.py` görevini tamamladı (silinebilir veya `_archive/` altına taşınabilir). Makale "Methods" bölümünde CWD altında bu impl detayı 2-3 cümle ile geçecek.

---

## 2026-05-17 — CWD Sigma Parametresi: σ = 1.0

- **Karar:** Choi-Williams Distribution kerneli için **σ = 1.0** (Gaussian-product kernel parametresi). `cwd.py`'de `DEFAULT_SIGMA` sabiti olarak tanımlı.
- **Gerekçe:** σ Choi-Williams kernelinin tek tunable parametresi: σ küçük → güçlü cross-term süzme + auto-term blur, σ büyük → cross-term'ler süzülmez (WVD limitine yaklaşır). Bizim sinyallerimiz **mono-component** (her örnek tek bir radar darbe sınıfı), cross-term sorunu multi-component sinyallerden çok daha az. σ = 1.0 değeri Choi & Williams (1989) orijinal makalesindeki "balanced default" ve MATLAB TFTB'nin `tfrcw` fonksiyonundaki default ile aynı. Reviewer "neden bu değer?" sorusunda "Choi & Williams 1989, original recommendation" net cevap. 8-sınıf doğrulama figürü (`cwd_clean_8classes.png`) σ=1.0 ile üretildi ve **görsel olarak başarılı**: cross-term'ler -50 dB'nin altında, auto-term'ler keskin, hiçbir sınıf "siyah kutu" olarak görünmüyor.
- **Alternatifler:** σ = 0.1 (agresif suppression, multi-component sinyaller için, bizde gereksiz); σ = 3.0 veya σ = 10 (cross-term'ler yetersiz süzülür, WVD'ye yaklaşır → CWD'nin avantajı kaybolur); değişken σ (parameter sweep) — gereksiz deney maliyeti, akademik literatürde standart pratik tek değer kullanmak.
- **Sonuç/Etki:** `cwd.py` API'sı `sigma` parametresini opsiyonel argüman olarak alır, default 1.0. Makale "Methods" bölümünde "CWD kernel parameter σ = 1.0, following Choi & Williams (1989)" cümlesi yeterli. İleride düşük SNR'da CWD performansı beklenmedik şekilde düşerse (Modül D), σ değeri tekrar tartışmaya açılabilir.

---

## 2026-05-17 — CWD Downsampling: (256, 64) — Phase 1 STFT ile Uyumlu

- **Karar:** CWD çıktı boyutu **(n_freq=256, n_time=64)** olarak sabitlendi. Hesaplama parametreleri: `time_step=32` (decimation in time), `n_freq=256` (FFT noktası), `max_lag = n_freq // 2 - 1 = 127`.
- **Gerekçe:** Tam çözünürlük CWD (2048×2048) tek örnek başına 32 MB RAM tüketir; 5000 örnek × 8 sınıf = 1.28 TB pre-compute olur (kabul edilemez). On-the-fly hesaplama stratejimiz olsa bile (decisions.md 2026-05-05 AWGN girdisi), tam çözünürlük her DataLoader call başına 50+ saniye gerektirir — eğitim tıkanır. Downsampling sonrası (256, 64): tek örnek 64 KB, hesaplama ~33 ms. Phase 1 STFT (256, 57) ile **aynı çözünürlük kategorisinde**, böylece Modül C'de aynı CNN/ResNet/ViT mimarisi her iki gösterimi de aynı input shape ile kabul eder. Bu, "apples-to-apples" akademik karşılaştırma için kritik — modeller TF'nin **bilgi içeriğini** karşılaştırır, çözünürlük farkını değil. `max_lag = n_freq // 2 - 1` formülü Hermitian symmetry için pozitif ve negatif lag bölgelerinin ayrık kalmasını garanti eder (FFT buffer'da çakışma olmaz).
- **Alternatifler:** (1) Tam (2048, 2048) → RAM/hız sorunu; (2) 224×224 (model input ile aynı) → STFT (256, 57) ile inconsistency; (3) (512, 128) — orta yol, ama Phase 1 STFT ile uyumsuz; (4) Variable resolution per class — debugging zorlaşır, akademik karşılaştırma bozulur.
- **Sonuç/Etki:** `cwd.py` default parametreleri Modül A datasetiyle (`fs=100 MHz`, `N=2048`) optimize edildi. Modül C'de TF gösterimini 224×224'e çevirmek için Modül B Phase 3'te ortak bir `tf_to_image()` fonksiyonu yazılacak — bu fonksiyon STFT (256, 57), CWD (256, 64), WVD (256, ?) çıktılarını dB-scale + bilinear resize ile 224×224'e dönüştürür. 8-sınıf doğrulama figüründe çözünürlük yeterli: LFM diagonal, NLFM curve, Costas blokları, SteppedFH basamakları çıplak gözle ayırt edilebilir.

---

## 2026-05-17 — AWGN Runtime Fonksiyonu: Python NumPy, Active-Region Power, Explicit RNG

- **Karar:** `preprocessing/noise/awgn.py` yazıldı. Tek public fonksiyon `add_awgn(signal, snr_db, active_idx, rng)` — MATLAB tarafındaki `data_generation/matlab/utils/add_awgn.m`'in Python muadili. Sinyal gücü active region (non-zero pulse) üzerinden hesaplanır, AWGN complex (real + 1j*imag, her biri `N(0, noise_power/2)`), reproducibility için `rng` parametresi (numpy.random.Generator) zorunlu.
- **Gerekçe:** decisions.md 2026-05-05 "AWGN on-the-fly" kararı gereği gürültü pre-compute edilmiyor; her DataLoader call'da runtime'da ekleniyor. Active-region power kullanımı kritik: padding sıfırları sinyal gücünü düşürür, SNR target sapar (test 5 sayısal olarak gösterdi: `active_idx=None` ile 0 dB hedefte empirik +3 dB sapma). Explicit `rng`: DataLoader worker'larında her worker kendi seed'iyle çalışmak zorunda (decisions.md 2026-05-04 "Random Seed: Global 42, Katmanlı" gereği), `worker_init_fn` her worker'a kendi Generator'unu pass edecek. MATLAB ile birebir convention: `add_awgn.m`'in `randn(N,1) + 1j*randn(N,1)` davranışıyla uyumlu (total complex variance = `noise_power`, per-component variance = `noise_power/2`).
- **Alternatifler:** **(a)** Global `np.random` state — DataLoader worker'larında state paylaşılmadığı için reproducibility kırılır, akademik makale için kabul edilemez; **(b)** Active-region tespit otomatik (sıfır-eşik ile) — fonksiyon içinde değil caller'da, çünkü dataset zaten `start_idx`/`stop_idx`'i biliyor (HDF5'te yoksa, Modül A `params.start_idx` üretiyor); **(c)** Per-component variance = `noise_power` (her ikisi de tam variance) — MATLAB convention'undan sapar, reviewer kafa karışıklığı yaratır.
- **Sonuç/Etki:** ~120 satır impl + docstring. Birim testleri (50 deneme × 6 SNR seviyesi) target vs empirical SNR sapması ≤ 0.13 dB std. Reproducibility test: aynı seed → max abs diff = 0.00. Görsel test (`test_awgn_snr_sweep.py`): LFM idx=4314'te 6 SNR seviyesinde STFT magnitude figure üretildi, -10 dB'de diagonal zar zor seçiliyor (gürültü baskın), +20 dB'de temiz baseline — Modül D'nin "SNR robustness" hikâye anlatımının görsel temeli.
---

## 2026-05-17 — PyTorch RadarPulseDataset + DataLoader: On-the-Fly Pipeline, Per-Sample Seeded, Multi-Worker Safe

- **Karar:** `preprocessing/datasets/radar_pulse_dataset.py` yazıldı. `RadarPulseDataset(Dataset)` class'ı + `radar_pulse_worker_init(worker_id)` worker_init_fn. Pipeline her `__getitem__`'da: HDF5 read → AWGN at intended SNR → TF representation (STFT/CWD/WVD seçimi __init__'te) → tf_to_image → PyTorch (1, 224, 224) float32 tensor + int label. Per-sample seeding: `master_seed + global_sample_idx` her sample için unique ve reproducible. HDF5 file handle lazy açılır (h5py NOT fork-safe), worker_init_fn her worker'da reset eder.
- **Gerekçe:** Modül B Phase 1'deki "AWGN runtime, gürültü pre-compute değil" kararıyla tutarlı pipeline. Per-sample seeding: Modül D'nin SNR-stratified evaluation reproducibility için kritik — aynı (idx, master_seed) → aynı output. tf_repr `__init__`'te seçilir: 3 mimari × 3 gösterim = 9 deney matrisinde her deney kendi Dataset/Loader pair'ini kurar (memory verimli, kod temiz). Worker_init_fn'in iki rolü: (a) h5py file handle'ı her worker process'te tekrar açar (fork-safe), (b) numpy/torch RNG'lerini per-worker seed'ler (defansif).
- **Alternatifler:** **(a)** Per-epoch seed (augmentation) — şimdilik out, ileride `epoch` parametresi eklenebilir; **(b)** Single Dataset multi-TF — `__init__`'te tf_repr seçmek yerine `__getitem__`'da, esnek ama API karmaşık; **(c)** Eager HDF5 load tüm dataset'i RAM'a — 280 MB tutar, mümkün ama Kaggle/Colab kısıtları için lazy daha güvenli.
- **Sonuç/Etki:** ~280 satır impl. 12 birim/entegrasyon testten geçti (single-process, multi-process, reproducibility, subset, throughput). Test 10: multi-process output single-process ile **bit-for-bit aynı** (max abs diff = 0.00, labels match). Akademik açıdan: aynı (master_seed, indices) ile farklı eğitim run'ları arasında tam reproducibility garantili. Pylance 75 statik analiz uyarısı verir (h5py + Module callable yanlış-pozitif), runtime sorunsuz.

---

## 2026-05-17 — Modül B Phase 2b Throughput Benchmark + Modül C İçin DataLoader Config

- **Karar:** Modül C için **tüm 3 TF gösterimde** optimum DataLoader config: `num_workers=4, batch_size=64`. Phase 2b realistic benchmark (200 örnek class-balanced, 12 config: 3 TF × 2 workers × 2 batch_size) sonucu.
- **Gerekçe:** Senin RTX 3050 Laptop CPU'sunda ölçülen değerler:
    | Rep | best config | samples/sec | 50 epoch / 28k train tahmini |
    |---|---|---|---|
    | STFT | workers=4, batch=64 | 147.7 | 2.6 saat |
    | WVD | workers=4, batch=64 | 54.3 | 7.2 saat |
    | CWD | workers=4, batch=64 | 9.2 saat |
  
  Üç TF için de aynı config optimum — Modül C'de **tek hyperparameter setup üç eğitime de uyar**, akademik karşılaştırma için ideal. Multi-worker speedup tutarlı **2.2×** her TF için. Batch=64 her config'te batch=32'den hızlı (CPU verimliliği). Modül C tahmini: 9 deney (3 mimari × 3 TF) toplamı ~57 saat ≈ **2.4 gün lokal**. Phase 1'deki "Kaggle/Colab zorunlu" varsayımı revize edildi — RTX 3050 hesaplanabilir aralıkta.
- **Alternatifler:** Bigger batch (128, 256) — RTX 3050 4 GB VRAM kısıtı, GPU side test gerekir; num_workers=8 — laptop core sayısı tipik 4-8, marginal kazanç beklenir; pre-compute pipeline (Phase 1'de reddedildi) — disk 24 GB, augmentation kayıp, akademik dürüstlük azalır.
- **Sonuç/Etki:** Modül C planı revize edildi: STFT eğitimleri lokal (gece tek seferde), WVD eğitimleri lokal (gece tek seferde), CWD eğitimleri lokal veya Kaggle T4 GPU (Kaggle 3-4× hızlı tahmini). Modül C için DataLoader configs: `num_workers=4, batch_size=64, worker_init_fn=radar_pulse_worker_init`. Benchmark scripti `preprocessing/datasets/tests/benchmark_phase2b.py` ile her makinede tekrarlanabilir (akademik reproducibility).

## 2026-05-17 — TF-to-Image Transform: dB-Scale + Per-Sample Max-Normalize + 224×224 Resize, 1 Channel

- **Karar:** `preprocessing/transforms/tf_to_image.py` yazıldı. Tek public fonksiyon `tf_to_image(tf_magnitude, output_size=(224, 224), db_floor=-60.0, eps=1e-12, interpolation="linear")`. Pipeline: ham magnitude → dB (peak-relative) → clip [db_floor, 0] → linearly remap [0, 1] → cv2.INTER_LINEAR resize → float32 single-channel. Modül C için **STFT, CWD, WVD üçü de aynı (1, 224, 224) tensor üretir**.
- **Gerekçe:** **(a)** Per-sample max-normalize: AWGN gürültü patternlarına karşı dayanıklı (absolute power değil, peak'e göre rölatif), train/test tutarlılığı garantili. Global z-score alternatifi reddedildi çünkü AWGN per-sample mean/std değiştirir, dataset distribution shift'i yaratır. **(b)** Tek kanal: TF magnitude doğal olarak grayscale, 3 kanala kopyalamak gereksiz bellek. ResNet/ViT'in ilk conv layer'ı `Conv2d(1, 64, ...)` ile expand eder, pretrain için Modül C'de `mean=0.5, std=0.5` normalize eklenebilir. **(c)** 224×224: ResNet-50, ViT, Swin için standart, Modül C'de aynı mimariler 3 TF gösterimi için kullanılacak (9 deney matrisi). **(d)** db_floor=-60.0: Phase 1/2/2a görsel testlerinde kullandığımız değer, tutarlı sonuçlar. **(e)** cv2.INTER_LINEAR: time ekseninde upsample (57-64 → 224), freq ekseninde downsample (256 → 224), iki yön için de güvenli default.
- **Alternatifler:** **(B)** Global z-score normalization — AWGN ile distribution shift, reddedildi; **(C)** 3-kanal (R=G=B=magnitude veya STFT/CWD/WVD birlikte) — multi-representation fusion ayrı bir akademik soru, şimdilik out of scope; **(D)** cv2.INTER_AREA — sadece pure downsample için, mixed resize için INTER_LINEAR daha doğal.
- **Sonuç/Etki:** ~80 satır impl + docstring. 7 birim testten geçti (shape, range, dtype, zero input, reproducibility, dB clip, interpolation modes, custom output_size, error handling). Görsel doğrulama (`test_tf_to_image_visualization.py`) 3×8 grid figürü üretti — 8 sınıf × 3 TF gösterim. Bu figür **makale "Methods" bölümünün merkez parçası**: "model gerçekten ne görüyor" sorusuna çıplak gözle cevap. Per-representation aggregate stats (gerçek dataset): STFT (mean=0.12, std=0.20), CWD (mean=0.10, std=0.15), WVD (mean=0.24, std=0.16) — üç gösterim model'a farklı statistical structure sunuyor.
---

❓ Açık Sorular (Modül A İlerlerken Karar Verilecek)

 Sample rate: 100 MHz mi 200 MHz mi? → 100 MHz (2026-05-04)
 Pulse width aralığı → 1-20 µs, 2048 örnek sabit uzunluk (2026-05-04)
 Sınıf başına örnek sayısı → 5000, SNR rastgele (2026-05-04)
 SNR adım büyüklüğü → 2 dB, 16 nokta (2026-05-04)
 Train/Val/Test bölünmesi → 70/15/15 (2026-05-04)
 Class balance → Tam dengeli (2026-05-04)
 Dosya formatı → HDF5 (.h5) (2026-05-04)
 Random seed yönetimi → Global 42, katmanlı (2026-05-04)

Modül A için Açık Sorular (kapandı):

 Padding stratejisi → Random (LFM testinde doğrulandı, 2026-05-04)
 P1-P4 sınıfı → Tek birleşik sınıf, P1+P2+P3+P4 karışımı (%25 eşit) (2026-05-04)
 Barker kod uzunlukları → B7 + B11 + B13 karışımı, rectangular chip (2026-05-04)
 Costas dizi uzunluğu → N ∈ {5,6,7,8}, her N için 2 dizi, Δf rastgele [2,5] MHz (2026-05-04)
 Frekans aralığı (carrier) → Complex baseband, fc ∈ [1, 20] MHz, %5 guard band (config'de tanımlı, 2026-05-04)
 AWGN'in eklendiği nokta → Padding sonrası tam frame'e, SNR aktif bölge gücüne göre (2026-05-04)

Modül B için Açık Sorular:

- [x] ~~CWD parametreleri~~ → Custom NumPy impl, σ=1.0, (256, 64) downsampling (2026-05-17)
- [x] ~~WVD parametreleri~~ → Sigma=∞ via compute_cwd wrapper (2026-05-17)
- [x] ~~AWGN runtime fonksiyonu~~ → `preprocessing/noise/awgn.py` (2026-05-17)
- [x] ~~dB-scale + normalizasyon stratejisi~~ → Per-sample max-normalize + db_floor=-60 + cv2.INTER_LINEAR resize (2026-05-17)
- [x] ~~Görüntü dönüşümü 224×224, tek/üç kanal~~ → 224×224, 1 channel (2026-05-17)
- [x] ~~PyTorch Dataset/DataLoader mimarisi~~ → `RadarPulseDataset` + `radar_pulse_worker_init`, per-sample seed (2026-05-17)
- [x] ~~Mini batch hız benchmark~~ → workers=4, batch_size=64 optimal her TF için (2026-05-17)