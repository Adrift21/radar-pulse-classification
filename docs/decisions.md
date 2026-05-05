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
- [x] ~~Costas dizi uzunluğu~~ → **N ∈ {5,6,7,8}, her N için 2 dizi, Δf rastgele [2,5] MHz** (2026-05-04)
- [x] ~~Frekans aralığı (carrier)~~ → **Complex baseband, fc ∈ [1, 20] MHz, %5 guard band** (config'de tanımlı, 2026-05-04)
- [x] ~~AWGN'in eklendiği nokta~~ → **Padding sonrası tam frame'e, SNR aktif bölge gücüne göre** (2026-05-04)