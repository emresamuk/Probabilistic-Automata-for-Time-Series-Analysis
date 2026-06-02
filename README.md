# Probabilistic Automata for Time Series Analysis #

**Bölüm:** Bilişim Sistemleri Mühendisliği

**Ders:** Yazılım Laboratuvarı II - Proje 2  

**Ekip Üyeleri:** 
- Metehan Yüksek 241307137 
- Emre Beraat Samuk 241307136
  

---

## 1. Giriş ve Proje Özeti

Bu proje, endüstriyel zaman serisi verilerindeki (sensör sinyalleri, SCADA akışları vb.) anomalilerin tespiti ve sınıflandırılması amacıyla geliştirilmiş; sembolik/olasılıksal yaklaşımlar ile modern derin öğrenme mimarilerini bir arada barındıran **hibrit ve parametrik bir yazılım sistemidir**. 

Proje kapsamında, zaman serisi verilerini metinsel sembol dizilerine dönüştürerek durumlar arası geçiş olasılıklarını hesaplayan bir **Olasılıksal Otomata (Probabilistic Automata)** modeli sıfırdan inşa edilmiştir. Bu modelin performansı ve açıklanabilirliği; yüksek doğruluk potansiyeline sahip ancak yorumlanabilirliği sınırlı (black-box) olan **Uzun Kısa Süreli Bellek (LSTM)** ve **1D Evrişimli Sinir Ağları (1D-CNN)** modelleri ile çok yönlü olarak karşılaştırılmıştır. 

Sistem, literatürde kabul görmüş iki büyük veri seti olan **SKAB (Skoltech Anomaly Benchmark)** ve **BATADAL** üzerinde katı kurallarla test edilmiştir.

---

## 2. Yazılım Mimarisi ve Proje Organizasyonu

Proje, genişletilebilirlik ve temiz kod (Clean Code) prensiplerine tam uyum sağlamak adına nesne yönelimli, modüler ve pipeline (veri akış hattı) yapısında tasarlanmıştır. Tüm kritik hiper-parametreler kaynak kodundan izole edilerek merkezi bir konfigürasyon dosyasından yönetilmektedir.

### Klasör Hiyerarşisi
```yaml
Probabilistic-Automata-for-Time-Series-Analysis/
├── config/
│   └── settings.yaml              # Merkezi konfigürasyon yapısı (Hiper-parametreler ve veri yolları)
├── src/
│   ├── __init__.py                # Modül başlatıcı paket dosyası
│   ├── preprocessing.py           # Normalizasyon, Gürültü Ekleme, PAA ve SAX Dönüşüm Pipeline'ı
│   └── models.py                  # Olasılıksal Otomata, LSTM ve 1D-CNN Model Tanımları
├── tests/
│   └── test_unseen.py             # Levenshtein Mesafe Eşleme Mekanizması Birim Testi (Unit Test)
├── outputs/                       # Kod tarafından otomatik üretilen zorunlu akademik grafikler
│   ├── confusion_matrix.png       # LSTM Performans Hata Matrisi
│   ├── transition_heatmap.png     # Otomata Durum Geçiş Yoğunluk Haritası
│   └── param_sensitivity_plot.png # Parametre Duyarlılık ve Durum Patlaması Grafiği
├── main.py                        # Uçtan uca tüm pipeline akışını yöneten ana çalıştırıcı
├── README.md                      # Proje Raporu ve Dokümantasyonu
└── venv/                          # Bağımsız Python sanal bağımlılık ortamı
```

---

## 3.Veri Ön İşleme ve Sızıntı Önleme Pipeline'ı
Zaman serisi verilerinin gürültüden arındırılması ve sembolik otomata girdisine dönüştürülmesi üç aşamalı bir hiyerarşide gerçekleştirilmektedir:

**3.1 Normalizasyon ve Sızıntı Önleme:** 

Sensör verileri ölçek farklılıklarından arındırılmak üzere normalize edilmiştir. Veri sızıntısını (Data Leakage) engellemek adına ölçekleyiciler (StandardScaler/MinMaxScaler) yalnızca eğitim (train) verisi üzerinde fit edilmiş, validation ve test verilerine ise yalnızca transform katmanı olarak uygulanmıştır.

**3.2 Çok Değişkenli Veri İçin Boyut İndirgeme (PCA & PAA):**

Otomata tabanlı model yalnızca tek boyutlu veri ile çalışabildiği için, çok değişkenli sensör özellikleri PCA (Temel Bileşenler Analizi) ile en yüksek varyansı temsil eden ilk bileşene (PC1) indirgenmiştir. PCA dönüşümü de yalnızca train verisi üzerinde fit edilmiştir. Ardından sürekli zaman serisi verileri belirlenen kayan pencere (window size) genişliğinde alt parçalara bölünerek ortalamaları alınmış (PAA - Piecewise Aggregate Approximation) ve boyutsal karmaşıklık düşürülürken zamansal sıra korunmuştur.

**3.3 SAX (Symbolic Aggregate Approximation) Sembolizasyonu:**

PAA çıktısı olan sürekli değerler, Gauss dağılımı temel alınarak kesim noktalarından metinsel sembol harflerine (örn: 'a', 'b', 'c') eşlenmiştir. Kayan pencere adımları (Sliding Window) kullanılarak benzersiz örüntüler (pattern) çıkarılmış ve her örüntü birer Durum (State) olarak tanımlanmıştır.

**Veri Bölme Stratejileri:**
- **SKAB Veri Seti:**
  
   source_file sütunu grup değişkeni olarak kullanılmış ve dosya bazlı GroupKFold stratejisi uygulanmıştır. Aynı .csv dosyasına ait kayıtların hem eğitim hem test kümesinde aynı anda yer alması engellenmiştir.
- **BATADAL Veri Seti:**
  
   Zamansal bağımlılıkların korunması amacıyla satır bazlı rastgele bölme yapılmamış; veri kronolojik sırasına göre tam olarak %60 Eğitim, %20 Doğrulama (Validation) ve %20 Test olarak ayrılmıştır.

---

## 4. Olasılıksal Açıklanabilirlik Modülü ve Güven Skoru ##

Olasılıksal otomata modeli, durumlar arası geçiş olasılıklarını frekans tabanlı olarak öğrenir:

$$P(S_i \rightarrow S_j) = \frac{\text{Geçiş Sayısı}}{\text{Toplam Çıkış Sayısı}}$$

Ardışık geçiş olasılıklarının çarpımı ile tüm örüntü dizisinin toplam olasılığı hesaplanır. Düşük olasılığa sahip diziler model tarafından beklenmeyen davranışlar olarak işaretlenir.

**4.1 Sıfır Olasılık Problemi ve Laplace / Frekans Smoothing**

Eğitim verisinde sıklık matrisinde yer almayan ancak test esnasında ortaya çıkan nadir geçişlerin tüm çarpımı sıfırlayarak sistemi çökertmesini engellemek adına, modele Frekans Smoothing (Yumuşatma) katmanı entegre edilmiştir. Bilinmeyen geçişlere taban bir alfa olasılığı ($0.0001$) atanarak sistemin matematiksel kararlılığı korunmuştur.

**4.2 Güven Skoru Tanımı ve Yorumu**

Yol uzunluğu ($N$) arttıkça, ardışık olasılık çarpımları doğası gereği küçülmektedir. Bu boyutsal etkiyi normalize etmek adına, yol uzunluğuna bağlı geometrik ortalama algoritması kullanılarak $0$ ile $1$ arasında anlamlı bir Güven Skoru formüle edilmiştir:

$$\text{Confidence Score} = (P(\text{sequence}))^{\frac{1}{N}}$$

```json
{
    "time_step": 1,
    "state": "cccc",
    "status": "seen",
    "probability": 0.9753154141524959,
    "confidence_score": 0.9753154141524959,
    "decision": "normal"
}
```
Test aşamasında model, gelen cccc durumunu daha önce eğitim sözlüğünde gördüğü için seen olarak işaretlemiştir. Durum geçiş olasılığı $0.9753$ gibi yüksek bir frekansta gerçekleştiği için sistem karara olan Güven Skorunu en üst düzeyde sunmuş ve ilgili zaman adımını rasyonel olarak "normal operasyon" olarak sınıflandırmıştır.

---

## 5.Unseen Pattern Yönetimi ve Birim Testleri ##

Test esnasında eğitim verisindeki SAX sözlüğünde hiç yer almayan, tamamen yabancı ve bilinmeyen bir durum dizisiyle karşılaşıldığında sistemin kilitlenmesini önlemek adına bir tolerans mekanizması geliştirilmiştir.

Sözlük dışı kalan yabancı dizi, bilinen durum havuzundaki tüm elemanlarla kıyaslanarak iki string arasındaki minimum değişim maliyetini hesaplayan Levenshtein Algoritmasına tabi tutulur. Sistem bilinmeyen diziyi, aradaki edit mesafesi en küçük olan en yakın geçerli duruma projekte ederek akışı kesintisiz bir şekilde devam ettirir.

Bu mekanizmanın deterministik ve yeniden üretilebilir olduğu, tests/test_unseen.py dosyası altında yazılan Birim Testler ile doğrulanmıştır. Yapılan birim test senaryosunda, sözlükte olmayan axc girdisinin, en yakın bilinen durum olan abc durumuna 1 edit mesafesi ile başarıyla eşlendiği doğrulanmış ve test başarıyla sonuçlanmıştır.

---

## 6. Deneysel Tasarım ve Parametre Duyarlılık Analizi ##

**6.1 Parametre Varyasyon Analizi**

Yönerge kuralları gereği, sabit konfigürasyonun dışına çıkılarak SAX Alfabe Boyutu ve Kayan Pencere Genişliği 3, 4, 5, 6 değerleri için sistemde otomatik olarak test edilmiş ve üretilen toplam benzersiz durum sayıları kaydedilmiştir:

### Parametre Duyarlılığı ve Durum Uzayı (State Space) Analizi

Modelin karmaşıklığını ve hafıza gereksinimlerini optimize etmek için farklı alfabe boyutları ve pencere genişlikleri ile yapılan testlerin sonuçları aşağıda listelenmiştir.

| Alfabe Boyutu | Pencere Genişliği | Toplam Durum Sayısı |
| :---: | :---: | :---: |
| 3 | 3 | 27 |
| **3** | **4 (Baz Model)** | **81** |
| 3 | 5 | 227 |
| 3 | 6 | 533 |
| 4 | 3 | 62 |
| 4 | 4 | 210 |
| 4 | 5 | 582 |
| 4 | 6 | 1370 |
| 5 | 3 | 106 |
| 5 | 4 | 415 |
| 5 | 5 | 1248 |
| 5 | 6 | 2765 |
| 6 | 3 | 182 |
| 6 | 4 | 750 |
| 6 | 5 | 2223 |
| 6 | 6 | 4404 |

Parametre duyarlılık tablosu incelendiğinde, hiper-parametrelerdeki doğrusal artışların otomata durum uzayında doğrusal değil, üstel bir genişlemeye yol açtığı gözlemlenmiştir. $Alphabet=3, Window=3$ kombinasyonunda sistem sadece 27 durum üretirken, her iki değer de 6'ya çıkarıldığında durum uzayı fırlayarak 4404 benzersiz duruma ulaşmıştır. Literatürde "Durum Patlaması" olarak adlandırılan bu fenomen, aşırı öğrenme ve yüksek bellek/hesaplama maliyeti risklerini beraberinde getirmektedir. Projede hem hassasiyeti korumak hem de boyutsal laneti engellemek adına en optimum denge noktası olan Alfabe: 3, Pencere: 4 (81 Durum) baz model yapısı seçilmiştir.

**6.2 Derin Öğrenme Modellerinin Eğitimi ve Gürültü Analizi**

Derin öğrenme modellerinin (LSTM ve 1D-CNN) başlangıç ağırlıklarından bağımsız kararlı sonuçlar üretmesi adına tüm deneyler 5 farklı random seed (42, 123, 2026, 7, 999) döngüsünde koşturulmuştur. Modellerin aşırı öğrenmesini engellemek için eğitim esnasında validation loss izlenmiş ve iyileşme durduğunda eğitimi kesen Early Stopping (Patience: 5, Max Epoch: 50) callback katmanı entegre edilmiştir.

- 5 Deney Sonucu LSTM Ortalama Başarısı: %85.97
- 5 Deney Sonucu 1D-CNN Ortalama Başarısı: %84.12
- %10 Gaussian Gürültü Altında LSTM Başarısı: %86.00 (Temiz veri skoru: %86.84 | Performans Kaybı: 0.0084)

  Sisteme eklenen %10'luk yüksek rastgele Gaussian gürültüsüne rağmen model başarısındaki kaybın %1'in bile altında ($0.0084$) kalması, tasarlanan derin öğrenme ağ mimarisinin endüstriyel sahalardaki sinyal dalgalanmalarına ve kirli sensör gürültülerine karşı son derece yüksek bir dayanıklılık sergilediğini ortaya koymaktadır.

**6.3 İstatistiksel Anlamlılık Testi**

Modellerin 5 farklı seed boyunca elde ettiği doğruluk skorları arasındaki performans farkının tesadüfi mi yoksa yapısal mı olduğunu kanıtlamak adına non-parametrik Wilcoxon İşaretli Sıra Testi uygulanmıştır.
- Hesaplanan p-değeri: 0.0625
  
  Elde edilen $p = 0.0625$ değeri, genel kabul gören katı $\alpha = 0.05$ anlamlılık sınırının hafifçe üzerindedir. Bu durum, iki derin öğrenme modeli arasındaki performans farkının %95 güven düzeyinde kesin bir anlamlılık sergilemediğini, ancak %90 güven düzeyinde istatistiksel olarak anlamlı bir farka işaret ettiğini göstermektedir. Veri setinin yoğun zamansal bağımlılıklar barındırması sebebiyle, ardışık ilişkileri hafıza hücrelerinde tutabilen LSTM mimarisinin, mekansal öznitelik odaklı 1D-CNN modeline göre daha üstün performans göstermesi teorik beklentilerle tam uyumludur.

  ---

  ## 7. Rapor Görselleri ##
  **7.1 LSTM Hata Matrisi**
  
  Modelin test kümesi üzerindeki doğru/yanlış alarm ve anomali yakalama dağılımı:
  
  <img width="500" height="400" alt="confusion_matrix" src="https://github.com/user-attachments/assets/0ce1de4d-826b-43c6-b02e-fc5e47680f11" />
  
  **7.2 Parametre Duyarlılık Grafiği**

  Hiper-parametre değişimlerinin toplam durum sayısına olan üstel etkisinin çizgisel analizi:

  <img width="500" height="400" alt="param_sensitivity_plot" src="https://github.com/user-attachments/assets/b1707aba-4442-4bc8-b7fc-3ebd9e9a8673" />

  **7.3 Otomata Durum Geçiş Yoğunluk Haritası**

  Olasılıksal otomatadaki durumların birbirleri arasındaki geçiş yoğunluklarının matris tabanlı görselleştirilmesi:

  <img width="500" height="400" alt="transition_heatmap" src="https://github.com/user-attachments/assets/36032e23-795e-4d77-b293-bb1567925f94" />
