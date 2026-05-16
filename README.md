# Ortadoğu Silah Tedarik Örüntülerinin Yapısal Analizi: Soğuk Savaş Sonrası Dönüşümün Kesintili Regresyon Tasarımı ile İncelenmesi

Bu repo, Soğuk Savaş'ın sona ermesinin Ortadoğu silah tedarik örüntüleri üzerindeki etkilerini SIPRI Silah Transferleri Veritabanı (Arms Transfers Database) verileri üzerinden inceleyen çalışmanın replikasyon paketini içermektedir. Analiz, 1990 yılını dışsal bir kırılma noktası olarak ele alan kesintili regresyon tasarımı (Regression Discontinuity Design, RDD) çerçevesinde yürütülmüş; bulgular Newey-West HAC standart hataları ile raporlanmıştır.

## Araştırma Sorusu

Çalışma, küresel sistemde tek kutupluluğa geçişin Ortadoğu silah pazarında nasıl bir yapısal dönüşüme yol açtığını altı farklı bağımlı değişken üzerinden test etmektedir: (i) ABD'nin tedarik anlaşmalarındaki payı, (ii) ABD'nin gerçekleşen tedarikteki payı (TIV bazlı), (iii) SSCB/Rusya'nın tedarikteki payı, (iv) iki süper gücün birleşik payı, (v) alıcı ülke başına ortalama tedarikçi sayısı ve (vi) bölge içi tedariğin oranı.

## Veri

Analizde kullanılan ham veri seti, SIPRI Silah Transferleri Veritabanı'nın işlem-bazlı kayıt cetvelinden (trade register) elde edilmiştir. Veri, 1948–2023 dönemini kapsamakta ve `trade-register.csv` dosyası içinde sunulmaktadır. Veri kaynağı: [SIPRI Arms Transfers Database](https://www.sipri.org/databases/armstransfers). SIPRI veri kullanım koşulları için orijinal kaynağa başvurulmalıdır.

Veride bazı idiosenkratik düzeltmeler yapılmıştır: 2024 yılı henüz tam kaydı içermediği için analiz dışında bırakılmış, ABD için 1952 öncesi (kayıt eksikliği) ve ülke başına ortalama tedarikçi sayısı için 2020 sonrası kapsam dışı tutulmuş, SIPRI'nin 2021 Rusya kaydındaki boşluk 0 olarak atanmıştır.

## Repo Yapısı

```
.
├── README.md                # Mevcut dosya
├── analysis.jl              # Ana analiz betiği (RDD, LOESS, grafikler)
├── trade-register.csv       # SIPRI ham veri
└── outputs/                 # Üretilen grafikler ve tablolar
    ├── p1_comb.pdf          # ABD anlaşma payı (RDD + LOESS)
    ├── p2_comb.pdf          # ABD TIV payı (RDD + LOESS)
    ├── p3_comb.pdf          # SSCB/Rusya TIV payı (RDD + LOESS)
    ├── p4_comb.pdf          # ABD + SSCB/Rusya birleşik payı
    ├── p5_comb.pdf          # ABD vs SSCB/Rusya karşılaştırmalı LOESS
    ├── p7_comb.pdf          # Ülke başına ortalama tedarikçi sayısı
    ├── p8_comb.pdf          # Bölge içi tedarik oranı
    └── p9.pdf               # Avrupalı tedarikçiler + Çin LOESS
```

## Gereksinimler

Analiz Julia (≥ 1.9) ortamında geliştirilmiştir. Gerekli paketler:

```julia
using Pkg
Pkg.add([
    "DataFrames", "CSV", "Plots", "StatsBase", "GLM",
    "CovarianceMatrices", "RegressionTables", "LinearAlgebra",
    "Distributions", "Loess", "ColorSchemes"
])
```

## Replikasyon

Analizi yeniden üretmek için:

```bash
julia script.jl
```

Betik tüm RDD tahminlerini çalıştırır, anlamlılık testlerini raporlar ve grafiklerin PDF çıktılarını üretir. Tüm yorum satırları Türkçedir ve her bölüm hangi bağımlı değişkene karşılık geldiğini açıkça belirtir.

## Metodoloji Özeti

Her bağımlı değişken için aşağıdaki RDD spesifikasyonu tahmin edilmiştir:

$$Y_t = \beta_0 + \beta_1 D_t + \beta_2 (\text{Yıl}_t - 1990) + \beta_3 \, D_t \times (\text{Yıl}_t - 1990) + \varepsilon_t$$

Burada $D_t$ Soğuk Savaş sonrası kukla değişkenidir (yıl ≥ 1990 ise 1). $\beta_1$ eşik anındaki seviye sıçramasını, $\beta_3$ ise eğim kırılmasını yakalar. Standart hatalar Newey-West (Bartlett çekirdeği) HAC tahmincisi ile hesaplanmıştır. RDD tahminlerinin yanı sıra LOESS düzgünleştirme eğrileri (span parametreleri seriye göre 0.5–0.825 arasında) doğrusal-olmayan trendlerin görsel kontrolü için tahmin edilmiştir.

## Atıf

Bu repodaki kod ya da bulguları kullanmanız hâlinde aşağıdaki şekilde atıfta bulunmanız rica olunur:

```
Güney, Ahmet Zahit, ve Abdullah Kabaoğlu. Kutupluluk ve Tedarik: Soğuk Savaş’tan Günümüze Ortadoğu Silah Ticaretinin Yapısal Dönüşümü. Kültür Medeniyet Vakfı, 2026. https://doi.org/10.5281/zenodo.20229063.

Replikasyon paketi: https://github.com/kume-arastirma/Kutupluluk-ve-Ortado-u-Silah-Ticareti/
```

## Lisans

Kod MIT lisansı altında dağıtılmaktadır. SIPRI verisi için SIPRI'nin kendi kullanım koşulları geçerlidir.

## İletişim

Sorular ve geri bildirimler için: guvenlik.strateji@kumevakfi.org
