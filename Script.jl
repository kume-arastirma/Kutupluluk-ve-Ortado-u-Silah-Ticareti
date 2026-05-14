using DataFrames, CSV, Plots, Plots.Measures, StatsBase, GLM, CovarianceMatrices, RegressionTables, LinearAlgebra, Distributions, Loess

# Tüm grafikler için ortak görsel ayarlar (750×450 piksel, 300 dpi)
default(size=(750, 450), dpi=300, left_margin=5mm)

# =============================================================================
# BÖLÜM 1: ABD'nin Tedarikçi Payı — Anlaşma Sayısı Bazlı
# =============================================================================

# Ham SIPRI verisi yükleniyor
sipri_dat = CSV.read("trade-register.csv", DataFrame)

# Sütun 2 = Supplier; ABD ise 1, değilse 0 ikili değişken oluşturuluyor
transform!(sipri_dat, 2 => ByRow(x -> x == "United States" ? 1 : 0) => :US_dummy)

# Yıl (sütun 3) bazında gruplama; her yılda ABD anlaşmalarının oranı = US_dummy'nin ortalaması
sipri_new = combine(groupby(sipri_dat, 3), :US_dummy => mean)
# 2024 henüz tam veri içermediğinden dışarıda bırakılıyor
sipri_new = subset(sipri_new, 1 => ByRow(x -> x != 2024))

rename!(sipri_new, 1 => "Year", 2 => "Ratio")

# Tüm dönem için tek doğrusal trend (ön bakış)
ratio_coefs = coef(lm(@formula(Ratio ~ Year), sipri_new))

plot(sipri_new.Year, sipri_new.Ratio)
Plots.abline!(reverse(ratio_coefs)...)  # abline! (eğim, kesişim) sırasını bekler; coef() [kesişim, eğim] döndürür
ylims!(0.1, 0.7)


# Soğuk Savaş / Sonrası ayrımı; 1990 her iki kümede bırakılıyor (RDD sürekliliği için)
sipri_coldwar = subset(sipri_new, :Year => ByRow(x -> x <= 1990))
sipri_postcoldwar = subset(sipri_new, :Year => ByRow(x -> x >= 1990))

# Her döneme ayrı OLS regresyonu
ratio_coefs_cw = coef(lm(@formula(Ratio ~ Year), sipri_coldwar))
ratio_coefs_postcw = coef(lm(@formula(Ratio ~ Year), sipri_postcoldwar))

# Tahmin değerleri yeni kolon olarak ekleniyor (df[!, :col] = ... DataFrames API'si)
sipri_postcoldwar[!, :predicted] = ratio_coefs_postcw[1] .+ sipri_postcoldwar.Year .* ratio_coefs_postcw[2]
sipri_coldwar[!, :predicted] = ratio_coefs_cw[1] .+ sipri_coldwar.Year .* ratio_coefs_cw[2]

# Ana grafik: gözlemler (siyah) + dönemlik trend çizgileri
# kırmızı = Soğuk Savaş, lacivert = Soğuk Savaş sonrası
plot(sipri_new.Year, sipri_new.Ratio, c=:black, label=:none)
scatter!(sipri_new.Year, sipri_new.Ratio, ms=2, c=:black, label=:none)
plot!(sipri_postcoldwar.Year, sipri_postcoldwar.predicted, label=:none, c=:navyblue)
plot!(sipri_coldwar.Year, sipri_coldwar.predicted, label=:none, c=:firebrick)
p1 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(0.2:0.1:0.6, ["%20", "%30", "%40", "%50", "%60"]), ylabel="ABD ile Yapılan Tedarik Anlaşmalarının Oranı")  # 


# =============================================================================
# RDD (Regression Discontinuity Design) — Kesim Noktası: 1990
# Model: Ratio ~ D + Year_centered + D*Year_centered
#   D            = Soğuk Savaş sonrası kukla (>= 1990 ise 1)
#   Year_centered = Yıl - 1990 (kesim noktası etrafında merkezleniyor)
#   D*Year_centered = eğim kırılmasını (slope change) yakalar
# =============================================================================
cutoff = 1990
sipri_new[!, :Year_centered] = sipri_new.Year .- cutoff
sipri_new[!, :D] = [val >= cutoff ? 1 : 0 for val in sipri_new.Year]
rdd_model = lm(@formula(Ratio ~ D * Year_centered), sipri_new)

# Newey-West (Bartlett çekirdeği) ile HAC standart hatalar — ardışık bağımlılığa karşı güçlü
vcov_hc = vcov(Bartlett{NeweyWest}(), rdd_model)
# LaTeX formatında regresyon tablosu (HAC varyans-kovaryans matrisi ile)
RegressionTables.regtable(rdd_model, vcov=vcov_hc, digits=5, render=LatexTable(), regression_statistics=[Nobs, R2])

# p-değerinden anlamlılık yıldızı döndüren yardımcı fonksiyon
# df = gözlem sayısı - parametre sayısı (4 katsayı: sabit, D, Year_centered, D*Year_centered)
function p_values(t_scores::Vector{<:Real}, df::Int)
    df <= 0 && throw(ArgumentError("Degrees of freedom must be positive"))
    dist = TDist(df)
    res = [ccdf(dist, abs(t)) for t in t_scores]

    function stars(p)
        if p < 0.01
            return "***"
        elseif p < 0.05
            return "**"
        elseif p < 0.1
            return "*"
        else
            return ""
        end
    end
    return stars.(res)
end

# HAC standart hatalarla t istatistikleri: katsayı / SE
t_scores = coef(rdd_model) ./ sqrt.(diag(vcov_hc))
p_values(t_scores, Int64(nobs(rdd_model)) - 4)



# LOESS eğrisi (span=0.6): doğrusal olmayan trendi görselleştirmek için
loess_mod = loess(sipri_new.Year, sipri_new.Ratio, span=0.6)

cw_idx = findfirst(==(cutoff), sipri_new.Year)  # cutoff yılının (1990) indeksi

# Kombine grafik: regresyon trendleri (düz) + LOESS (noktalı, yarı saydam)
plot(sipri_new.Year, sipri_new.Ratio, c=:black, label=:none)
scatter!(sipri_new.Year, sipri_new.Ratio, ms=2, c=:black, label=:none)
plot!(sipri_postcoldwar.Year, sipri_postcoldwar.predicted, label=:none, c=:navyblue)
plot!(sipri_coldwar.Year, sipri_coldwar.predicted, label=:none, c=:firebrick)
plot!(sipri_new.Year[1:cw_idx], predict(loess_mod, sipri_new.Year[1:cw_idx]), c=:firebrick, label=:none, alpha=0.45, linestyle=:dot)
plot!(sipri_new.Year[cw_idx:end], predict(loess_mod, sipri_new.Year[cw_idx:end]), c=:navyblue, label=:none, alpha=0.45, linestyle=:dot)
p1_comb = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(0.2:0.1:0.6, ["%20", "%30", "%40", "%50", "%60"]), ylabel="ABD ile Yapılan Tedarik Anlaşmalarının Oranı")

# savefig(p1_comb, "p1_comb.pdf")


# =============================================================================
# BÖLÜM 2: ABD'nin Tedarikçi Payı — TIV (Silah Transfer Değeri) Bazlı
# Sütunlar: [2]=Supplier, [3]=Year, [15]=SIPRI TIV değeri
# =============================================================================

sipri_dat_2 = CSV.read("trade-register.csv", DataFrame)

select!(sipri_dat_2, [2, 3, 15])

# Her yıl toplam küresel silah transferi (TIV)
year_sale = combine(groupby(sipri_dat_2, 2), 3 => sum)

# Sadece ABD kaynaklı transferler
sipri_dat_2_us = subset(sipri_dat_2, 1 => ByRow(x -> x == "United States"))
sipri_dat_2_us = combine(groupby(sipri_dat_2_us, 2), 3 => sum)

rename!(year_sale, [1, 2] .=> [:year, :total_sale])
rename!(sipri_dat_2_us, [1, 2] .=> [:year, :us_sale])

# ABD satışlarını toplam satışla birleştir; eşleşmeyen yıllar için us_sale = 0 olacak
leftjoin!(sipri_dat_2_us, year_sale, on=:year)

# ABD'nin küresel TIV içindeki payı
transform!(sipri_dat_2_us, [2, 3] => ByRow((x, y) -> x / y) => :us_ratio)

# 1952 öncesinde ABD veri girişi yok; bu yıllar analizden dışarıda bırakılıyor
sipri_dat_2_us_post1952 = subset(sipri_dat_2_us, :year => ByRow(x -> x > 1952))

sipri_dat_2_us_coldwar = subset(sipri_dat_2_us_post1952, :year => ByRow(x -> x <= 1990))
sipri_dat_2_us_postcoldwar = subset(sipri_dat_2_us_post1952, :year => ByRow(x -> x >= 1990))

ratio_coefs_cw = coef(lm(@formula(us_ratio ~ year), sipri_dat_2_us_coldwar))
ratio_coefs_postcw = coef(lm(@formula(us_ratio ~ year), sipri_dat_2_us_postcoldwar))

sipri_dat_2_us_coldwar[!, :predicted] = ratio_coefs_cw[1] .+ sipri_dat_2_us_coldwar.year .* ratio_coefs_cw[2]
sipri_dat_2_us_postcoldwar[!, :predicted] = ratio_coefs_postcw[1] .+ sipri_dat_2_us_postcoldwar.year .* ratio_coefs_postcw[2]

plot(sipri_dat_2_us_post1952.year, sipri_dat_2_us_post1952.us_ratio, c=:black, label=:none)
scatter!(sipri_dat_2_us_post1952.year, sipri_dat_2_us_post1952.us_ratio, ms=2, c=:black, label=:none)
plot!(sipri_dat_2_us_coldwar.year, sipri_dat_2_us_coldwar.predicted, label=:none, c=:firebrick)
plot!(sipri_dat_2_us_postcoldwar.year, sipri_dat_2_us_postcoldwar.predicted, label=:none, c=:navyblue)
p2 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xlim=(1950, 2024), xticks=1953:10:2023, yticks=(0.2:0.2:0.8, ["%20", "%40", "%60", "%80"]), ylabel="ABD'den Tedarik Edilen Silahların Oranı", right_margin=5mm)


# RDD için merkez ve kukla değişkeni
sipri_dat_2_us_post1952[!, :year_centered] = sipri_dat_2_us_post1952.year .- cutoff
sipri_dat_2_us_post1952[!, :D] = [val >= cutoff ? 1 : 0 for val in sipri_dat_2_us_post1952.year]
rdd_model = lm(@formula(us_ratio ~ D * year_centered), sipri_dat_2_us_post1952)

# HAC standart hatalar ve regresyon tablosu
vcov_hc = vcov(Bartlett{NeweyWest}(), rdd_model)
RegressionTables.regtable(rdd_model, vcov=vcov_hc, digits=5, render=LatexTable(), regression_statistics=[Nobs, R2])

t_scores = coef(rdd_model) ./ sqrt.(diag(vcov_hc))
p_values(t_scores, Int64(nobs(rdd_model)) - 4)

# LOESS eğrisi (span=0.6); veri 1953'ten başladığı için cutoff indeksi buraya özgü hesaplanıyor
loess_mod = loess(sipri_dat_2_us_post1952.year, sipri_dat_2_us_post1952.us_ratio, span=0.6)


# Veri 1953'ten başlıyor: index(1990) = 1990 - 1953 + 1 = 38
cw_idx2 = findfirst(==(cutoff), sipri_dat_2_us_post1952.year)

# LOESS grafik — sadece LOESS eğrileri (tek renk bölünmesi)
plot(sipri_dat_2_us_post1952.year, sipri_dat_2_us_post1952.us_ratio, c=:black, label=:none)
scatter!(sipri_dat_2_us_post1952.year, sipri_dat_2_us_post1952.us_ratio, c=:black, ms=2, label=:none)
plot!(sipri_dat_2_us_post1952.year[1:cw_idx2], predict(loess_mod, sipri_dat_2_us_post1952.year[1:cw_idx2]), c=:firebrick, label=:none)
plot!(sipri_dat_2_us_post1952.year[cw_idx2:end], predict(loess_mod, sipri_dat_2_us_post1952.year[cw_idx2:end]), c=:navyblue, label=:none)
p3 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1953:10:2023, yticks=(0.2:0.2:0.8, ["%20", "%40", "%60", "%80"]), ylabel="ABD'den Tedarik Edilen Silahların Oranı")

# Kombine grafik: regresyon trendleri (düz) + LOESS (noktalı, yarı saydam)
plot(sipri_dat_2_us_post1952.year, sipri_dat_2_us_post1952.us_ratio, c=:black, label=:none)
scatter!(sipri_dat_2_us_post1952.year, sipri_dat_2_us_post1952.us_ratio, c=:black, ms=2, label=:none)
plot!(sipri_dat_2_us_coldwar.year, sipri_dat_2_us_coldwar.predicted, label=:none, c=:firebrick)
plot!(sipri_dat_2_us_postcoldwar.year, sipri_dat_2_us_postcoldwar.predicted, label=:none, c=:navyblue)
plot!(sipri_dat_2_us_post1952.year[1:cw_idx2], predict(loess_mod, sipri_dat_2_us_post1952.year[1:cw_idx2]), c=:firebrick, label=:none, alpha=0.45, linestyle=:dot)
plot!(sipri_dat_2_us_post1952.year[cw_idx2:end], predict(loess_mod, sipri_dat_2_us_post1952.year[cw_idx2:end]), c=:navyblue, label=:none, alpha=0.45, linestyle=:dot)
p2_comb = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2030, yticks=(0.2:0.2:0.8, ["%20", "%40", "%60", "%80"]), ylabel="ABD'den Tedarik Edilen Silahların Oranı", xlims=(1952, 2024))

# savefig(p2_comb, "p2_comb.pdf")






# NOT: bu blok sipri_dat_2_us_coldwar/postcoldwar'ı 1952 filtresi olmadan TAM veriyle yeniden tanımlıyor
# (önceki blokta post1952 filtreli versiyonlar oluşturulmuştu; buradaki versiyon Bölüm 2.4 için)
sipri_dat_2_us_coldwar = subset(sipri_dat_2_us, :year => ByRow(x -> x <= 1990))
sipri_dat_2_us_postcoldwar = subset(sipri_dat_2_us, :year => ByRow(x -> x >= 1990))

ratio_coefs_cw = coef(lm(@formula(us_ratio ~ year), sipri_dat_2_us_coldwar))
ratio_coefs_postcw = coef(lm(@formula(us_ratio ~ year), sipri_dat_2_us_postcoldwar))

sipri_dat_2_us_postcoldwar[!, :predicted] = ratio_coefs_postcw[1] .+ sipri_dat_2_us_postcoldwar.year .* ratio_coefs_postcw[2]
sipri_dat_2_us_coldwar[!, :predicted] = ratio_coefs_cw[1] .+ sipri_dat_2_us_coldwar.year .* ratio_coefs_cw[2]

# Tam ABD verisi üzerinde LOESS — Bölüm 2.4 karşılaştırma grafiği için saklanıyor
loess_mod = loess(sipri_dat_2_us.year, sipri_dat_2_us.us_ratio, span=0.66)

cw_idx_us = findfirst(==(cutoff), sipri_dat_2_us.year)
us_cw_loess = predict(loess_mod, sipri_dat_2_us.year[1:43])
us_pcw_loess = predict(loess_mod, sipri_dat_2_us.year[43:76])

# =============================================================================
# BÖLÜM 2.1: SSCB/Rusya'nın Tedarikçi Payı — TIV Bazlı
# =============================================================================

sipri_dat_2 = CSV.read("trade-register.csv", DataFrame)

select!(sipri_dat_2, [2, 3, 15])

# Her yıl toplam küresel silah transferi (TIV)
year_sale = combine(groupby(sipri_dat_2, 2), 3 => sum)

# SSCB ve Rusya birleştirilerek tek tedarikçi olarak ele alınıyor
sipri_dat_2_russia = subset(sipri_dat_2, 1 => ByRow(x -> x in ["Soviet Union", "Russia"]))
sipri_dat_2_russia = combine(groupby(sipri_dat_2_russia, 2), 3 => sum)

rename!(year_sale, [1, 2] .=> [:year, :total_sale])
rename!(sipri_dat_2_russia, [1, 2] .=> [:year, :russian_sale])

# 2021 verisinde Rusya kaydı yok; 0 olarak ekleniyor (SIPRI boş bırakmış)
push!(sipri_dat_2_russia, [2021, 0])
sort!(sipri_dat_2_russia, :year)

leftjoin!(sipri_dat_2_russia, year_sale, on=:year)

# SSCB/Rusya'nın küresel TIV içindeki payı
transform!(sipri_dat_2_russia, [2, 3] => ByRow((x, y) -> x / y) => :russia_ratio)

# Soğuk Savaş / Sonrası ayrımı
sipri_dat_2_russia_coldwar = subset(sipri_dat_2_russia, :year => ByRow(x -> x <= 1990))
sipri_dat_2_russia_postcoldwar = subset(sipri_dat_2_russia, :year => ByRow(x -> x >= 1990))

ratio_coefs_cw = coef(lm(@formula(russia_ratio ~ year), sipri_dat_2_russia_coldwar))
ratio_coefs_postcw = coef(lm(@formula(russia_ratio ~ year), sipri_dat_2_russia_postcoldwar))


sipri_postcoldwar_russia = DataFrame(Year=sipri_dat_2_russia_postcoldwar.year)
sipri_coldwar_russia = DataFrame(Year=sipri_dat_2_russia_coldwar.year)

sipri_postcoldwar_russia[!, :predicted] = ratio_coefs_postcw[1] .+ sipri_dat_2_russia_postcoldwar.year .* ratio_coefs_postcw[2]
sipri_coldwar_russia[!, :predicted] = ratio_coefs_cw[1] .+ sipri_dat_2_russia_coldwar.year .* ratio_coefs_cw[2]

plot(sipri_dat_2_russia.year, sipri_dat_2_russia.russia_ratio, c=:black, label=:none)
scatter!(sipri_dat_2_russia.year, sipri_dat_2_russia.russia_ratio, ms=2, c=:black, label=:none)
plot!(sipri_coldwar_russia.Year, sipri_coldwar_russia.predicted, label=:none, c=:firebrick)
plot!(sipri_postcoldwar_russia.Year, sipri_postcoldwar_russia.predicted, label=:none, c=:navyblue)
p4 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(0:0.2:0.6, ["%0", "%20", "%40", "%60"]), ylabel="SSCB/Rusya'dan Tedarik Edilen Silahların Oranı")


# RDD için merkez ve kukla değişkeni
sipri_dat_2_russia[!, :year_centered] = sipri_dat_2_russia.year .- cutoff  # 
sipri_dat_2_russia[!, :D] = [val >= cutoff ? 1 : 0 for val in sipri_dat_2_russia.year]
rdd_model = lm(@formula(russia_ratio ~ D * year_centered), sipri_dat_2_russia)

# HAC standart hatalar ve regresyon tablosu
vcov_hc = vcov(Bartlett{NeweyWest}(), rdd_model)
RegressionTables.regtable(rdd_model, vcov=vcov_hc, digits=5, render=LatexTable(), regression_statistics=[Nobs, R2])

t_scores = coef(rdd_model) ./ sqrt.(diag(vcov_hc))
p_values(t_scores, Int64(nobs(rdd_model)) - 4)

# LOESS eğrisi (span=0.75 — Rusya verisi daha gürültülü olduğundan daha geniş bant)
loess_mod = loess(sipri_dat_2_russia.year, sipri_dat_2_russia.russia_ratio, span=0.75)

cw_idx_russia = findfirst(==(cutoff), sipri_dat_2_russia.year)

# LOESS grafik — sadece LOESS eğrileri
plot(sipri_dat_2_russia.year, sipri_dat_2_russia.russia_ratio, c=:black, label=:none)
scatter!(sipri_dat_2_russia.year, sipri_dat_2_russia.russia_ratio, ms=2, c=:black, label=:none)
plot!(sipri_dat_2_russia.year[1:cw_idx_russia], predict(loess_mod, sipri_dat_2_russia.year[1:cw_idx_russia]), c=:firebrick, label=:none)
plot!(sipri_dat_2_russia.year[cw_idx_russia:end], predict(loess_mod, sipri_dat_2_russia.year[cw_idx_russia:end]), c=:navyblue, label=:none)
p5 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(0:0.2:0.6, ["%0", "%20", "%40", "%60"]), ylabel="SSCB/Rusya'dan Tedarik Edilen Silahların Oranı")

# Bölüm 2.4 karşılaştırma grafiği için LOESS tahminleri saklanıyor
russia_cw_loess = predict(loess_mod, sipri_dat_2_russia.year[1:cw_idx_russia])
russia_pcw_loess = predict(loess_mod, sipri_dat_2_russia.year[cw_idx_russia:end])


# Kombine grafik: regresyon trendleri (düz) + LOESS (noktalı, yarı saydam)
plot(sipri_dat_2_russia.year, sipri_dat_2_russia.russia_ratio, c=:black, label=:none)
scatter!(sipri_dat_2_russia.year, sipri_dat_2_russia.russia_ratio, ms=2, c=:black, label=:none)
plot!(sipri_coldwar_russia.Year, sipri_coldwar_russia.predicted, label=:none, c=:firebrick)
plot!(sipri_postcoldwar_russia.Year, sipri_postcoldwar_russia.predicted, label=:none, c=:navyblue)
plot!(sipri_dat_2_russia.year[1:cw_idx_russia], predict(loess_mod, sipri_dat_2_russia.year[1:cw_idx_russia]), c=:firebrick, label=:none, alpha=0.45, linestyle=:dot)
plot!(sipri_dat_2_russia.year[cw_idx_russia:end], predict(loess_mod, sipri_dat_2_russia.year[cw_idx_russia:end]), c=:navyblue, label=:none, alpha=0.45, linestyle=:dot)
p3_comb = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(0:0.2:0.6, ["%0", "%20", "%40", "%60"]), ylabel="SSCB/Rusya'dan Tedarik Edilen Silahların Oranı")

# savefig(p3_comb, "p3_comb.pdf")






# =============================================================================
# BÖLÜM 2.3: ABD + SSCB/Rusya Birleşik Payı — TIV Bazlı
# =============================================================================

sipri_dat_2 = CSV.read("trade-register.csv", DataFrame)

select!(sipri_dat_2, [2, 3, 15])

# Her yıl toplam küresel silah transferi (TIV)
year_sale = combine(groupby(sipri_dat_2, 2), 3 => sum)

# ABD + SSCB/Rusya filtreleniyor; iki güç birleştirilerek toplam hesaplanıyor
sipri_dat_2_total = subset(sipri_dat_2, 1 => ByRow(x -> x in ["Soviet Union", "Russia", "United States"]))

# Tedarikçi adları "US" / "Russia" olarak normalleştiriliyor (groupby için)
transform!(sipri_dat_2_total, 1 => ByRow(x -> x == "United States" ? "US" : "Russia"), renamecols=false)

# Yıl bazında iki gücün toplam TIV'i
sipri_dat_2_total = combine(groupby(sipri_dat_2_total, 2), 3 => sum)

rename!(year_sale, [1, 2] .=> [:year, :total_sale])
rename!(sipri_dat_2_total, [1, 2] .=> [:year, :us_russia_sale])

leftjoin!(sipri_dat_2_total, year_sale, on=:year)

# İki gücün birleşik küresel TIV payı
transform!(sipri_dat_2_total, [2, 3] => ByRow((x, y) -> x / y) => :us_russia_ratio)

# Soğuk Savaş / Sonrası ayrımı
sipri_dat_2_total_coldwar = subset(sipri_dat_2_total, :year => ByRow(x -> x <= 1990))
sipri_dat_2_total_postcoldwar = subset(sipri_dat_2_total, :year => ByRow(x -> x >= 1990))

ratio_coefs_cw = coef(lm(@formula(us_russia_ratio ~ year), sipri_dat_2_total_coldwar))
ratio_coefs_postcw = coef(lm(@formula(us_russia_ratio ~ year), sipri_dat_2_total_postcoldwar))

sipri_postcoldwar_total = DataFrame(Year=sipri_dat_2_total_postcoldwar.year)
sipri_coldwar_total = DataFrame(Year=sipri_dat_2_total_coldwar.year)

sipri_postcoldwar_total[!, :predicted] = ratio_coefs_postcw[1] .+ sipri_dat_2_total_postcoldwar.year .* ratio_coefs_postcw[2]
sipri_coldwar_total[!, :predicted] = ratio_coefs_cw[1] .+ sipri_dat_2_total_coldwar.year .* ratio_coefs_cw[2]

plot(sipri_dat_2_total.year, sipri_dat_2_total.us_russia_ratio, c=:black, label=:none)
scatter!(sipri_dat_2_total.year, sipri_dat_2_total.us_russia_ratio, ms=2, c=:black, label=:none)
plot!(sipri_coldwar_total.Year, sipri_coldwar_total.predicted, label=:none, c=:firebrick)
plot!(sipri_postcoldwar_total.Year, sipri_postcoldwar_total.predicted, label=:none, c=:navyblue)
p6 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(0.2:0.2:0.8, ["%20", "%40", "%60", "%80"]), ylabel="ABD veya SSCB/Rusya'dan Tedarik Edilen Silahların Oranı", yguidefontsize=9)


# RDD için merkez ve kukla değişkeni
sipri_dat_2_total[!, :year_centered] = sipri_dat_2_total.year .- cutoff
sipri_dat_2_total[!, :D] = [val >= cutoff ? 1 : 0 for val in sipri_dat_2_total.year]
rdd_model = lm(@formula(us_russia_ratio ~ D * year_centered), sipri_dat_2_total)

# HAC standart hatalar ve regresyon tablosu
vcov_hc = vcov(Bartlett{NeweyWest}(), rdd_model)
RegressionTables.regtable(rdd_model, vcov=vcov_hc, digits=5, render=LatexTable(), regression_statistics=[Nobs, R2])

t_scores = coef(rdd_model) ./ sqrt.(diag(vcov_hc))
p_values(t_scores, Int64(nobs(rdd_model)) - 4)


loess_mod = loess(sipri_dat_2_total.year, sipri_dat_2_total.us_russia_ratio, span=0.66)
cw_idx_total = findfirst(==(cutoff), sipri_dat_2_total.year)

# LOESS grafik — sadece LOESS eğrileri
plot(sipri_dat_2_total.year, sipri_dat_2_total.us_russia_ratio, c=:black, label=:none)
scatter!(sipri_dat_2_total.year, sipri_dat_2_total.us_russia_ratio, ms=2, c=:black, label=:none)
plot!(sipri_dat_2_total.year[1:cw_idx_total], predict(loess_mod, sipri_dat_2_total.year[1:cw_idx_total]), c=:firebrick, label=:none)
plot!(sipri_dat_2_total.year[cw_idx_total:end], predict(loess_mod, sipri_dat_2_total.year[cw_idx_total:end]), c=:navyblue, label=:none)
p7 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(0.2:0.2:0.8, ["%20", "%40", "%60", "%80"]), ylabel="ABD veya SSCB/Rusya'dan Tedarik Edilen Silahların Oranı", yguidefontsize=9)

# Kombine grafik: regresyon trendleri (düz) + LOESS (noktalı, yarı saydam)
plot(sipri_dat_2_total.year, sipri_dat_2_total.us_russia_ratio, c=:black, label=:none)
scatter!(sipri_dat_2_total.year, sipri_dat_2_total.us_russia_ratio, ms=2, c=:black, label=:none)
plot!(sipri_coldwar_total.Year, sipri_coldwar_total.predicted, label=:none, c=:firebrick)
plot!(sipri_postcoldwar_total.Year, sipri_postcoldwar_total.predicted, label=:none, c=:navyblue)
plot!(sipri_dat_2_total.year[1:cw_idx_total], predict(loess_mod, sipri_dat_2_total.year[1:cw_idx_total]), c=:firebrick, label=:none, alpha=0.45, linestyle=:dot)
plot!(sipri_dat_2_total.year[cw_idx_total:end], predict(loess_mod, sipri_dat_2_total.year[cw_idx_total:end]), c=:navyblue, label=:none, alpha=0.45, linestyle=:dot)
p4_comb = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(0.2:0.2:0.8, ["%20", "%40", "%60", "%80"]), ylabel="ABD veya SSCB/Rusya'dan Tedarik Edilen Silahların Oranı", yguidefontsize=9)

# savefig(p4_comb, "p4_comb.pdf")



# =============================================================================
# BÖLÜM 2.4: ABD vs SSCB/Rusya Karşılaştırmalı LOESS Grafiği (1954–2022)
# us_cw/pcw_loess ve russia_cw/pcw_loess önceki bölümlerde hesaplandı.
# Grafik aralığı: 1954:2022 = 69 yıl → her iki birleşik vektör 69 elemanlı olmalı.
# =============================================================================

# ABD serisini 1954–2022 aralığına hizalamak için dilimler:
#   us_cw_loess[7:43]  → veri 1948'den başladığından ilk 6 yıl (1948–1953) atlanıyor,
#                         43. eleman 1990'a karşılık geliyor
#   us_pcw_loess[2:33] → 1. eleman 1990 (CW serisinde zaten var), 33. eleman 2022
us_comb_loess = vcat(us_cw_loess[7:43], us_pcw_loess[2:33])

# Rusya serisini 1954–2022 aralığına hizalamak için dilimler:
#   russia_cw_loess[1:36]  → Rusya verisi 1954'ten başlıyor; [1:36] = 1954–1989
#   russia_pcw_loess[1:33] → 1. eleman 1990, 33. eleman 2022 (2023 atlanıyor)
russia_comb_loess = vcat(russia_cw_loess[1:37], russia_pcw_loess[2:33])

# Uzunluk kontrolü — veri değişirse mismatch'i erkenden yakalar
@assert length(us_comb_loess) == length(1954:2022) "us_comb_loess uzunluğu 1954:2022 aralığıyla eşleşmiyor"
@assert length(russia_comb_loess) == length(1954:2022) "russia_comb_loess uzunluğu 1954:2022 aralığıyla eşleşmiyor"

plot(1954:2022, us_comb_loess, c=:navyblue, label="ABD")
scatter!(1954:2022, us_comb_loess, c=:navyblue, label=:none, ms=2, msw=0)
plot!(1954:2022, russia_comb_loess, c=:firebrick, label="SSCB/Rusya")
scatter!(1954:2022, russia_comb_loess, c=:firebrick, label=:none, ms=2, msw=0)
Plots.vline!([1967], c=:grey50, linestyle=:dash, label=:none)
Plots.vline!([1980], c=:grey50, linestyle=:dash, label=:none)
annotate!(1967, 0.5, text(" 6 Gün Savaşı", :grey50, :left, 8, rotation=0))
annotate!(1980.5, 0.5, text("  Reagan\nBaşkanlığı", :grey50, :left, 8, rotation=0))
annotate!(1990, 0.3, text(" Soğuk Savaş'ın\n         Sonu", :black, :left, 8, rotation=0))
Plots.vline!([2003], c=:grey50, linestyle=:dash, label=:none)
annotate!(2003, 0.3, text(" Irak Savaşı", :grey50, :left, 8, rotation=0))
p5 = Plots.vline!([1990], c=:black, linestyle=:dash, label=:none, xticks=1955:10:2025, ylim=(0, 0.6), yticks=(0:0.1:0.5, ["%0", "%10", "%20", "%30", "%40", "%50"]), ylabel="Menşeine Göre Tedarik Edilen Silahların Oranı")

# savefig(p5, "p5.pdf")


# =============================================================================
# BÖLÜM 2.6: Ülke Bazlı Grafik Fonksiyonları
# Her fonksiyon trade-register.csv'yi yeniden okuyarak bağımsız çalışır.
# =============================================================================

# Yalnızca ham gözlemleri çizen temel grafik fonksiyonu
function drawer(country)
    sipri_dat_2 = CSV.read("trade-register.csv", DataFrame)
    select!(sipri_dat_2, [2, 3, 15])
    year_sale = combine(groupby(sipri_dat_2, 2), 3 => sum)
    sipri_dat_2_total = subset(sipri_dat_2, 1 => ByRow(x -> x == country))
    sipri_dat_2_total = combine(groupby(sipri_dat_2_total, 2), 3 => sum)

    rename!(year_sale, [1, 2] .=> [:year, :total_sale])
    rename!(sipri_dat_2_total, [1, 2] .=> [:year, :individual_sale])

    # outerjoin: ülkenin satış yapmadığı yıllar için individual_sale = 0
    sipri_dat_2_total = outerjoin(sipri_dat_2_total, year_sale, on=:year)
    transform!(sipri_dat_2_total, 2 => ByRow(x -> ismissing(x) ? 0 : x), renamecols=false)
    sort!(sipri_dat_2_total, :year)

    transform!(sipri_dat_2_total, [2, 3] => ByRow((x, y) -> x / y) => :individual_ratio)

    plot(sipri_dat_2_total.year, sipri_dat_2_total.individual_ratio, c=:black, label=:none)
    scatter!(sipri_dat_2_total.year, sipri_dat_2_total.individual_ratio, ms=2, c=:black, label=:none)
    p_result = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020)

    return p_result
end

# Gözlemler + dönemlik OLS regresyon trendleri
function reg_drawer(country)
    sipri_dat_2 = CSV.read("trade-register.csv", DataFrame)
    select!(sipri_dat_2, [2, 3, 15])
    year_sale = combine(groupby(sipri_dat_2, 2), 3 => sum)
    sipri_dat_2_total = subset(sipri_dat_2, 1 => ByRow(x -> x == country))
    sipri_dat_2_total = combine(groupby(sipri_dat_2_total, 2), 3 => sum)

    rename!(year_sale, [1, 2] .=> [:year, :total_sale])
    rename!(sipri_dat_2_total, [1, 2] .=> [:year, :individual_sale])

    sipri_dat_2_total = outerjoin(sipri_dat_2_total, year_sale, on=:year)
    transform!(sipri_dat_2_total, 2 => ByRow(x -> ismissing(x) ? 0 : x), renamecols=false)
    sort!(sipri_dat_2_total, :year)

    transform!(sipri_dat_2_total, [2, 3] => ByRow((x, y) -> x / y) => :individual_ratio)

    sipri_dat_2_total_coldwar = subset(sipri_dat_2_total, :year => ByRow(x -> x <= 1990))
    sipri_dat_2_total_postcoldwar = subset(sipri_dat_2_total, :year => ByRow(x -> x >= 1990))

    ratio_coefs_cw = coef(lm(@formula(individual_ratio ~ year), sipri_dat_2_total_coldwar))
    ratio_coefs_postcw = coef(lm(@formula(individual_ratio ~ year), sipri_dat_2_total_postcoldwar))

    sipri_postcoldwar_total = DataFrame(Year=sipri_dat_2_total_postcoldwar.year)
    sipri_coldwar_total = DataFrame(Year=sipri_dat_2_total_coldwar.year)

    sipri_postcoldwar_total[!, :predicted] = ratio_coefs_postcw[1] .+ sipri_dat_2_total_postcoldwar.year .* ratio_coefs_postcw[2]
    sipri_coldwar_total[!, :predicted] = ratio_coefs_cw[1] .+ sipri_dat_2_total_coldwar.year .* ratio_coefs_cw[2]

    plot(sipri_dat_2_total.year, sipri_dat_2_total.individual_ratio, c=:black, label=:none)
    scatter!(sipri_dat_2_total.year, sipri_dat_2_total.individual_ratio, ms=2, c=:black, label=:none)
    plot!(sipri_coldwar_total.Year, sipri_coldwar_total.predicted, label=:none, c=:firebrick)
    plot!(sipri_postcoldwar_total.Year, sipri_postcoldwar_total.predicted, label=:none, c=:navyblue)
    p_result = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020)

    return p_result
end

# Gözlemler + LOESS eğrisi; year_sale 2024 kısmi verisini dışarıda bırakmak için filtreleniyor
function loess_drawer(country, s=0.66)
    sipri_dat_2 = CSV.read("trade-register.csv", DataFrame)
    select!(sipri_dat_2, [2, 3, 15])
    year_sale = subset(combine(groupby(sipri_dat_2, 2), 3 => sum), 1 => ByRow(x -> x <= 2023))
    sipri_dat_2_total = subset(sipri_dat_2, 1 => ByRow(x -> x == country))
    sipri_dat_2_total = combine(groupby(sipri_dat_2_total, 2), 3 => sum)

    rename!(year_sale, [1, 2] .=> [:year, :total_sale])
    rename!(sipri_dat_2_total, [1, 2] .=> [:year, :individual_sale])

    sipri_dat_2_total = outerjoin(sipri_dat_2_total, year_sale, on=:year)
    transform!(sipri_dat_2_total, 2 => ByRow(x -> ismissing(x) ? 0 : x), renamecols=false)
    sort!(sipri_dat_2_total, :year)

    transform!(sipri_dat_2_total, [2, 3] => ByRow((x, y) -> x / y) => :individual_ratio)

    loess_mod = loess(sipri_dat_2_total.year, sipri_dat_2_total.individual_ratio, span=s)

    cw_idx = findfirst(==(1990), sipri_dat_2_total.year)
    plot(sipri_dat_2_total.year, sipri_dat_2_total.individual_ratio, c=:black, label=:none)
    scatter!(sipri_dat_2_total.year, sipri_dat_2_total.individual_ratio, ms=2, c=:black, label=:none)
    plot!(sipri_dat_2_total.year[1:cw_idx], predict(loess_mod, sipri_dat_2_total.year[1:cw_idx]), c=:firebrick, label=:none)
    plot!(sipri_dat_2_total.year[cw_idx:end], predict(loess_mod, sipri_dat_2_total.year[cw_idx:end]), c=:navyblue, label=:none)
    p_result = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020)

    return p_result
end

# Tüm yıllar için LOESS tahminlerini döndürür — p9 karşılaştırma grafiği için
function loess_predictor(country, s=0.66)
    sipri_dat_2 = CSV.read("trade-register.csv", DataFrame)
    select!(sipri_dat_2, [2, 3, 15])
    year_sale = subset(combine(groupby(sipri_dat_2, 2), 3 => sum), 1 => ByRow(x -> x <= 2023))
    sipri_dat_2_total = subset(sipri_dat_2, 1 => ByRow(x -> x == country))
    sipri_dat_2_total = combine(groupby(sipri_dat_2_total, 2), 3 => sum)

    rename!(year_sale, [1, 2] .=> [:year, :total_sale])
    rename!(sipri_dat_2_total, [1, 2] .=> [:year, :individual_sale])

    sipri_dat_2_total = outerjoin(sipri_dat_2_total, year_sale, on=:year)
    transform!(sipri_dat_2_total, 2 => ByRow(x -> ismissing(x) ? 0 : x), renamecols=false)
    sort!(sipri_dat_2_total, :year)

    transform!(sipri_dat_2_total, [2, 3] => ByRow((x, y) -> x / y) => :individual_ratio)

    loess_mod = loess(sipri_dat_2_total.year, sipri_dat_2_total.individual_ratio, span=s)

    return predict(loess_mod, sipri_dat_2_total.year)
end

using ColorSchemes

color_pal = cgrad(:inferno, 6, categorical=true)
vis_span = 0.66

# =============================================================================
# BÖLÜM 2.6 — Diğer Büyük Tedarikçiler (Fransa, Almanya, İngiltere, İtalya, Çin)
# =============================================================================
plot(1948:2023, loess_predictor("France", vis_span), label="Fransa", c=color_pal[1])
plot!(1948:2023, loess_predictor("Germany", vis_span), label="Almanya", c=color_pal[2])
plot!(1948:2023, loess_predictor("United Kingdom", vis_span), label="Birleşik Kırallık", c=color_pal[3])
plot!(1948:2023, loess_predictor("Italy", vis_span), label="İtalya", c=color_pal[4])
plot!(1948:2023, loess_predictor("China", vis_span), label="Çin", c=color_pal[5])
p9 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(0.0:0.05:0.25, ["%0", "%5", "%10", "%15", "%20", "%25"]), ylabel="Menşeine Göre Tedarik Edilen Silahların Oranı")

# savefig(p9, "p9.pdf")


# =============================================================================
# BÖLÜM 3: Ülke Başına Ortalama Tedarikçi Sayısı (Çeşitlenme Analizi)
# Sütunlar: [1]=Recipient, [2]=Supplier, [3]=Year
# =============================================================================

sipri_dat_3 = CSV.read("trade-register.csv", DataFrame)

select!(sipri_dat_3, 1:3)
# Bilinmeyen tedarikçiler analizden çıkarılıyor
subset!(sipri_dat_3, 2 => ByRow(x -> x != "unknown supplier(s)"))

# Adım 1: (Recipient, Year) bazında unique satırlar → her (alıcı, yıl, tedarikçi) tekli
# combine(..., unique): her grup içindeki tekrar eden satırları siler
# combine sonrası sütun sırası: [Recipient, Year, Supplier] (groupby anahtarları öne geçer)
sipri_dat_3_unique = combine(groupby(sipri_dat_3, [1, 3]), unique)

# Adım 2: (Recipient, Year) bazında tedarikçi sayısı → nrow = o yıl kaç farklı tedarikçi var
sipri_dat_3_unique2 = combine(groupby(sipri_dat_3_unique, [1, 2]), nrow)

# Adım 3: yıl bazında alıcı başına ortalama tedarikçi sayısı
sipri_dat_3_final = combine(groupby(sipri_dat_3_unique2, 2), :nrow => mean)

rename!(sipri_dat_3_final, [1, 2] .=> ["year", "avg_supp"])

# 2020 sonrası veri kalitesi düşük olduğundan kapsam dışı bırakılıyor
subset!(sipri_dat_3_final, :year => ByRow(x -> x <= 2020))

plot(sipri_dat_3_final.year, sipri_dat_3_final.avg_supp)  # ön bakış grafiği

# Soğuk Savaş / Sonrası ayrımı
sipri_dat_3_final_coldwar = subset(sipri_dat_3_final, :year => ByRow(x -> x <= 1990))
sipri_dat_3_final_postcoldwar = subset(sipri_dat_3_final, :year => ByRow(x -> x >= 1990))

ratio_coefs_cw = coef(lm(@formula(avg_supp ~ year), sipri_dat_3_final_coldwar))
ratio_coefs_postcw = coef(lm(@formula(avg_supp ~ year), sipri_dat_3_final_postcoldwar))

sipri_dat_3_final_postcoldwar[!, :predicted] = ratio_coefs_postcw[1] .+ sipri_dat_3_final_postcoldwar.year .* ratio_coefs_postcw[2]
sipri_dat_3_final_coldwar[!, :predicted] = ratio_coefs_cw[1] .+ sipri_dat_3_final_coldwar.year .* ratio_coefs_cw[2]

plot(sipri_dat_3_final.year, sipri_dat_3_final.avg_supp, c=:black, label=:none)
scatter!(sipri_dat_3_final.year, sipri_dat_3_final.avg_supp, ms=2, c=:black, label=:none)
plot!(sipri_dat_3_final_coldwar.year, sipri_dat_3_final_coldwar.predicted, label=:none, c=:firebrick)
plot!(sipri_dat_3_final_postcoldwar.year, sipri_dat_3_final_postcoldwar.predicted, label=:none, c=:navyblue)
p10 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, ylabel="Ülke Başına Ortalama Tedarikçi Sayısı")

# RDD için merkez ve kukla değişkeni
sipri_dat_3_final[!, :year_centered] = sipri_dat_3_final.year .- cutoff
sipri_dat_3_final[!, :D] = [val >= cutoff ? 1 : 0 for val in sipri_dat_3_final.year]
rdd_model = lm(@formula(avg_supp ~ D * year_centered), sipri_dat_3_final)

# HAC standart hatalar ve regresyon tablosu
vcov_hc = vcov(Bartlett{NeweyWest}(), rdd_model)
RegressionTables.regtable(rdd_model, vcov=vcov_hc, digits=5, render=LatexTable(), regression_statistics=[Nobs, R2])

t_scores = coef(rdd_model) ./ sqrt.(diag(vcov_hc))
p_values(t_scores, Int64(nobs(rdd_model)) - 4)






# LOESS eğrisi (span=0.5 — diğer serilere göre daha dar bant, veri 1948–2020 arası)
loess_mod = loess(sipri_dat_3_final.year, sipri_dat_3_final.avg_supp, span=0.5)


cw_idx3 = findfirst(==(cutoff), sipri_dat_3_final.year)

# LOESS grafik — sadece LOESS eğrileri
plot(sipri_dat_3_final.year, sipri_dat_3_final.avg_supp, c=:black, label=:none)
scatter!(sipri_dat_3_final.year, sipri_dat_3_final.avg_supp, c=:black, ms=2, label=:none)
plot!(sipri_dat_3_final.year[1:cw_idx3], predict(loess_mod, sipri_dat_3_final.year[1:cw_idx3]), c=:firebrick, label=:none)
plot!(sipri_dat_3_final.year[cw_idx3:end], predict(loess_mod, sipri_dat_3_final.year[cw_idx3:end]), c=:navyblue, label=:none)
p11 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, ylabel="Ülke Başına Ortalama Tedarikçi Sayısı")

# Kombine grafik: regresyon trendleri (düz) + LOESS (noktalı, yarı saydam)
plot(sipri_dat_3_final.year, sipri_dat_3_final.avg_supp, c=:black, label=:none)
scatter!(sipri_dat_3_final.year, sipri_dat_3_final.avg_supp, c=:black, ms=2, label=:none)
plot!(sipri_dat_3_final_coldwar.year, sipri_dat_3_final_coldwar.predicted, label=:none, c=:firebrick)
plot!(sipri_dat_3_final_postcoldwar.year, sipri_dat_3_final_postcoldwar.predicted, label=:none, c=:navyblue)
plot!(sipri_dat_3_final.year[1:cw_idx3], predict(loess_mod, sipri_dat_3_final.year[1:cw_idx3]), c=:firebrick, label=:none, alpha=0.45, linestyle=:dot)
plot!(sipri_dat_3_final.year[cw_idx3:end], predict(loess_mod, sipri_dat_3_final.year[cw_idx3:end]), c=:navyblue, label=:none, alpha=0.45, linestyle=:dot)
p7_comb = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, ylabel="Ülke Başına Ortalama Tedarikçi Sayısı")

# savefig(p7_comb, "p7_comb.pdf")




# =============================================================================
# BÖLÜM 3.1: Orta Doğu İçi (Bölgesel) Silah Tedariği Oranı
# =============================================================================

all_recipients = unique(sipri_dat.Recipient)

sipri_dat_new = select(
    subset(subset(sipri_dat,
            [1, 2, 3] => ByRow((x, y, z) -> !(x == "Syria" && y == "Libya" && z == 1978))),
        12 => ByRow(x -> x == "New")),
    [1, 2, 3, 15])

# Alıcı bölge içindeyse Regional=1 (TIV çarpımı için), değilse 0
transform!(sipri_dat_new, 2 => ByRow(x -> x in all_recipients ? 1 : 0) => :Regional)
# RegionalTIV = TIV * Regional (bölge dışı tedarikler 0 olur)
transform!(sipri_dat_new, [4, 5] => ByRow((x, y) -> x * y) => :RegionalTIV)

# Yıl bazında bölgesel toplam TIV
sipri_dat_new_end = combine(groupby(sipri_dat_new, 3), 6 => sum)
rename!(sipri_dat_new_end, [1, 2] .=> ["Year", "RegionalTIV"])

sipri_dat_new_end_postcw = subset(sipri_dat_new_end, :Year => ByRow(x -> x >= 1990))
sipri_dat_new_end_cw = subset(sipri_dat_new_end, :Year => ByRow(x -> x <= 1990))

coef_cw = coef(lm(@formula(RegionalTIV ~ Year), sipri_dat_new_end_cw))
coef_postcw = coef(lm(@formula(RegionalTIV ~ Year), sipri_dat_new_end_postcw))

sipri_dat_new_end_postcw[!, :predicted] = coef_postcw[1] .+ sipri_dat_new_end_postcw.Year .* coef_postcw[2]
sipri_dat_new_end_cw[!, :predicted] = coef_cw[1] .+ sipri_dat_new_end_cw.Year .* coef_cw[2]

loess_mod = loess(sipri_dat_new_end.Year, sipri_dat_new_end.RegionalTIV, span=0.825)


# Bölgesel TIV oranı: RegionalTIV / toplam TIV (yıl bazında)
sipri_dat_new_end2 = transform(
    combine(groupby(sipri_dat_new, 3), [4, 6] .=> sum),
    [2, 3] => ByRow((x, y) -> y / x))

rename!(select!(sipri_dat_new_end2, [1, 4]), [1, 2] .=> ["Year", "RegionalTIVRatio"])

# Oran log-dönüşümü; log(0) = -Inf olan yıllar minimum gözlemlenen log değeriyle doldurulur
log_vals = log.(sipri_dat_new_end2.RegionalTIVRatio)
sipri_dat_new_end2[!, :RegionalTIVRatio] = map(v -> isinf(v) ? minimum(filter(!isinf, log_vals)) : v, log_vals)

sipri_dat_new_end2_postcw = subset(sipri_dat_new_end2, :Year => ByRow(x -> x >= 1990))
sipri_dat_new_end2_cw = subset(sipri_dat_new_end2, :Year => ByRow(x -> x <= 1990))

coef_cw = coef(lm(@formula(RegionalTIVRatio ~ Year), sipri_dat_new_end2_cw))
coef_postcw = coef(lm(@formula(RegionalTIVRatio ~ Year), sipri_dat_new_end2_postcw))

sipri_dat_new_end2_postcw[!, :predicted] = coef_postcw[1] .+ sipri_dat_new_end2_postcw.Year .* coef_postcw[2]
sipri_dat_new_end2_cw[!, :predicted] = coef_cw[1] .+ sipri_dat_new_end2_cw.Year .* coef_cw[2]

plot(sipri_dat_new_end2.Year, sipri_dat_new_end2.RegionalTIVRatio, c=:black, label=:none)
scatter!(sipri_dat_new_end2.Year, sipri_dat_new_end2.RegionalTIVRatio, c=:black, ms=2, label=:none)
plot!(sipri_dat_new_end2_cw.Year, sipri_dat_new_end2_cw.predicted, label=:none, c=:firebrick)
plot!(sipri_dat_new_end2_postcw.Year, sipri_dat_new_end2_postcw.predicted, label=:none, c=:navyblue)
p12 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(-8:2:-2, "%" .* string.(round.(exp.(collect(-8:2:-2)), digits=4))), ylabel="Bölge İçi Tedariğin Oranı")




sipri_dat_new_end2[!, :Year_centered] = sipri_dat_new_end2.Year .- cutoff
sipri_dat_new_end2[!, :D] = [val >= cutoff ? 1 : 0 for val in sipri_dat_new_end2.Year]
rdd_model = lm(@formula(RegionalTIVRatio ~ D * Year_centered), sipri_dat_new_end2)

vcov_hc = vcov(Bartlett{NeweyWest}(), rdd_model)
RegressionTables.regtable(rdd_model, vcov=vcov_hc, digits=5, render=LatexTable(), regression_statistics=[Nobs, R2])

t_scores = coef(rdd_model) ./ sqrt.(diag(vcov_hc))
p_values(t_scores, Int64(nobs(rdd_model)) - 4)


loess_mod = loess(sipri_dat_new_end2.Year, sipri_dat_new_end2.RegionalTIVRatio, span=0.625)


cw_idx_end2 = findfirst(==(cutoff), sipri_dat_new_end2.Year)
plot(sipri_dat_new_end2.Year, sipri_dat_new_end2.RegionalTIVRatio, c=:black, label=:none)
scatter!(sipri_dat_new_end2.Year, sipri_dat_new_end2.RegionalTIVRatio, c=:black, ms=2, label=:none)
plot!(sipri_dat_new_end2.Year[1:cw_idx_end2], predict(loess_mod, sipri_dat_new_end2.Year[1:cw_idx_end2]), c=:firebrick, label=:none)
plot!(sipri_dat_new_end2.Year[cw_idx_end2:end], predict(loess_mod, sipri_dat_new_end2.Year[cw_idx_end2:end]), c=:navyblue, label=:none)
p13 = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(-8:2:-2, "%" .* string.(round.(exp.(collect(-8:2:-2)), digits=4))), ylabel="Bölge İçi Tedarik Oranı")

# Kombine grafik: lineer regresyon + LOESS katmanlari; cw_idx_end2 yukarida hesaplandi
plot(sipri_dat_new_end2.Year, sipri_dat_new_end2.RegionalTIVRatio, c=:black, label=:none)
scatter!(sipri_dat_new_end2.Year, sipri_dat_new_end2.RegionalTIVRatio, c=:black, ms=2, label=:none)
plot!(sipri_dat_new_end2_cw.Year, sipri_dat_new_end2_cw.predicted, label=:none, c=:firebrick)
plot!(sipri_dat_new_end2_postcw.Year, sipri_dat_new_end2_postcw.predicted, label=:none, c=:navyblue)
plot!(sipri_dat_new_end2.Year[1:cw_idx_end2], predict(loess_mod, sipri_dat_new_end2.Year[1:cw_idx_end2]), c=:firebrick, label=:none, alpha=0.45, linestyle=:dot)
plot!(sipri_dat_new_end2.Year[cw_idx_end2:end], predict(loess_mod, sipri_dat_new_end2.Year[cw_idx_end2:end]), c=:navyblue, label=:none, alpha=0.45, linestyle=:dot)
p8_comb = Plots.vline!([1990], c=:black, linestyle=:dash, label="Soğuk Savaş'ın Sonu", xticks=1950:10:2020, yticks=(-8:2:-2, "%" .* string.(round.(exp.(collect(-8:2:-2)) * 100, digits=2))), ylabel="Bölge İçi Tedarik Oranı")

# savefig(p8_comb, "p8_comb.pdf")