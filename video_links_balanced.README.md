# `video_links_balanced.csv` — dengeleme listesi

`video_links.csv` ile **birebir aynı 7 kolon** (`url,channel,start,end,pipeline,reporter,completed`).

**Bu satırlar `video_links.csv`'ye eklendi** (`scripts/merge_links.py` ile) — ana CSV
artık 4699 yerine **6602 satır**, 336 `completed=1` bayrağı korunarak. Bu dosya
kaynak liste olarak duruyor; merge idempotent olduğu için tekrar çalıştırmak zararsız.

- **1903 satır**, hepsi `pipeline=2`, hepsi `completed` boş
- **18 sunucu**, `cuneyt_ozdemir` **yok**
- **~661 saat** kaynak video → **~274 saat** beklenen klip

---

## 0. ÖNEMLİ — çalışan run bu dosyayı geri alabilir

`run_all.mark_completed`, her tamamlanan satırda `video_links.csv`'yi **başlangıçta
okuduğu listeden baştan yazıyor** (`run_all.py:83`). Yani bir run açıkken merge
yapılırsa, o run'ın bir sonraki tamamlanan satırında yeni satırlar sessizce
kayboluyor — ve zaten çalışan process yeni satırları görmüyor, çünkü CSV'yi sadece
başlangıçta okuyor.

Merge yapıldığı sırada bir run açıktı. Yapılacak:

```powershell
# 1. çalışan run'ı durdur (Ctrl+C) — resumable, sadece o anki video kaybolur
# 2. satırlar hâlâ yerinde mi diye bak / gerekirse geri koy:
uv run python scripts/merge_links.py video_links_balanced.csv
# 3. yeniden başlat
uv run python scripts/run_all.py --exclude-reporters cuneyt_ozdemir
```

`merge_links.py` idempotent: var olan satırlara ve `completed` bayraklarına
dokunmuyor, sadece eksik olanları ekliyor. Ne zaman çalıştırırsan çalıştır doğru
sonucu verir, arada ilerleyen cüneyt satırlarını geri almaz.

## 0.5. Tablodaki "beklenen klip sa" artık düşük tahmin

Tablo, satırlar seçilirken yürürlükte olan filtreye göre hesaplandı. O filtre bir
sahnedeki örneklerden biri 2 yüz görünce **tüm sahneyi** eliyordu ve kesme içermeyen
yayın tek sahne olduğu için, 35 dakikalık bir monolog tek bir kötü kare yüzünden
sıfır klip veriyordu. `solo_segments` artık sahneyi 30 sn'lik hücrelere bölüp sadece
kötü hücreleri atıyor (CLAUDE.md invariant #2).

Ölçülen fark, `deniz_zeyrek 8mWUgIuzNOM` üzerinde: **0 klip / 0 dk → 3 klip / 33.7 dk.**
Yani özellikle uzun tek-plan formatlı sunucularda (rusen_cakir, sabahattin_onkibar,
levent_gultekin, yilmaz_ozdil, deniz_zeyrek, fatih_altayli, nagehan_alci,
ozlem_gurses) gerçek çıktı tablodakinden belirgin şekilde yüksek olacak.

## 1. Seçim kriteri: süre değil, **tek yüz oranı**

Mevcut `data/reporter_clips.csv` üzerinden ölçülen gerçek verim:

| Sunucu | Kaynak/video | Kullanılabilir oran |
|---|---|---|
| fatih_portakal | 7.3 dk | **%74.6** |
| gulsah_ekinci | 100.9 dk | %13.5 |
| ismail_kucukkaya | 134.3 dk | %13.2 |
| kubra_par | 107.6 dk | %10.1 |
| ece_uner | 105.9 dk | %5.3 |
| cem_ogretir | 46.2 dk | %2.5 |

Belirleyici olan videonun uzunluğu değil, **ekranda tek kişinin olup olmadığı**.
"Ana Haber" bültenleri uzun oldukları için değil, sürekli B-roll / muhabir / iki
kişilik split ekran içerdikleri için %5–13'te kalıyor. Bu yüzden listedeki kanallar
ağırlıklı olarak **kameraya tek başına konuşan yorum kanalları**.

Her aday kanal için 1080p'lik ~70 saniyelik kesitler indirilip 10 kareye HOG yüz
tespiti çalıştırıldı; tablodaki "tek yüz" sütunu bu ölçümün sonucu.

## 2. Liste

| reporter | kanal | çizgi | tek yüz | satır | kaynak sa | beklenen klip sa |
|---|---|---|---|---|---|---|
| turgay_guler | Turgay Güler | iktidar | 21/30 | 340 | 58.0 | 30.2 |
| hadi_ozisik | Hadi Özışık | iktidar | 20/30 | 320 | 55.6 | 26.7 |
| osman_gokcek | Osman Gökçek | iktidar | 12/30 | 240 | 47.6 | 13.8 |
| nagehan_alci | Nagehan Alçı | iktidar | 24/30 | 74 | 31.3 | 18.1 |
| ersan_sen | Ersan Şen | iktidar | 10/10 | 25 | 5.4 | 3.5 |
| can_okanar | A Haber | iktidar | ölçüm %12.7 | 25 | 29.4 | 3.8 |
| sinan_burhan | Sinan Burhan | iktidar | 10/20 | 19 | 3.3 | 1.2 |
| rusen_cakir | Medyascope | merkez | 29/30 | 110 | 28.3 | 19.8 |
| deniz_zeyrek | Deniz Zeyrek | merkez | 10/10 | 55 | 30.7 | 21.5 |
| enver_aysever | Enver Aysever | muhalefet | 20/30 | 180 | 32.2 | 15.5 |
| fatih_portakal | Fatih Portakal TV | muhalefet | ölçüm %74.6 | 170 | 24.6 | 18.4 |
| sabahattin_onkibar | Sabahattin Önkibar | muhalefet | 20/20 | 110 | 28.5 | 20.0 |
| **ece_uner** | Halk TV | muhalefet | ölçüm %5.3 | 45 | 84.3 | 4.5 |
| levent_gultekin | Levent Gültekin | muhalefet | 10/10 | 40 | 26.5 | 18.5 |
| yilmaz_ozdil | Yılmaz Özdil | muhalefet | 27/30 | 40 | 25.5 | 16.6 |
| **ismail_kucukkaya** | Halk TV + Sözcü TV | muhalefet | ölçüm %13.2 | 40 | 105.9 | 14.0 |
| fatih_altayli | Fatih Altaylı | muhalefet | 20/20 | 35 | 30.5 | 21.4 |
| ozlem_gurses | Özlem Gürses | muhalefet | 19/30 | 35 | 14.4 | 6.5 |

Beklenen klip saatine göre denge: **iktidar %35 · merkez %15 · muhalefet %50**.
Tam eşitlik çıkmadı çünkü iktidara yakın yorumcuların YouTube'da tek kişilik kanal
açma alışkanlığı belirgin şekilde daha az; kurumsal kanallar (TGRT, Ülke TV, 24 TV,
A Haber) ise ya kısa haber klibi ya da 2–6 kişilik panel yayınlıyor — ikisi de bu
filtreye uygun değil.

**Not:** "çizgi" sütunu sadece veri setini dengelemek için kabaca gruplama; kesin bir
siyasi etiket iddiası değil. CSV'de böyle bir kolon yok, yalnızca bu dosyada.

### ece_uner / ismail_kucukkaya hakkında

İkisi de korpusun **pahalı ucu** — 190 saat kaynak karşılığında ~18.5 saat klip.
Video başına düşen çıktıya bakınca ayrışıyorlar:

- **ismail_kucukkaya** iyi: ~159 dk kaynaktan ~21 dk klip, veri setindeki en yüksek
  video-başı çıktı. 9 videosu vardı, 49'a çıktı. Yarısı eski Halk TV "Yeni Bir Sabah",
  yarısı şimdiki Sözcü TV "Günaydın Türkiyem" — bu sayede **aynı konuşmacı iki
  kanalda**, kanal-bağımsız test split'i için işe yarar.
- **ece_uner** pahalı: ~112 dk kaynaktan ~6 dk klip, veri setindeki en düşük ikinci
  verim. 15 videosu vardı, 61'e çıktı. Daha fazlası orantısız olurdu; kesmek
  istersen ilk budanacak yer burası.

Kırpma yapılamadı: `run_all.run_pipeline2` `select_clips`'e sadece `url` geçiyor,
`start`/`end` kolonları **pipeline 2'de yok sayılıyor** (sadece pipeline 1 kullanıyor).
Bu 2–3 saatlik yayınların yoğun kısmını kesebilmek için pipeline 2'nin de time-range
desteklemesi gerekir — ayrı bir iş.

## 3. Satır sırası — baştan kestiğinde de dengeli

Satırlar sunucular arasında **beklenen klip dakikasına göre** dönüşümlü diziliyor:
her adımda o ana kadar en az klip dakikası biriken sunucudan bir video ekleniyor.
Düz round-robin olsaydı bir turda 52 dakikalık Altaylı videosu ile 8 dakikalık
Portakal videosu yan yana gelir, satır sayısı dengeli görünürken **dakika** bazında
bozulurdu.

Pratik sonucu: bu dosyanın ilk N satırını alsan da denge korunur. (Ana CSV'ye
eklendiğinde sıra korunuyor ama `run_all` zaten pending satırları baştan sona
işliyor, o yüzden `--limit` ile kesersen de dengeli bir alt küme alırsın.)

## 4. Çalıştırmadan önce

1. **Referans fotoğraflar hazır.** 14 yeni sunucu için
   `pipeline2_reporter_pictures/<slug>.jpeg` üretildi (örnek videolardan, kimlik
   birden fazla videoda tekrar eden yüz kümesiyle doğrulanarak, cepheden ve gözü açık
   kare seçilerek). 23 sunucunun hepsinde fotoğraf mevcut ve
   `load_reference_encoding` ile yükleniyor. Bir sunucudan hiç klip çıkmazsa ilk
   bakılacak yer bu fotoğraf — elle daha iyisiyle değiştir.
2. **`enver_aysever`**: örneklenen 5 videonun 2'si 720p altında. Bunlar
   `DOWNLOAD_FORMAT` yüzünden pending kalır, run'ı bozmaz. Diğer 15 sunucuda
   örneklenen videoların tamamı ≥720p ve canlı yayın değil.

## 5. Elenenler

| Aday | Sebep |
|---|---|
| Nevşin Mengü | 30 karenin 3'ünde tek yüz — format dış ses + arşiv görüntü |
| Ersoy Dede | Kanalda ≥720p kaynak yok, `DOWNLOAD_FORMAT` hiç tutmuyor |
| Cevheri Güven | Kanal erişime kapalı |
| Buket Aydın | Kanal artık astroloji içeriği, haber değil |
| Fuat Uğur | Ağırlıkla eski/düşük çözünürlüklü arşiv |
| Spor programları | %100 Futbol, NEO Spor, Libero TV, beIN — hepsi 2–6 kişilik panel |
| Sabah programları | Müge Anlı, Esra Erol, Didem Arslan — stüdyoda sürekli çok kişi |

Spor tarafında tek kişilik format pratikte yok; listedeki `sinan_burhan` videolarının
bir kısmı futbol yorumu, bu konudaki tek gerçekçi katkı o.

---

Üretim scriptleri oturum scratchpad'inde: `build_csv.py` (seçim + sıralama),
`probe.py` (yüz yoğunluğu ölçümü), `make_refs.py` (referans fotoğraf),
`verify.py` (720p / canlı yayın kontrolü). Repo'ya kalıcı eklenen tek script
`scripts/merge_links.py`.
