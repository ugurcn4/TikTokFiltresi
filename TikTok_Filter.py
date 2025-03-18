import cv2
import numpy as np

def TikTok_Filter(image):
    """
    Hangi Yüz Profilin daha iyi? Hiç düşündünüz mü? Bu filtre ile sol ve sağ yüzünüzü karşılaştırabilirsiniz.
    Komik bir filtre olması için yüzün sol ve sağ yarımlarını ve bunların ayna görüntülerini yan yana gösteren filtre.
    Yüzün sol ve sağ yarımlarını ve bunların ayna görüntülerini yan yana gösteren filtre.
    Sol tarafta sol yarım ve sağ tarafta sağ yarım gösterilir, her biri kendi ayna görüntüsüyle birlikte.

    Args:
        image: İşlenecek görüntü

    Returns:
        Filtrelenmiş görüntü
    """
    # Görüntünün boyutlarını al
    height, width = image.shape[:2]

    # Yüz tespiti için Haar Cascade sınıflandırıcısını yükle
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Gri tonlamalı görüntüye dönüştür (yüz tespiti için)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Sonuç görüntüsünü oluştur (orijinal görüntü boyutunda)
    result = image.copy()

    # Eğer yüz tespit edilirse
    if len(faces) > 0:
        # İlk yüzü al (birden fazla yüz varsa sadece ilkini alacağız)
        x, y, w, h = faces[0]

        # Yüz bölgesini kırp
        face = image[y:y+h, x:x+w]

        # Yüzün orta noktasını bul
        mid_x = w // 2

        # Sol yarım yüzü al
        left_half = face[:, :mid_x].copy()

        # Sağ yarım yüzü al
        right_half = face[:, mid_x:].copy()

        # Sol yarımı yatay olarak aynala
        left_mirror = cv2.flip(left_half, 1)
        # flip ile yatay olarak aynalama yapıyoruz. yatay takla gibi düşünürsek.

        # Sağ yarımı yatay olarak aynala
        right_mirror = cv2.flip(right_half, 1)

        # Boyutları kontrol et ve eşitle
        # Shape ile boyutları kontrol ediyoruz..
        left_h, left_w = left_half.shape[:2]
        right_h, right_w = right_half.shape[:2]

        # Eğer sağ ve sol yarımların boyutları farklıysa
        # Bu kısımda Shape sorunları ile sıkça karşılaştık. Bu yüzden bu kısmı düzeltmek için bir kontrol ekledik.
        if left_w != right_w:
            # İkisini de aynı boyuta getir
            target_width = min(left_w, right_w)
            left_half = cv2.resize(left_half, (target_width, left_h))
            right_half = cv2.resize(right_half, (target_width, right_h))
            left_mirror = cv2.resize(left_mirror, (target_width, left_h))
            right_mirror = cv2.resize(right_mirror, (target_width, right_h))
            # resize ile boyutları eşitleyip sorunu çözüyoruz.
            # ayna görüntülerin ve orijinal görüntülerin boyutlarını eşitleyip sorunu çözüyoruz.

        # Sol taraf: sol yarım + sol yarımın aynası
        left_combined = np.hstack((left_half, left_mirror)) # sol sağ ters çünkü simetrik olması gerekiyor.
        # hstack ile yatay birleştirme yaptık.
        # Sağ taraf: sağ yarımın aynası + sağ yarım
        right_combined = np.hstack((right_mirror, right_half)) # sol sağ ters çünkü simetrik olması gerekiyor.

        # İki tarafı birleştir hstack ile yatay birleştirme yaptık.
        combined = np.hstack((left_combined, right_combined))

        # Birleştirilmiş görüntüyü yeniden boyutlandır (orijinal yüz boyutuna)
        combined = cv2.resize(combined, (w*2, h)) # yükseklikte bir değişiklik yapmadık. genişlik 2x olcak.
        # resize --> yeniden boyutlandırma işlemi yapar.

        # Yüzün genişletilmiş bölgesini hesapla
        new_x = max(0, x - w//2)
        new_width = min(width - new_x, w*2)

        # Arka planı bulanıklaştır
        result = blur_background(image, (new_x, y, w*2, h), blur_amount=15)
        # blur_amount değeri ile bulanıklaştırma miktarını ayarladık.
        # eğer yüz bulunmazsa bulanıklaştırma yapmaz. normal görüntü gösterir. bunu yüz tespiti kısmında yaptım.

        # Birleştirilmiş görüntüyü yerleştir
        if new_width == w*2:  # Tam sığıyorsa
            result[y:y+h, new_x:new_x+new_width] = combined
        else:  # Sığmıyorsa, sığan kısmını yerleştir
            result[y:y+h, new_x:new_x+new_width] = combined[:, :new_width]

        # Renkli çerçeve ekle (sadece sol ve sağ yüzler arasında olacak şekilde )
        result = add_colored_frame(result, (x, y, w, h), (new_x, y, w*2, h))

        # Yazıları ekle (bulanıklaştırma ve çerçeve ekledikten sonra ekleyeceğiz çünkü bunu yapmadığımızda yazılar da bulanıklaşıyor)
        add_text(result, "Sol", (new_x + w//2 - 40, y-10), " <--")
        add_text(result, "Sag", (new_x + w + w//2 - 40, y-10), " -->")

    return result

def add_text(image, text, position, emoji_text=None):
    """
    Görüntüye metin ekler

    Args:
        image: Üzerine yazı eklenecek görüntü
        text: Eklenecek metin
        position: (x, y) konumu
        emoji_text: Emoji yerine kullanılacak metin sembolü (opsiyonel)

    Returns:
        Yazı eklenmiş görüntü
    """
    x, y = position

    # Yazı tipi ve kalınlık
    font = cv2.FONT_HERSHEY_DUPLEX  # Daha net bir yazı tipi
    font_size = 0.8
    thickness = 2

    # Türkçe karakter düzeltmesi
    if text == "Sag":
        text = "Sag"  # OpenCV'de Türkçe karakter desteği sınırlı

    # Yazı için gölge efekti
    cv2.putText(image, text, (x+2, y+2), font, font_size, (0, 0, 0), thickness+1)
    # thickness gölgenin kalınlık değeri thickness+1 ile yazı boyutundan 1 fazla olması için.

    # Ana yazı
    cv2.putText(image, text, (x, y), font, font_size, (255, 255, 255), thickness)

    # Emoji metni ekle (eğer varsa)
    if emoji_text:
        emoji_x = x + len(text) * 10  # Metinden sonra emoji
        cv2.putText(image, emoji_text, (emoji_x, y), font, font_size, (255, 255, 255), thickness)

    return image

def blur_background(image, face_rect, blur_amount=15):
    """
    Yüz dışındaki arka planı bulanıklaştırır

    Args:
        image: Bulanıklaştırılacak görüntü
        face_rect: Yüz bölgesi (x, y, w, h) - şu anda kullanılmıyor
        blur_amount: Bulanıklaştırma miktarı

    Returns:
        Arka planı bulanıklaştırılmış görüntü
    """

    # Tüm görüntüyü bulanıklaştır
    blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    # Bulanık görüntüyü sonuç görüntüsüne kopyala
    result = blurred.copy()

    return result

def add_colored_frame(image, face_rect, expanded_rect):
    """
    Sol ve sağ yüzler arasına renkli çerçeve ekler

    Args:
        image: Çerçeve eklenecek görüntü
        face_rect: Yüz bölgesi (x, y, w, h) - şu anda kullanılmıyor
        expanded_rect: Genişletilmiş yüz bölgesi (x, y, w, h)

    Returns:
        Çerçeve eklenmiş görüntü
    """
    # face_rect parametresi şu anda kullanılmıyor
    ex, ey, ew, eh = expanded_rect

    # Görüntünün bir kopyasını al
    result = image.copy()

    # Orta çizgi için gradyan çerçeve oluştur (sol ve sağ yüzler arasında)
    mid_x = ex + ew // 2

    # Gradyan renkleri
    colors = [
        (255, 100, 100),  # Kırmızımsı
        (100, 100, 255),  # Mavimsi
        (255, 255, 100),  # Sarımsı
        (100, 255, 100),  # Yeşilimsi
    ]

    # Zamanla değişen renk seçimi
    color_index = (cv2.getTickCount() // 10) % len(colors)
    color1 = colors[color_index]

    # Orta çizgi için gradyan çerçeve çiz (3 piksel genişliğinde)
    for i in range(3):
        cv2.line(result, (mid_x-1+i, ey), (mid_x-1+i, ey+eh), color1, 1)

    return result

def main():
    """
    Ana fonksiyon - kamera akışını başlatır ve filtreyi uygular
    """
    # Kamera akışını başlat
    cap = cv2.VideoCapture(0)

    while True:
        # Kameradan bir kare oku
        ret, frame = cap.read()

        if not ret: # kameradan görüntü alınıp alınamadığını kontrol eder eğer alınmazsa kırılacak
            break

        # Filtreyi uygula
        filtered_frame = TikTok_Filter(frame)

        # Sonucu göster
        cv2.imshow('TikTok_Filter', filtered_frame)

        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kaynakları serbest bırak
    cap.release() # bunu yapmazsak kamera açık kalabilir ve başka bir uygulama kullanamayabilir
    cv2.destroyAllWindows() # cv2 tarafından oluşturulan tüm pencereleri kapatir

if __name__ == "__main__":
    main()
