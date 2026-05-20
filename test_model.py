from ultralytics import YOLO

# 1. Ładujemy Wasz model
model = YOLO("runs/detect/train3/weights/best.pt")

# 2. Definiujemy listę plików do przetestowania
# Możecie tu wpisać same nazwy (jeśli są w tym samym folderze) albo pełne ścieżki
pliki_testowe = ["test1.jpg", "test2.jpg", "test3.jpg"]

print(f"Rozpoczynam testowanie {len(pliki_testowe)} plików...")

# 3. Pętla przechodząca przez każde zdjęcie z listy
for sciezka_do_pliku in pliki_testowe:
    try:
        print(f"Testuję plik: {sciezka_do_pliku}")
        
        # Wykonujemy predykcję
        model.predict(
            source=sciezka_do_pliku, 
            save=True,       # Zapisuje obrazek z ramką na dysku
            conf=0.5,        # Pewność modelu min. 50%
            project="testy_kaucji", # Wyniki wpadną do folderu testy_kaucji/predict/
            exist_ok=True    # Nie twórz za każdym razem nowego folderu (predict2, predict3...), tylko wrzucaj do jednego
        )
        
    except Exception as e:
        print(f"❌ Błąd przy przetwarzaniu pliku {sciezka_do_pliku}: {e}")

print("\nWszystkie testy zakończone! Wyniki znajdziecie w folderze: testy_kaucji/predict/")