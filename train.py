from ultralytics import YOLO

'''
STRUKTURA FOLDEROW (podział 80/20)

dataset/
├── images/
│   ├── train/ (tu 80 zdjęć)
│   └── val/   (tu 20 zdjęć)
└── labels/
    ├── train/ (tu 80 plików .txt)
    └── val/   (tu 20 plików .txt)
'''
def main():
    #Zastosowano Transfer Learning - zbyt mała baza danych aby od początku stawiać wlasny model
    model = YOLO("yolov8n.pt") #wersja nano

    # Uruchomienie procesu szkolenia - potem dodać augmentacje danych
    results = model.train(
        data="data.yaml",    # Ścieżka do stworzonego wcześniej pliku YAML
        epochs=25,          # Liczba epok - liczba przejść modelu przez grupę zdjęć
        imgsz=640,          # Rozmiar obrazu
        batch=8,            # Rozmiar paczki - ile zdjęć na jedną grupę/krok
        device="0",          # szkolenie na karcie (jeżeli nie działa to sprawdzić wersję biblioteki lub szkolić na cpu ('cpu'))
        workers=2,           # Liczba wątków procesora do ładowania danych
        plots=True           # Wygeneruj wykresy skuteczności - do sprawozdania konieczne, aby sprawdzić jak działa
    )
    
    '''
    TO DO:
    - zwiększyć rozmiar obrazu,
    - DODAĆ AUGMENTACJĘ,
    - zwiększyć batch size,
    - zmienić model z nano na inny (np. small)
    '''
    print("Zakończono trening")

if __name__ == "__main__":
    main()