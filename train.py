from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt") 

    results = model.train(
        data="data.yaml",
        
        epochs=150,          
        imgsz=1024,
        batch=32,
        device=0,            
        workers=4,           
        plots=True,
        deterministic=True,  

        degrees=20.0,
        scale=0.5,
        translate=0.1,
        shear=3.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0, 
        
        hsv_h=0.015,         
        hsv_s=0.6,           
        hsv_v=0.4
        
        mosaic=1.0,
        mixup=0.0,
        close_mosaic=15,
        
        lr0=0.01,            
        lrf=0.01,            
        cos_lr=True,         
        weight_decay=0.0005, 
        warmup_epochs=3.0,   
        patience=35
    )
    
    print("Zakończono optymalny trening.")

if __name__ == "__main__":
    main()
