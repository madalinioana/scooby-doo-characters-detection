import os
import shutil
import numpy as np
import cv2
import yaml
from ultralytics import YOLO

# configurari si constante
ROOT_DIR = '/kaggle/input/scooby-doo-characters-detection'
TRAIN_DIR = os.path.join(ROOT_DIR, 'antrenare')
VAL_DIR = os.path.join(ROOT_DIR, 'validare', 'validare')
OUTPUT_DIR = '/kaggle/working/yolo_results'

# structura dataset yolo
DATASET_DIR = '/kaggle/working/yolo_dataset'
IMG_TRAIN = os.path.join(DATASET_DIR, 'images/train')
LBL_TRAIN = os.path.join(DATASET_DIR, 'labels/train')
IMG_VAL = os.path.join(DATASET_DIR, 'images/val')

# definim clasele (inclusiv unknown pentru task 1)
CLASE_YOLO = ['daphne', 'fred', 'shaggy', 'velma', 'unknown']
CLASSES = ['daphne', 'fred', 'shaggy', 'velma']
# CLASSES = CLASE_YOLO[:4]  # alternativa
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASE_YOLO)}

# functii utilitare cutii
def convBox(marime, cutie):
    # conversie (xmin, ymin, xmax, ymax) -> (x_center, y_center, w, h) normalizat
    dw = 1. / marime[0]
    dh = 1. / marime[1]
    x = (cutie[0] + cutie[2]) / 2.0
    y = (cutie[1] + cutie[3]) / 2.0
    w = cutie[2] - cutie[0]
    h = cutie[3] - cutie[1]
    return (x * dw, y * dh, w * dw, h * dh)

# pregatirea datelor
def prepare_data():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    
    os.makedirs(IMG_TRAIN, exist_ok=True)
    os.makedirs(LBL_TRAIN, exist_ok=True)
    os.makedirs(IMG_VAL, exist_ok=True)
    
    print("Indexam datele pentru YOLO...")
    imagini_procesate = set()
    # counter = 0
    
    # citim fisierele de adnotare
    annotation_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('_annotations.txt')]
    # print(f"Fisiere gasite: {len(annotation_files)}")
    
    for annot_file in annotation_files:
        class_name = annot_file.split('_')[0]
        annot_path = os.path.join(TRAIN_DIR, annot_file)
        
        with open(annot_path, 'r') as f:
            linii = f.readlines()
            
        for linie in linii:
            parts = linie.strip().split()
            if len(parts) != 6: 
                continue
            
            img_name = parts[0]
            eticheta = parts[5].lower()
            if eticheta not in CLASE_YOLO: 
                continue
                
            cls_idx = CLASS_TO_IDX[eticheta]
            bbox = tuple(map(float, parts[1:5]))
            
            img_path = os.path.join(TRAIN_DIR, class_name, img_name)
            if not os.path.exists(img_path): 
                continue
            
            imagine = cv2.imread(img_path)
            if imagine is None: 
                continue
            img_h, img_w = imagine.shape[:2]
            
            bbox_yolo = convBox((img_w, img_h), bbox)
            
            # copiem imaginea si cream eticheta
            dest_img = os.path.join(IMG_TRAIN, f"{class_name}_{img_name}")
            if dest_img not in imagini_procesate:
                shutil.copy(img_path, dest_img)
                imagini_procesate.add(dest_img)
                
                # copiem prima imagine si la validare (cerinta yolo)
                if len(imagini_procesate) == 1:
                     shutil.copy(img_path, os.path.join(IMG_VAL, f"{class_name}_{img_name}"))
            
            txt_name = os.path.splitext(os.path.basename(dest_img))[0] + ".txt"
            label_path = os.path.join(LBL_TRAIN, txt_name)
            
            with open(label_path, 'a') as lbl_f:
                lbl_f.write(f"{cls_idx} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")
                
                if len(imagini_procesate) == 1:
                     val_label = os.path.join(DATASET_DIR, 'labels/val', txt_name)
                     os.makedirs(os.path.dirname(val_label), exist_ok=True)
                     with open(val_label, 'a') as f_val:
                         f_val.write(f"{cls_idx} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")

    # creare fisier configurare yaml
    yaml_config = f"""
    path: {DATASET_DIR}
    train: images/train
    val: images/val
    names:
      0: daphne
      1: fred
      2: shaggy
      3: velma
      4: unknown
    """
    with open('data.yaml', 'w') as fisier_yaml:
        fisier_yaml.write(yaml_config)

# antrenare
def antreneaza_model():
    # folosim modelul nano pre-antrenat
    model = YOLO('yolov8n.pt') 
    
    print("Incepem antrenarea (15 epoci)...")
    model.train(data='data.yaml', epochs=15, imgsz=640, plots=False)
    
    # salvam cel mai bun model
    best_path = model.trainer.best
    model_dest = os.path.join(OUTPUT_DIR, 'model_yolo_best.pt')
    shutil.copy(best_path, model_dest)
    print(f"Model salvat la {model_dest}")
    
    return model

# inferenta
def ruleaza_inferenta(model):
    print("Rulare inferenta pe validare...")
    imagini_validare = [f for f in os.listdir(VAL_DIR) if f.endswith('.jpg')]
    
    rezultate_t1 = {'cutii': [], 'scoruri': [], 'fisiere': []}
    rezultate_t2 = {c: {'cutii': [], 'scoruri': [], 'fisiere': []} for c in CLASSES}
    
    for idx, filename in enumerate(imagini_validare):
        img_path = os.path.join(VAL_DIR, filename)
        # num_detections = 0
        
        # predictie cu prag scazut pentru a capta tot
        results = model.predict(img_path, conf=0.01, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        
        for k in range(len(boxes)):
            box = boxes[k]
            scor = scores[k]
            nume_cls = CLASE_YOLO[classes[k]]
            
            # task 1: orice fata
            rezultate_t1['cutii'].append(box)
            rezultate_t1['scoruri'].append(scor)
            rezultate_t1['fisiere'].append(filename)
            
            # task 2: doar personajele tinta
            if nume_cls in CLASSES:
                rezultate_t2[nume_cls]['cutii'].append(box)
                rezultate_t2[nume_cls]['scoruri'].append(scor)
                rezultate_t2[nume_cls]['fisiere'].append(filename)
                
        if idx % 50 == 0: 
            print(f"Procesat {idx}/{len(imagini_validare)}")
            # print(f"  Boxes: {len(boxes)}")

    # salvare rezultate
    os.makedirs(os.path.join(OUTPUT_DIR, 'task1'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'task2'), exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, 'task1', 'detections_all_faces.npy'), 
            np.array(rezultate_t1['cutii'], dtype=object))
    np.save(os.path.join(OUTPUT_DIR, 'task1', 'scores_all_faces.npy'), 
            np.array(rezultate_t1['scoruri']))
    np.save(os.path.join(OUTPUT_DIR, 'task1', 'file_names_all_faces.npy'), 
            np.array(rezultate_t1['fisiere']))
    
    for char_name in CLASSES:
        np.save(os.path.join(OUTPUT_DIR, 'task2', f'detections_{char_name}.npy'), 
                np.array(rezultate_t2[char_name]['cutii'], dtype=object))
        np.save(os.path.join(OUTPUT_DIR, 'task2', f'scores_{char_name}.npy'), 
                np.array(rezultate_t2[char_name]['scoruri']))
        np.save(os.path.join(OUTPUT_DIR, 'task2', f'file_names_{char_name}.npy'), 
                np.array(rezultate_t2[char_name]['fisiere']))

if not os.path.exists(TRAIN_DIR):
    TRAIN_DIR = os.path.join(ROOT_DIR, 'scooby-doo-characters-detection', 'antrenare')
    VAL_DIR = os.path.join(ROOT_DIR, 'scooby-doo-characters-detection', 'validare', 'validare')

prepare_data()
model = antreneaza_model()
ruleaza_inferenta(model)