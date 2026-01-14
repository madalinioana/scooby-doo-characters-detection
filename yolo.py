import os
import shutil
import numpy as np
import cv2
import yaml
from ultralytics import YOLO

# configurari si constante
DIR_RADACINA = '/kaggle/input/scooby-doo-characters-detection'
DIR_ANTRENARE_SURSA = os.path.join(DIR_RADACINA, 'antrenare')
DIR_VALIDARE_SURSA = os.path.join(DIR_RADACINA, 'validare', 'validare')
DIR_IESIRE = '/kaggle/working'

# structura dataset yolo
DIR_DATASET = '/kaggle/working/yolo_dataset'
IMG_TRAIN = os.path.join(DIR_DATASET, 'images/train')
LBL_TRAIN = os.path.join(DIR_DATASET, 'labels/train')
IMG_VAL = os.path.join(DIR_DATASET, 'images/val')

# definim clasele (inclusiv unknown pentru task 1)
CLASE_YOLO = ['daphne', 'fred', 'shaggy', 'velma', 'unknown']
CLASE_TINTA = ['daphne', 'fred', 'shaggy', 'velma']
IDX_CLASE = {cls: i for i, cls in enumerate(CLASE_YOLO)}

# functii utilitare
def converteste_bbox(size, box):
    # conversie (xmin, ymin, xmax, ymax) -> (x_center, y_center, w, h) normalizat
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

# pregatirea datelor
def pregateste_date():
    if os.path.exists(DIR_DATASET):
        shutil.rmtree(DIR_DATASET)
    
    os.makedirs(IMG_TRAIN, exist_ok=True)
    os.makedirs(LBL_TRAIN, exist_ok=True)
    os.makedirs(IMG_VAL, exist_ok=True)
    
    print("Pregatire dataset YOLO...")
    imagini_procesate = set()
    
    # citim fisierele de adnotare
    fisiere_adnotare = [f for f in os.listdir(DIR_ANTRENARE_SURSA) if f.endswith('_annotations.txt')]
    
    for fisier_txt in fisiere_adnotare:
        nume_clasa = fisier_txt.split('_')[0]
        cale_txt = os.path.join(DIR_ANTRENARE_SURSA, fisier_txt)
        
        with open(cale_txt, 'r') as f:
            linii = f.readlines()
            
        for linie in linii:
            p = linie.strip().split()
            if len(p) != 6: continue
            
            img_nume = p[0]
            eticheta = p[5].lower()
            if eticheta not in CLASE_YOLO: continue
                
            cls_id = IDX_CLASE[eticheta]
            box = tuple(map(float, p[1:5]))
            
            cale_img_sursa = os.path.join(DIR_ANTRENARE_SURSA, nume_clasa, img_nume)
            if not os.path.exists(cale_img_sursa): continue
            
            img = cv2.imread(cale_img_sursa)
            if img is None: continue
            h, w = img.shape[:2]
            
            bbox_yolo = converteste_bbox((w, h), box)
            
            # copiem imaginea si cream eticheta
            cale_img_dest = os.path.join(IMG_TRAIN, f"{nume_clasa}_{img_nume}")
            if cale_img_dest not in imagini_procesate:
                shutil.copy(cale_img_sursa, cale_img_dest)
                imagini_procesate.add(cale_img_dest)
                
                # copiem prima imagine si la validare (cerinta yolo)
                if len(imagini_procesate) == 1:
                     shutil.copy(cale_img_sursa, os.path.join(IMG_VAL, f"{nume_clasa}_{img_nume}"))
            
            nume_txt = os.path.splitext(os.path.basename(cale_img_dest))[0] + ".txt"
            cale_lbl = os.path.join(LBL_TRAIN, nume_txt)
            
            with open(cale_lbl, 'a') as f_lbl:
                f_lbl.write(f"{cls_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")
                
                if len(imagini_procesate) == 1:
                     cale_lbl_val = os.path.join(DIR_DATASET, 'labels/val', nume_txt)
                     os.makedirs(os.path.dirname(cale_lbl_val), exist_ok=True)
                     with open(cale_lbl_val, 'a') as f_val:
                         f_val.write(f"{cls_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")

    # creare fisier configurare yaml
    yaml_content = f"""
    path: {DIR_DATASET}
    train: images/train
    val: images/val
    names:
      0: daphne
      1: fred
      2: shaggy
      3: velma
      4: unknown
    """
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)

# antrenare
def antreneaza_model():
    # folosim modelul nano pre-antrenat
    model = YOLO('yolov8n.pt') 
    
    print("Incepem antrenarea (15 epoci)...")
    model.train(data='data.yaml', epochs=15, imgsz=640, plots=False)
    
    # salvam cel mai bun model
    path_best = model.trainer.best
    dest_model = os.path.join(DIR_IESIRE, 'model_yolo_best.pt')
    shutil.copy(path_best, dest_model)
    print(f"Model salvat la {dest_model}")
    
    return model

# inferenta
def ruleaza_inferenta(model):
    print("Rulare inferenta pe validare...")
    imagini_validare = [f for f in os.listdir(DIR_VALIDARE_SURSA) if f.endswith('.jpg')]
    
    rez_t1 = {'cutii': [], 'scoruri': [], 'fisiere': []}
    rez_t2 = {c: {'cutii': [], 'scoruri': [], 'fisiere': []} for c in CLASE_TINTA}
    
    for i, nume_fisier in enumerate(imagini_validare):
        cale_img = os.path.join(DIR_VALIDARE_SURSA, nume_fisier)
        
        # predictie cu prag scazut pentru a capta tot
        results = model.predict(cale_img, conf=0.01, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        
        for k in range(len(boxes)):
            box = boxes[k]
            score = scores[k]
            nume_clasa = CLASE_YOLO[classes[k]]
            
            # task 1: orice fata
            rez_t1['cutii'].append(box)
            rez_t1['scoruri'].append(score)
            rez_t1['fisiere'].append(nume_fisier)
            
            # task 2: doar personajele tinta
            if nume_clasa in CLASE_TINTA:
                rez_t2[nume_clasa]['cutii'].append(box)
                rez_t2[nume_clasa]['scoruri'].append(score)
                rez_t2[nume_clasa]['fisiere'].append(nume_fisier)
                
        if i % 50 == 0: print(f"Procesat {i}/{len(imagini_validare)}")

    # salvare rezultate
    os.makedirs(os.path.join(DIR_IESIRE, 'task1'), exist_ok=True)
    os.makedirs(os.path.join(DIR_IESIRE, 'task2'), exist_ok=True)
    
    np.save(os.path.join(DIR_IESIRE, 'task1', 'detections_all_faces.npy'), np.array(rez_t1['cutii'], dtype=object))
    np.save(os.path.join(DIR_IESIRE, 'task1', 'scores_all_faces.npy'), np.array(rez_t1['scoruri']))
    np.save(os.path.join(DIR_IESIRE, 'task1', 'file_names_all_faces.npy'), np.array(rez_t1['fisiere']))
    
    for c in CLASE_TINTA:
        np.save(os.path.join(DIR_IESIRE, 'task2', f'detections_{c}.npy'), np.array(rez_t2[c]['cutii'], dtype=object))
        np.save(os.path.join(DIR_IESIRE, 'task2', f'scores_{c}.npy'), np.array(rez_t2[c]['scores']))
        np.save(os.path.join(DIR_IESIRE, 'task2', f'file_names_{c}.npy'), np.array(rez_t2[c]['fisiere']))

if not os.path.exists(DIR_ANTRENARE_SURSA):
    DIR_ANTRENARE_SURSA = os.path.join(DIR_RADACINA, 'scooby-doo-characters-detection', 'antrenare')
    DIR_VALIDARE_SURSA = os.path.join(DIR_RADACINA, 'scooby-doo-characters-detection', 'validare', 'validare')

pregateste_date()
model = antreneaza_model()
ruleaza_inferenta(model)