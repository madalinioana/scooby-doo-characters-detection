import os
import numpy as np
import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torchvision.ops import nms

# configurari si constante
ROOT_DIR = '/kaggle/input/scooby-doo-characters-detection'
TRAIN_DIR = os.path.join(ROOT_DIR, 'antrenare')
VAL_DIR = os.path.join(ROOT_DIR, 'validare', 'validare')
OUTPUT_DIR = '/kaggle/working/cnn_results'
MODEL_PATH = 'model_scooby.pth'

CLASSES = ['daphne', 'fred', 'shaggy', 'velma']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {DEVICE}")
ANCHORS = [(60, 80), (50, 50), (70, 70), (40, 60)]

class FaceDetector(nn.Module):
    def __init__(self):
        super(FaceDetector, self).__init__()
        
        resnet = models.resnet18(weights=None)
        
        # pastram doar straturile convolutionale
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.shared_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # capetele retelei pentru: detectie, clasificare si regresie
        self.face_cls = nn.Linear(512, 1)
        self.char_cls = nn.Linear(512, len(CLASSES))
        self.bbox_reg = nn.Linear(512, 4) 
        
        self.initializeaza_greutati()
        
    def initializeaza_greutati(self):
        # initializare manuala pentru convergenta mai rapida
        # am incercat si cu xavier dar kaiming e mai bun
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.features(x).flatten(1)
        feats = self.shared_head(feats)
        
        face_score = torch.sigmoid(self.face_cls(feats))
        char_logits = self.char_cls(feats)
        bbox_offsets = self.bbox_reg(feats)
        
        return face_score, char_logits, bbox_offsets

# augmentari
train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

val_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# functii utilitare cutii
def encodeBox(gt_box, anchor_box):
    gx, gy, gw, gh = gt_box
    ax, ay, aw, ah = anchor_box
    
    tx = (gx - ax) / aw
    ty = (gy - ay) / ah
    tw = np.log(gw / aw + 1e-6)
    th = np.log(gh / ah + 1e-6)
    
    return torch.tensor([tx, ty, tw, th], dtype=torch.float32)

def decodifica_cutie(offsets, anchors):
    if offsets.ndim == 1:
        offsets = offsets.unsqueeze(0)
    if anchors.ndim == 1:
        anchors = anchors.unsqueeze(0)
    
    pred_x = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
    pred_y = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
    pred_w = anchors[:, 2] * torch.exp(offsets[:, 2])
    pred_h = anchors[:, 3] * torch.exp(offsets[:, 3])
    
    boxes = torch.stack([pred_x, pred_y, pred_w, pred_h], dim=1)
    return boxes

# dataset personalizat
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        meta = self.metadata[idx]
        img = cv2.imread(meta['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if meta['type'] == 'positive':
            x1, y1, x2, y2 = meta['bbox']
            img_h, img_w = img.shape[:2]
            
            # padding si jitter pentru augmentare
            pad_x = max(1, int((x2 - x1) * 0.15))
            pad_y = max(1, int((y2 - y1) * 0.15))
            
            crop_x1 = max(0, int(x1 - np.random.randint(-pad_x, pad_x + 1)))
            crop_y1 = max(0, int(y1 - np.random.randint(-pad_y, pad_y + 1)))
            crop_x2 = min(img_w, int(x2 + np.random.randint(-pad_x, pad_x + 1)))
            crop_y2 = min(img_h, int(y2 + np.random.randint(-pad_y, pad_y + 1)))
            
            if crop_x2 <= crop_x1 + 5 or crop_y2 <= crop_y1 + 5:
                crop_x1, crop_y1 = int(x1), int(y1)
                crop_x2, crop_y2 = int(x2), int(y2)
            
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # recalculam tintele fata de noul crop
            gt_w = x2 - x1
            gt_h = y2 - y1
            gt_cx = (x1 + gt_w/2) - crop_x1
            gt_cy = (y1 + gt_h/2) - crop_y1
            
            anchor_w = crop_x2 - crop_x1
            anchor_h = crop_y2 - crop_y1
            
            bbox_target = encodeBox((gt_cx, gt_cy, gt_w, gt_h), 
                                    (anchor_w/2, anchor_h/2, anchor_w, anchor_h))
            face_label = 1.0
            char_label = meta['label']
        else:
            # negativ - decupaj random din fundal
            anchor = ANCHORS[np.random.randint(len(ANCHORS))]
            h, w = img.shape[:2]
            
            if h > anchor[1] + 10 and w > anchor[0] + 10:
                y = np.random.randint(0, h - anchor[1])
                x = np.random.randint(0, w - anchor[0])
                crop = img[y:y+anchor[1], x:x+anchor[0]]
            else:
                crop = cv2.resize(img, anchor)
            
            bbox_target = torch.tensor([0,0,0,0], dtype=torch.float32)
            face_label = 0.0
            char_label = -1

        if crop.size == 0:
            crop = np.zeros((64, 64, 3), dtype=np.uint8)
            
        if self.transform:
            crop = self.transform(crop)
            
        return crop, torch.tensor(face_label, dtype=torch.float32), \
               torch.tensor(char_label, dtype=torch.long), bbox_target

# pregatirea datelor
def prepare_data(root_dir):
    sample_pozitive = []
    sample_negative = []
    
    toate_imaginile = glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True)
    
    print("Indexam datele...")
    for class_name in CLASSES:
        annot_path = os.path.join(root_dir, f'{class_name}_annotations.txt')
        # print(f"Procesez clasa: {class_name}")
        if not os.path.exists(annot_path):
            continue
        
        with open(annot_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 6:
                    img_path = os.path.join(root_dir, class_name, parts[0])
                    bbox = tuple(map(float, parts[1:5]))
                    label = CLASS_TO_IDX.get(parts[5], -1)
                    
                    sample_pozitive.append({
                        'path': img_path, 
                        'bbox': bbox, 
                        'label': label, 
                        'type': 'positive'
                    })

    # generam negative (raport 3:1)
    num_negatives = len(sample_pozitive) * 3
    for _ in range(num_negatives):
        random_path = np.random.choice(toate_imaginile)
        sample_negative.append({'path': random_path, 'type': 'negative'})
        
    return sample_pozitive, sample_negative

# bucla de antrenare
def antreneaza_model():
    pos_samples, neg_samples = prepare_data(TRAIN_DIR)
    toate_samples = pos_samples + neg_samples
    
    dataset = WindowDataset(toate_samples, transform=train_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, 
                                             shuffle=True, num_workers=2)
    
    model = FaceDetector().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    reg_loss = nn.SmoothL1Loss()
    
    print("Incepem antrenarea (25 epoci)...")
    num_epoci = 25
    # am incercat cu 30 dar overfittingul era prea mare
    # num_epoci = 30
    for epoch in range(num_epoci):
        model.train()
        pierdere_totala = 0
        
        for imgs, face_labels, char_labels, bbox_targets in dataloader:
            imgs = imgs.to(DEVICE)
            face_labels = face_labels.to(DEVICE)
            char_labels = char_labels.to(DEVICE)
            bbox_targets = bbox_targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            face_preds, char_preds, bbox_preds = model(imgs)
            
            loss = bce_loss(face_preds.squeeze(), face_labels)
            loss = loss + ce_loss(char_preds, char_labels)
            
            masca_pozitive = face_labels > 0.5
            if masca_pozitive.sum() > 0:
                loss = loss + reg_loss(bbox_preds[masca_pozitive], bbox_targets[masca_pozitive]) * 2.0
            
            loss.backward()
            optimizer.step()
            
            pierdere_totala += loss.item()
            
        scheduler.step()
        avg_loss = pierdere_totala / len(dataloader)
        print(f"Epoca {epoch+1}, Loss Mediu: {avg_loss:.4f}")
        # if epoch % 5 == 0:
        #     print(f"  LR curent: {optimizer.param_groups[0]['lr']}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    return model

# inferenta
def ruleaza_inferenta(model):
    model.eval()
    
    imagini_validare = [f for f in os.listdir(VAL_DIR) if f.endswith('.jpg')]
    
    rez_task1 = {'boxes': [], 'scores': [], 'files': []}
    rez_task2 = {c: {'boxes': [], 'scores': [], 'files': []} for c in CLASSES}
    
    scari = [1.2, 0.9, 0.6, 0.4]  # am testat si cu mai multe scari dar dura prea mult
    stride = 16
    # threshold_conf = 0.75
    print(f"Procesam {len(imagini_validare)} imagini de validare...")
    for img_idx, filename in enumerate(imagini_validare):
        img_path = os.path.join(VAL_DIR, filename)
        imagine_originala = cv2.imread(img_path)
        imagine_originala = cv2.cvtColor(imagine_originala, cv2.COLOR_BGR2RGB)
        
        toate_detectiile = []
        toate_scorurile = []
        toate_personajele = []
        scoruri_personaje = []
        
        for scale in scari:
            img_scalat = cv2.resize(imagine_originala, (0,0), fx=scale, fy=scale)
            
            if min(img_scalat.shape[:2]) < 60:
                continue
            
            for anchor_w, anchor_h in ANCHORS:
                crops = []
                coordonate = []
                lista_ancore = []
                
                # fereastra glisanta
                for y in range(0, img_scalat.shape[0] - anchor_h, stride):
                    for x in range(0, img_scalat.shape[1] - anchor_w, stride):
                        crop = img_scalat[y:y+anchor_h, x:x+anchor_w]
                        crops.append(crop)
                        coordonate.append((x, y))
                        lista_ancore.append([anchor_w/2, anchor_h/2, anchor_w, anchor_h])
                
                if not crops:
                    continue
                
                # procesare batch - 128 e optimal pentru GPU
                marime_batch = 128
                for batch_start in range(0, len(crops), marime_batch):
                    batch_end = batch_start + marime_batch
                    batch_crops = crops[batch_start:batch_end]
                    batch_anchors = lista_ancore[batch_start:batch_end]
                    batch_coords = coordonate[batch_start:batch_end]
                    
                    batch_tensor = torch.stack([val_transform(c) for c in batch_crops])
                    batch_tensor = batch_tensor.to(DEVICE)
                    
                    ancore_tensor = torch.tensor(batch_anchors, dtype=torch.float32)
                    ancore_tensor = ancore_tensor.to(DEVICE)
                    
                    with torch.no_grad():
                        pred_fata, pred_char, pred_bbox = model(batch_tensor)
                        probabilitati_char = torch.softmax(pred_char, dim=1)
                        
                        masca_fata = pred_fata.squeeze() > 0.75
                        
                        if not torch.any(masca_fata):
                            continue
                        
                        indici_pozitivi = torch.nonzero(masca_fata).flatten()
                        cutii_prezise = decodifica_cutie(pred_bbox[indici_pozitivi], 
                                               ancore_tensor[indici_pozitivi])
                        
                        for k, idx in enumerate(indici_pozitivi):
                            i = idx.item()
                            pred_cx, pred_cy, pred_w, pred_h = cutii_prezise[k].tolist()
                            
                            fereastra_x, fereastra_y = batch_coords[i]
                            
                            # coordonate finale
                            cx_final = (fereastra_x + pred_cx) / scale
                            cy_final = (fereastra_y + pred_cy) / scale
                            w_final = pred_w / scale
                            h_final = pred_h / scale
                            
                            x1 = int(cx_final - w_final/2)
                            y1 = int(cy_final - h_final/2)
                            x2 = int(cx_final + w_final/2)
                            y2 = int(cy_final + h_final/2)
                            
                            toate_detectiile.append([x1, y1, x2, y2])
                            toate_scorurile.append(pred_fata[i].item())
                            
                            cel_mai_probabil_char = torch.argmax(probabilitati_char[i]).item()
                            toate_personajele.append(cel_mai_probabil_char)
                            scoruri_personaje.append(probabilitati_char[i][cel_mai_probabil_char].item())

        if toate_detectiile:
            # aplicam nms
            cutii_tensor = torch.tensor(toate_detectiile, dtype=torch.float32).to(DEVICE)
            scoruri_tensor = torch.tensor(toate_scorurile, dtype=torch.float32).to(DEVICE)
            
            indici_pastrate = nms(cutii_tensor, scoruri_tensor, 0.3)
            indici_pastrate = indici_pastrate.cpu().numpy()
            
            for k in indici_pastrate:
                rez_task1['boxes'].append(toate_detectiile[k])
                rez_task1['scores'].append(toate_scorurile[k])
                rez_task1['files'].append(filename)
                
                nume_personaj = CLASSES[toate_personajele[k]]
                scor_combinat = toate_scorurile[k] * scoruri_personaje[k]
                
                rez_task2[nume_personaj]['boxes'].append(toate_detectiile[k])
                rez_task2[nume_personaj]['scores'].append(scor_combinat)
                rez_task2[nume_personaj]['files'].append(filename)

        if img_idx % 50 == 0:
            print(f"Procesat {img_idx}/{len(imagini_validare)}")
            # print(f"  Detectii gasite: {len(toate_detectiile)}")

    # salvare rezultate
    dir_task1 = os.path.join(OUTPUT_DIR, 'task1')
    dir_task2 = os.path.join(OUTPUT_DIR, 'task2')
    
    os.makedirs(dir_task1, exist_ok=True)
    os.makedirs(dir_task2, exist_ok=True)
    
    np.save(os.path.join(dir_task1, 'detections_all_faces.npy'), 
            np.array(rez_task1['boxes'], dtype=object))
    np.save(os.path.join(dir_task1, 'scores_all_faces.npy'), 
            np.array(rez_task1['scores']))
    np.save(os.path.join(dir_task1, 'file_names_all_faces.npy'), 
            np.array(rez_task1['files']))
    
    for char_name in CLASSES:
        np.save(os.path.join(dir_task2, f'detections_{char_name}.npy'), 
                np.array(rez_task2[char_name]['boxes'], dtype=object))
        np.save(os.path.join(dir_task2, f'scores_{char_name}.npy'), 
                np.array(rez_task2[char_name]['scores']))
        np.save(os.path.join(dir_task2, f'file_names_{char_name}.npy'), 
                np.array(rez_task2[char_name]['files']))

if not os.path.exists(TRAIN_DIR):
    TRAIN_DIR = os.path.join(ROOT_DIR, 'scooby-doo-characters-detection', 'antrenare')
    VAL_DIR = os.path.join(ROOT_DIR, 'scooby-doo-characters-detection', 'validare', 'validare')

model = FaceDetector().to(DEVICE)

if os.path.exists(MODEL_PATH):
    print(f"Incarcam modelul din {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    model = antreneaza_model()

ruleaza_inferenta(model)