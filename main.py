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
DIR_RADACINA = '/kaggle/input/scooby-doo-characters-detection'
DIR_ANTRENARE = os.path.join(DIR_RADACINA, 'antrenare')
DIR_VALIDARE = os.path.join(DIR_RADACINA, 'validare', 'validare')
DIR_IESIRE = '/kaggle/working'
CALE_MODEL = 'model_scooby.pth'

CLASE = ['daphne', 'fred', 'shaggy', 'velma']
IDX_CLASE = {cls: i for i, cls in enumerate(CLASE)}
DISPOZITIV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ANCORE = [(60, 80), (50, 50), (70, 70), (40, 60)]

class DetectorFacial(nn.Module):
    def __init__(self):
        super(DetectorFacial, self).__init__()
        
        resnet = models.resnet18(weights=None)
        
        # pastram doar straturile convolutionale
        self.trasaturi = nn.Sequential(*list(resnet.children())[:-1])
        
        self.procesare_comuna = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # capetele retelei pentru: detectie, clasificare si regresie
        self.cls_fata = nn.Linear(512, 1)
        self.cls_personaj = nn.Linear(512, len(CLASE))
        self.reg_cutie = nn.Linear(512, 4) 
        
        self._initializare_greutati()
        
    def _initializare_greutati(self):
        # initializare manuala pentru convergenta mai rapida
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
        feat = self.trasaturi(x).flatten(1)
        feat = self.procesare_comuna(feat)
        return torch.sigmoid(self.cls_fata(feat)), self.cls_personaj(feat), self.reg_cutie(feat)

# augmentari
transform_antrenare = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

transform_validare = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# functii utilitare cutii
def codifica_cutii(box_gt, box_ancora):
    gx, gy, gw, gh = box_gt
    ax, ay, aw, ah = box_ancora
    return torch.tensor([
        (gx - ax) / aw,
        (gy - ay) / ah,
        np.log(gw / aw + 1e-6),
        np.log(gh / ah + 1e-6)
    ], dtype=torch.float32)

def decodifica_cutii(offseturi, ancore):
    if offseturi.ndim == 1: offseturi = offseturi.unsqueeze(0)
    if ancore.ndim == 1: ancore = ancore.unsqueeze(0)
    
    px = ancore[:, 0] + offseturi[:, 0] * ancore[:, 2]
    py = ancore[:, 1] + offseturi[:, 1] * ancore[:, 3]
    pw = ancore[:, 2] * torch.exp(offseturi[:, 2])
    ph = ancore[:, 3] * torch.exp(offseturi[:, 3])
    
    return torch.stack([px, py, pw, ph], dim=1)

# dataset personalizat
class SetDateFereastra(torch.utils.data.Dataset):
    def __init__(self, metadate, transform=None):
        self.metadate = metadate
        self.transform = transform
        
    def __len__(self):
        return len(self.metadate)
    
    def __getitem__(self, idx):
        item = self.metadate[idx]
        img = cv2.cvtColor(cv2.imread(item['cale']), cv2.COLOR_BGR2RGB)
        
        if item['tip'] == 'pozitiv':
            x1, y1, x2, y2 = item['box']
            h_img, w_img = img.shape[:2]
            
            # padding si jitter
            px = max(1, int((x2 - x1) * 0.15))
            py = max(1, int((y2 - y1) * 0.15))
            cx1 = max(0, int(x1 - np.random.randint(-px, px + 1)))
            cy1 = max(0, int(y1 - np.random.randint(-py, py + 1)))
            cx2 = min(w_img, int(x2 + np.random.randint(-px, px + 1)))
            cy2 = min(h_img, int(y2 + np.random.randint(-py, py + 1)))
            
            if cx2 <= cx1 + 5 or cy2 <= cy1 + 5:
                cx1, cy1, cx2, cy2 = int(x1), int(y1), int(x2), int(y2)
            
            decupaj = img[cy1:cy2, cx1:cx2]
            
            # recalculam tintele fata de noul crop
            gt_w, gt_h = (x2 - x1), (y2 - y1)
            gt_cx, gt_cy = (x1 + gt_w/2) - cx1, (y1 + gt_h/2) - cy1
            aw, ah = (cx2 - cx1), (cy2 - cy1)
            
            tinta_reg = codifica_cutii((gt_cx, gt_cy, gt_w, gt_h), (aw/2, ah/2, aw, ah))
            lbl_fata, lbl_pers = 1.0, item['eticheta']
        else:
            # negativ - decupaj random din fundal
            anc = ANCORE[np.random.randint(len(ANCORE))]
            h, w = img.shape[:2]
            if h > anc[1] + 10 and w > anc[0] + 10:
                y = np.random.randint(0, h - anc[1])
                x = np.random.randint(0, w - anc[0])
                decupaj = img[y:y+anc[1], x:x+anc[0]]
            else:
                decupaj = cv2.resize(img, anc)
            
            tinta_reg = torch.tensor([0,0,0,0], dtype=torch.float32)
            lbl_fata, lbl_pers = 0.0, -1

        if decupaj.size == 0: decupaj = np.zeros((64, 64, 3), dtype=np.uint8)
        if self.transform: decupaj = self.transform(decupaj)
            
        return decupaj, torch.tensor(lbl_fata, dtype=torch.float32), \
               torch.tensor(lbl_pers, dtype=torch.long), tinta_reg

# pregatirea datelor
def pregateste_date(dir_radacina):
    pozitive, negative = [], []
    toate_img = glob.glob(os.path.join(dir_radacina, '**', '*.jpg'), recursive=True)
    
    print("Indexam datele...")
    for cls in CLASE:
        cale_annot = os.path.join(dir_radacina, f'{cls}_annotations.txt')
        if not os.path.exists(cale_annot): continue
        
        with open(cale_annot, 'r') as f:
            for linie in f:
                p = linie.strip().split()
                if len(p) == 6:
                    cale = os.path.join(dir_radacina, cls, p[0])
                    pozitive.append({
                        'cale': cale, 
                        'box': tuple(map(float, p[1:5])), 
                        'eticheta': IDX_CLASE.get(p[5], -1), 
                        'tip': 'pozitiv'
                    })

    # generam negative (raport 3:1)
    target_neg = len(pozitive) * 3
    for _ in range(target_neg):
        path = np.random.choice(toate_img)
        negative.append({'cale': path, 'tip': 'negativ'})
        
    return pozitive, negative

# bucla de antrenare
def antreneaza_model():
    pos, neg = pregateste_date(DIR_ANTRENARE)
    dataset = SetDateFereastra(pos + neg, transform=transform_antrenare)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    model = DetectorFacial().to(DISPOZITIV)
    
    opt = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    sch = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    
    c_bce, c_ce, c_reg = nn.BCELoss(), nn.CrossEntropyLoss(ignore_index=-1), nn.SmoothL1Loss()
    
    print("Incepem antrenarea (25 epoci)...")
    for epoch in range(25):
        model.train()
        pierdere_totala = 0
        
        for img, l_fata, l_pers, t_reg in loader:
            img, l_fata, l_pers, t_reg = img.to(DISPOZITIV), l_fata.to(DISPOZITIV), \
                                         l_pers.to(DISPOZITIV), t_reg.to(DISPOZITIV)
            
            opt.zero_grad()
            p_fata, p_pers, p_reg = model(img)
            
            loss = c_bce(p_fata.squeeze(), l_fata) + c_ce(p_pers, l_pers)
            if (mask := l_fata > 0.5).sum() > 0:
                loss += c_reg(p_reg[mask], t_reg[mask]) * 2.0
            
            loss.backward()
            opt.step()
            pierdere_totala += loss.item()
            
        sch.step()
        print(f"Epoca {epoch+1}, Loss Mediu: {pierdere_totala/len(loader):.4f}")
    
    torch.save(model.state_dict(), CALE_MODEL)
    return model

# inferenta
def ruleaza_inferenta(model):
    model.eval()
    imagini_validare = [f for f in os.listdir(DIR_VALIDARE) if f.endswith('.jpg')]
    
    rez_t1 = {'cutii': [], 'scoruri': [], 'fisiere': []}
    rez_t2 = {c: {'cutii': [], 'scoruri': [], 'fisiere': []} for c in CLASE}
    
    scari, pas = [1.2, 0.9, 0.6, 0.4], 16
    
    print(f"Procesam {len(imagini_validare)} imagini de validare...")
    for i, nume_fisier in enumerate(imagini_validare):
        path = os.path.join(DIR_VALIDARE, nume_fisier)
        img_orig = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        det_img, scor_img, pers_img, scor_pers_img = [], [], [], []
        
        for scara in scari:
            img_redim = cv2.resize(img_orig, (0,0), fx=scara, fy=scara)
            if min(img_redim.shape[:2]) < 60: continue
            
            for aw, ah in ANCORE:
                decupaje, coordonate, ancore_tensor = [], [], []
                
                # fereastra glisanta
                for y in range(0, img_redim.shape[0] - ah, pas):
                    for x in range(0, img_redim.shape[1] - aw, pas):
                        decupaje.append(img_redim[y:y+ah, x:x+aw])
                        coordonate.append((x,y))
                        ancore_tensor.append([aw/2, ah/2, aw, ah])
                
                if not decupaje: continue
                
                # procesare batch
                for b in range(0, len(decupaje), 128):
                    batch_tens = torch.stack([transform_validare(c) for c in decupaje[b:b+128]]).to(DISPOZITIV)
                    ancore_batch = torch.tensor(ancore_tensor[b:b+128], dtype=torch.float32).to(DISPOZITIV)
                    coords_batch = coordonate[b:b+128]
                    
                    with torch.no_grad():
                        pred_fata, pred_pers, pred_reg = model(batch_tens)
                        prob_pers = torch.softmax(pred_pers, dim=1)
                        
                        mask = pred_fata.squeeze() > 0.75
                        if not torch.any(mask): continue
                        
                        idxs = torch.nonzero(mask).flatten()
                        cutii_pred = decodifica_cutii(pred_reg[idxs], ancore_batch[idxs])
                        
                        for k, idx in enumerate(idxs):
                            ii = idx.item()
                            px, py, pw, ph = cutii_pred[k].tolist()
                            wx, wy = coords_batch[ii]
                            
                            # coordonate finale
                            cx, cy = (wx + px) / scara, (wy + py) / scara
                            w, h = pw / scara, ph / scara
                            
                            det_img.append([int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)])
                            scor_img.append(pred_fata[ii].item())
                            
                            best_p = torch.argmax(prob_pers[ii]).item()
                            pers_img.append(best_p)
                            scor_pers_img.append(prob_pers[ii][best_p].item())

        if det_img:
            # aplicam nms
            t_boxes = torch.tensor(det_img, dtype=torch.float32).to(DISPOZITIV)
            t_scores = torch.tensor(scor_img, dtype=torch.float32).to(DISPOZITIV)
            pastreaza = nms(t_boxes, t_scores, 0.3).cpu().numpy()
            
            for k in pastreaza:
                rez_t1['cutii'].append(det_img[k])
                rez_t1['scoruri'].append(scor_img[k])
                rez_t1['fisiere'].append(nume_fisier)
                
                nume_pers = CLASE[pers_img[k]]
                rez_t2[nume_pers]['cutii'].append(det_img[k])
                rez_t2[nume_pers]['scoruri'].append(scor_img[k] * scor_pers_img[k])
                rez_t2[nume_pers]['fisiere'].append(nume_fisier)

        if i % 50 == 0: print(f"Procesat {i}/{len(imagini_validare)}")

    # salvare rezultate
    os.makedirs(os.path.join(DIR_IESIRE, 'task1'), exist_ok=True)
    os.makedirs(os.path.join(DIR_IESIRE, 'task2'), exist_ok=True)
    
    np.save(os.path.join(DIR_IESIRE, 'task1', 'detections_all_faces.npy'), np.array(rez_t1['cutii'], dtype=object))
    np.save(os.path.join(DIR_IESIRE, 'task1', 'scores_all_faces.npy'), np.array(rez_t1['scoruri']))
    np.save(os.path.join(DIR_IESIRE, 'task1', 'file_names_all_faces.npy'), np.array(rez_t1['fisiere']))
    
    for c in CLASE:
        np.save(os.path.join(DIR_IESIRE, 'task2', f'detections_{c}.npy'), np.array(rez_t2[c]['cutii'], dtype=object))
        np.save(os.path.join(DIR_IESIRE, 'task2', f'scores_{c}.npy'), np.array(rez_t2[c]['scoruri']))
        np.save(os.path.join(DIR_IESIRE, 'task2', f'file_names_{c}.npy'), np.array(rez_t2[c]['fisiere']))

if not os.path.exists(DIR_ANTRENARE):
    DIR_ANTRENARE = os.path.join(DIR_RADACINA, 'scooby-doo-characters-detection', 'antrenare')
    DIR_VALIDARE = os.path.join(DIR_RADACINA, 'scooby-doo-characters-detection', 'validare', 'validare')

model = DetectorFacial().to(DISPOZITIV)

if os.path.exists(CALE_MODEL):
    print(f"Incarcam modelul din {CALE_MODEL}...")
    model.load_state_dict(torch.load(CALE_MODEL, map_location=DISPOZITIV))
else:
    model = antreneaza_model()

ruleaza_inferenta(model)