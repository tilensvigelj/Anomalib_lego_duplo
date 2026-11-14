# Vodič od ničle: Kaj se tukaj dogaja in kako vse skupaj deluje?

Ta dokument razlaga celoten sistem **od popolne ničle** — za **eno kamero**, vzporedni 3-modelni ansambel **`PatchCore + PaDiM + EfficientAd`** (razporejen na **Intel GPU + NPU** z glasovanjem **2/3**) in neposredni izhod za VLA (**`"perfect"`** ali **`"defective"`**).

---

## 1. Velika slika: Kaj gradimo in kaj je tvoja vloga?

Robotska roka z **eno kamero** pregleduje in sortira LEGO Duplo kocke:
- **Čiste kocke (`"perfect"`)** gredo v **modro izhodno škatlo**.
- **Popisane kocke (`"defective"`)** (z oznako flomastra) gredo v **roza škatlo**.

**Tvoja vloga (Oseba C — zaznavanje anomalij):**
Ko kamera slika kocko na podlagi, VLA / orkestrator prek lokalne povezave ZeroMQ (`tcp://127.0.0.1:5555`) pošlje sliko tvojemu servisu ([anomaly/service.py](file:///c:/Users/tilen/Desktop/JUNCTION/physicalai_env/anomaly/service.py)). Tvoj servis vrne **samo čist tekstovni niz**:
- **`"perfect"`** -> čista kocka (`score < threshold`),
- **`"defective"`** -> zaznana napaka (`score >= threshold` ali neveljavna slika).

Ni nobenega vmesnega stanja in nobene druge kamere — en sam prag odloči takoj.

---

## 2. Katere knjižnice uporabljamo (razloženo po domače)

1. **OpenCV (`cv2`) in NumPy — Izrez kocke brez senc in odbleskov (`perception/`)**
   - Primerja fotografijo kocke s fotografijo prazne mize (`data/calibration/background_mat.png`), odstrani sence na mizi, z metodo `remove_specular_glare` zgladi ostre bele odbleske luči na plastiki (brez brisanja flomastra!) ter izreže kocko na sivo ozadje v velikosti **256 × 256 pikslov**.
2. **Anomalib (`PatchCore` + `PaDiM` + `EfficientAd`) — 3-modelni ansambel z 2/3 glasovanjem (`anomaly/`)**
   - Med učenjem vidi **samo čiste kocke** (`data/processed/mat/train/good/`).
   - Vsak model svojo surovo oceno najprej **normalizira glede na svoj kalibrirani prag** (`norm_i = score_i / threshold_i`, kjer `>= 1.0` pomeni glas za `"defective"`):
     - **`patchcore` (`GPU`)**: lokalni iskalnik najbližjih sosedov, odličen za drobne pikice in praske.
     - **`padim` (`GPU`)**: prostorski Gaussov model po mestih kocke.
     - **`efficient_ad` (`NPU`)**: hitra učitelj-učenec nevronska mreža s simetričnimi konvolucijami za Intel NPU.
   - Končna odločitev je sprejeta po principu **večine 2/3** (vsaj **2 od 3** modelov morata glasovati za `"defective"`).
3. **OpenVINO — Vzporedno izvajanje na Intel `NPU` + `GPU`**
   - Vsi trije modeli tečejo **istočasno (`start_async`)**: `EfficientAd` na **Intel NPU**, `PatchCore` in `PaDiM` pa vzporedno na **Intel Arc GPU** (s samodejnim varnostnim preklopom na `CPU`, če naprava ni prisotna).
4. **ZeroMQ (`zmq`) — Povezava z VLA (`tcp://127.0.0.1:5555`)**
   - Sprejme sliko (JSON `{"image": "<base64>"}`, navaden base64 niz ali surove PNG bajte) in odgovori s čistim nizom `"perfect"` ali `"defective"`.

---

## 3. Pot ene fotografije skozi tvoje programe (korak za korakom)

```text
[1. Surove fotografije]       data/raw/mat/background/  (prazna podlaga brez kocke)
                              data/raw/mat/train/good/  (čiste kocke za učenje)
                              data/raw/mat/test/good/   (čiste kocke za test)
                              data/raw/mat/test/defect/ (kocke s flomastrom)
         │
         ▼  python -m perception.preprocess
[2. Čisti 256×256 izrezi]     data/processed/mat/...    (kocka na sivi podlagi, brez senc)
         │
         ▼  python -m perception.augment_hue            (opcijsko: dodatne barve čistih kock)
         │
         ▼  python -m anomaly.train --model padim --version 1
         ▼  python -m anomaly.train --model patchcore --version 1
         ▼  python -m anomaly.train --model efficient_ad --version 1
[3. Natrenirani modeli]       anomaly/results/{padim,patchcore,efficient_ad}_mat_v1/
         │
         ▼  python -m anomaly.export --model padim --version 1 --precision fp16
         ▼  python -m anomaly.export --model patchcore --version 1 --precision fp16
         ▼  python -m anomaly.export --model efficient_ad --version 1 --precision fp16
[4. Hitri OpenVINO modeli]    anomaly/exports/{padim,patchcore,efficient_ad}_mat_v1/weights/openvino/model.xml
         │
         ▼  python -m anomaly.verify --device AUTO_INTEL --update-rules
[5. Kalibrirani pragovi]      configs/rules.yaml  (prag za vsak model + razporeditev NPU/GPU + 2/3 glasovanje)
         │
         ▼  python -m anomaly.service
[6. Živi servis za VLA]       Posluša na tcp://127.0.0.1:5555 -> vrne "perfect" ali "defective"
```

---

## 4. Kako kalibriraš sistem?

### A. Kalibracija izreza ozadja ([configs/defect_config.json](file:///c:/Users/tilen/Desktop/JUNCTION/physicalai_env/configs/defect_config.json))
- Poslikaj 5–15 slik prazne mize v `data/raw/mat/background/` in poženi `python -m perception.preprocess`.
- V mapi `data/processed/mat/debug/` preveri zelene obrobe:
  - Če program spregleda bledo kocko (`no_component`), znižaj `"CHROMA_THRESH"` (npr. na `10.0`).
  - Če zajame preveč temne sence, zvišaj `"LUMA_DARK_THRESH"` (npr. na `55.0`).

### B. Kalibracija praga `threshold` ([configs/rules.yaml](file:///c:/Users/tilen/Desktop/JUNCTION/physicalai_env/configs/rules.yaml))
Odločitev temelji na **enem samem pragu (`threshold`)** brez vmesnega območja:
```text
   0.0                     threshold                    1.0+
    ├──────────────────────────┼──────────────────────────► score
           "perfect"                  "defective"
```
- `score < threshold` -> **`"perfect"`**
- `score >= threshold` -> **`"defective"`**

Samodejna kalibracija (nastavi prag tako, da ujame 100 % popisanih testnih kock v `test/defect`):
```powershell
python -m anomaly.verify --device AUTO_INTEL --update-rules
```
Ročna prilagoditev v `configs/rules.yaml` (bere se sproti ob vsakem klicu brez ponovnega zagona servisa!):
- Če čiste kocke prepogosto dobijo `"defective"` -> rahlo **zvišaj** vrednost v `model_thresholds:` (npr. `patchcore_mat_v4` iz `0.025` na `0.030`).
- Če droben flomaster dobi `"perfect"` -> **znižaj** vrednost v `model_thresholds:` (npr. na `0.020`).

---

## 5. Kako preizkusiš posamezno sliko in pogledaš heatmap?

Za hiter preizkus katerekoli slike iz terminala uporabi [anomaly/evaluate.py](file:///c:/Users/tilen/Desktop/JUNCTION/physicalai_env/anomaly/evaluate.py):

```powershell
# Shrani vizualno sliko (izrez + heatmap + odločitev PERFECT / DEFECTIVE):
python -m anomaly.evaluate `
    --image data/raw/mat/test/defect/tvoja_slika.png `
    --save-vis anomaly/results/eval_vis.png

# Izpiše SAMO tekstovni niz ("perfect" ali "defective"), točno kot VLA servis:
python -m anomaly.evaluate `
    --image data/raw/mat/test/good/tvoja_slika.png `
    --verdict-only
```

---

## 6. Primer klica ZeroMQ servisa iz VLA kode (Python)

```python
import base64
from pathlib import Path
import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://127.0.0.1:5555")

# Lahko pošlješ surove PNG bajte, base64 niz ali JSON {"image": "<base64>"}:
sock.send(Path("camera_frame.png").read_bytes())
status = sock.recv_string()  # vrne samo "perfect" ali "defective"
print(status)
```
