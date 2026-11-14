import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import pyrealsense2 as rs


def realsense_rgb_zajemanje(privzeta_mapa: str = "."):
    """
    Odpre UI s predogledom RGB toka iz Intel RealSense kamere.
    Omogoča nastavljanje resolucije, mape za shranjevanje in zajem slik.
    """
    # 1. Preveri priključeno kamero in pridobi podprte RGB resolucije
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("Napaka", "Intel RealSense kamera ni bila najdena!")
        root_err.destroy()
        return

    dev = devices[0]
    podprte_resolucije = {}  # {(width, height): [fps1, fps2, ...]}
    for sensor in dev.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.color and profile.format() == rs.format.rgb8:
                vp = profile.as_video_stream_profile()
                wh = (vp.width(), vp.height())
                podprte_resolucije.setdefault(wh, []).append(vp.fps())

    if not podprte_resolucije:
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("Napaka", "Kamera nima podprtih RGB8 profilov.")
        root_err.destroy()
        return

    # Uredi resolucije po velikosti (od najmanjše do največje)
    urejene_resolucije = sorted(podprte_resolucije.keys(), key=lambda x: (x[0], x[1]))
    moznosti_resolucij = [f"{w}x{h}" for w, h in urejene_resolucije]

    # Izberi privzeto resolucijo (npr. 1280x720, če obstaja, sicer prvo)
    privzeta_res = "1280x720" if "1280x720" in moznosti_resolucij else moznosti_resolucij[0]

    # Stanje aplikacije
    pipeline = None
    trenutni_rgb_okvir = None
    aktivno = True

    # 2. Nastavitev Tkinter okna
    root = tk.Tk()
    root.title("Intel RealSense - RGB Zajemanje")
    root.resizable(True, True)

    mapa_var = tk.StringVar(value=os.path.abspath(privzeta_mapa))
    res_var = tk.StringVar(value=privzeta_res)
    status_var = tk.StringVar(value="Pripravljen na zajemanje.")

    # --- Zgornja vrstica z nastavitvami ---
    okvir_kontrole = ttk.Frame(root, padding=10)
    okvir_kontrole.pack(fill=tk.X)

    # Izbira resolucije
    ttk.Label(okvir_kontrole, text="Resolucija:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
    combo_res = ttk.Combobox(
        okvir_kontrole, textvariable=res_var, values=moznosti_resolucij, state="readonly", width=12
    )
    combo_res.grid(row=0, column=1, padx=(0, 15))

    # Izbira mape
    ttk.Label(okvir_kontrole, text="Mapa:").grid(row=0, column=2, padx=(0, 5), sticky=tk.W)
    vnos_mapa = ttk.Entry(okvir_kontrole, textvariable=mapa_var, width=40)
    vnos_mapa.grid(row=0, column=3, padx=(0, 5), sticky=tk.EW)
    okvir_kontrole.columnconfigure(3, weight=1)

    def izberi_mapo():
        izbrana = filedialog.askdirectory(initialdir=mapa_var.get(), title="Izberi mapo za shranjevanje")
        if izbrana:
            mapa_var.set(izbrana)

    gumb_mapa = ttk.Button(okvir_kontrole, text="Brskaj...", command=izberi_mapo)
    gumb_mapa.grid(row=0, column=4, padx=(0, 15))

    # Gumb za slikanje
    def zajemi_sliko(event=None):
        if trenutni_rgb_okvir is None:
            status_var.set("Opozorilo: Ni še slike za zajem.")
            return

        ciljna_mapa = mapa_var.get().strip()
        if not ciljna_mapa:
            messagebox.showwarning("Opozorilo", "Vnesite veljavno mapo za shranjevanje.")
            return

        os.makedirs(ciljna_mapa, exist_ok=True)
        casovni_zig = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ime_datoteke = f"realsense_rgb_{casovni_zig}.png"
        celotna_pot = os.path.join(ciljna_mapa, ime_datoteke)

        # Shrani sliko v polni izbrani resoluciji
        slika = Image.fromarray(trenutni_rgb_okvir)
        slika.save(celotna_pot)
        status_var.set(f"Shranjeno ({slika.width}x{slika.height}): {ime_datoteke}")

    gumb_zajemi = ttk.Button(okvir_kontrole, text="📷 Zajemi sliko", command=zajemi_sliko)
    gumb_zajemi.grid(row=0, column=5)

    # --- Predogled videa ---
    label_predogled = ttk.Label(root, anchor=tk.CENTER)
    label_predogled.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    # --- Statusna vrstica ---
    label_status = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
    label_status.pack(fill=tk.X, side=tk.BOTTOM)

    # 3. Upravljanje RealSense pipeline-a
    def zazeni_pipeline(resolucija_str: str):
        nonlocal pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

        w, h = map(int, resolucija_str.split("x"))
        razpolozljivi_fps = podprte_resolucije[(w, h)]
        # Izberi 30 fps če je na voljo, sicer najvišji možen
        fps = 30 if 30 in razpolozljivi_fps else max(razpolozljivi_fps)

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)

        try:
            pipeline.start(config)
            status_var.set(f"Predogled aktiven: {w}x{h} @ {fps} FPS")
        except Exception as e:
            messagebox.showerror("Napaka kamere", f"Ni bilo mogoče zagnati toka {w}x{h}:\n{e}")

    def ob_spremembi_resolucije(event):
        zazeni_pipeline(res_var.get())

    combo_res.bind("<<ComboboxSelected>>", ob_spremembi_resolucije)

    # 4. Zanka za osveževanje slike v UI
    def posodobi_predogled():
        nonlocal trenutni_rgb_okvir
        if not aktivno:
            return

        if pipeline is not None:
            try:
                frames = pipeline.poll_for_frames()
                if frames:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        # Polna resolucija za shranjevanje
                        trenutni_rgb_okvir = np.asanyarray(color_frame.get_data()).copy()

                        # Pomanjšana slika za predogled (da ne preseže velikosti zaslona)
                        pil_img = Image.fromarray(trenutni_rgb_okvir)
                        max_w, max_h = 960, 540
                        if pil_img.width > max_w or pil_img.height > max_h:
                            pil_img.thumbnail((max_w, max_h), Image.Resampling.BILINEAR)

                        imgtk = ImageTk.PhotoImage(image=pil_img)
                        label_predogled.imgtk = imgtk  # Hrani referenco proti garbage collectorju
                        label_predogled.configure(image=imgtk)
            except Exception as e:
                status_var.set(f"Napaka pri branju okvirja: {e}")

        root.after(15, posodobi_predogled)

    def ob_zaprtju():
        nonlocal aktivno
        aktivno = False
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", ob_zaprtju)
    root.bind("<space>", zajemi_sliko)  # Bližnjica s preslednico za slikanje

    # Začetni zagon
    zazeni_pipeline(privzeta_res)
    posodobi_predogled()
    root.mainloop()


# Primer klica funkcije:
if __name__ == "__main__":
    realsense_rgb_zajemanje(privzeta_mapa="./zajete_slike")