import os
import random
import zmq

def send_random_image(folder_path, server_address="tcp://127.0.0.1:5555"):
    if not os.path.exists(folder_path):
        print(f"Mapa {folder_path} ne obstaja.")
        return

    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"V mapi {folder_path} ni najdenih PNG slik.")
        return

    selected_image = random.choice(png_files)
    image_path = os.path.join(folder_path, selected_image)
    print(f"Izbrana naključna slika: {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    print(f"Povezovanje na {server_address}...")
    socket.connect(server_address)

    print("Pošiljanje slike servisu...")
    socket.send(image_bytes)

    message = socket.recv_string()
    print(f"Odgovor servisa: {message}")
    return message

if __name__ == "__main__":
    folder = "D:\\junction2026\\mat"
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Ustvarjena testna mapa '{folder}'. Prosim vanjo dodajte nekaj PNG slik.")
    else:
        send_random_image(folder)