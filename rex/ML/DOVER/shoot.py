from PyRexExt import REX #important, don't remove!
from pypylon import pylon
import datetime
import os
import cv2
from pathlib import Path
import gc  # Import garbage collector
import numpy as np

cam_interface=None # je nutné využít globální proměnnou aby instance interface přežila mezi voláními main()

def init():
    global cam_interface
    # Clean up any existing interface first
    if cam_interface is not None:
        cam_interface.release()
        cam_interface = None
        gc.collect()

    try:
        #base_output_dir = rf"/mnt/external_hdd/ATMTRexTest/test"
        base_output_dir = Path(REX.p0.v).expanduser().resolve()
        sequence_no = str(int(REX.u0.v)) if isinstance(REX.u0.v, (bool, float, int)) else REX.u0.v
        cam_interface = CamInterface(base_output_dir, sequence_no)
        REX.TraceInfo("Camera interface initialized successfully")
    except Exception as e:
        REX.TraceError(f"Error initializing camera interface: {e}")
        gc.collect()

def exit():
    global cam_interface
    try:
        if cam_interface is not None:
            cam_interface.release()
            cam_interface = None
        # Force garbage collection
        gc.collect()
        REX.TraceInfo("Camera interface released and memory cleaned up")
    except Exception as e:
        REX.TraceError(f"Error during exit: {e}")
        gc.collect()

def main():
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        detail = str(int(REX.p1.v)) if isinstance(REX.p1.v, (bool, float, int)) else REX.p1.v
        filename = f"{timestamp}_{detail}"
        sequence_no = str(int(REX.u0.v)) if isinstance(REX.u0.v, (bool, float, int)) else REX.u0.v

        global cam_interface
        if cam_interface is not None:
            cam_interface.capture_frame(filename, sequence_no)
        else:
            REX.TraceError("Camera interface is not initialized")

        # Force garbage collection after each frame capture
        gc.collect()
    except Exception as e:
        REX.TraceError(f"Error in main function: {e}")
        gc.collect()



class CamInterface:
    def __init__(self, save_dir="images", sequence_no="0"):
        # Inicializace kamery
        REX.TraceInfo(f"Inicializace kamery")
        self.camera = None
        try:
            # Create camera instance
            factory = pylon.TlFactory.GetInstance()
            device = factory.CreateFirstDevice()
            if device is None:
                REX.TraceError("Žádná kamera nebyla nalezena")
                return

            self.camera = pylon.InstantCamera(device)
            REX.TraceInfo(f"Používá se kamera: {self.camera.GetDeviceInfo().GetModelName()}")

            # Configure camera based on parameters
            if int(REX.p2.v) == 0:
                # Auto mode
                self.camera.Open()
                self.camera.Width.SetValue(self.camera.Width.GetMax())   # Nastaví šířku oblasti
                self.camera.Height.SetValue(self.camera.Height.GetMax())  # Nastaví výšku oblasti
                self.camera.ExposureAuto.SetValue("Continuous") # Automatická expozice
                self.camera.GainAuto.SetValue("Continuous")  # Automatické zesílení
                self.camera.BalanceWhiteAuto.SetValue("Continuous") # Automatický balanc bílé
                self.camera.PixelFormat.SetValue("BGR8") # Formmát
                self.camera.Close()
            else:
                # Manual mode
                self.camera.Open()
                self.camera.Width.SetValue(self.camera.Width.GetMax())   # Nastaví šířku oblasti
                self.camera.Height.SetValue(self.camera.Height.GetMax())  # Nastaví výšku oblasti
                self.camera.ExposureAuto.SetValue("Off")
                self.camera.ExposureTime.SetValue(float(REX.p3.v))
                self.camera.GainAuto.SetValue("Off")
                self.camera.Gain.SetValue(float(REX.p4.v))
                self.camera.BalanceWhiteAuto.SetValue("Once")
                self.camera.PixelFormat.SetValue("BGR8") # Formát
                self.camera.Close()

            REX.TraceInfo("Kamera úspěšně inicializována")
        except Exception as e:
            REX.TraceError(f"Chyba při inicializaci kamery: {e}")
            if self.camera is not None:
                if self.camera.IsOpen():
                    self.camera.Close()
                self.camera = None
            gc.collect()

        # Set up save directory
        self.save_dir = save_dir
        self.update_path(sequence_no)

    def update_path(self, sequence_no):
        self.sequence_no = sequence_no
        self.save_path  = os.path.join(self.save_dir, self.sequence_no)
        #self.create_unique_folder(self.save_path)
        self.create_folder_if_not_exists(self.save_path)

    def capture_frame(self, filename="image.png", sequence_no="0"):
        """ Otevře kameru, pořídí snímek a uloží jej na disk. """
        self.update_path(sequence_no)

        if self.camera is None:
            REX.TraceError("Kamera není k dispozici.")
            return None
        if self.camera.IsGrabbing():
            REX.TraceError("Kamera už snímá v jiném programu.")
            return None

        # Variables to track resources for cleanup
        grab_result = None
        img = None

        try:
            # Pořízení snímku
            self.camera.Open()
            self.camera.StartGrabbingMax(1)
            grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

            current_exposure = self.camera.ExposureTime.GetValue()
            current_gain = self.camera.Gain.GetValue()
            current_white_balance = "Continuous"

            name, ext = os.path.splitext(filename)
            if not ext:
                ext = ".png"
            if int(REX.p2.v) == 0:
                filename = f"{name}_Exp-{current_exposure}_Gain-{current_gain}_White-{current_white_balance}{ext}"
            else:
                filename = f"{name}{ext}"

            if grab_result.GrabSucceeded():
                # Get image data
                img = grab_result.Array.copy()  # Make a copy to avoid reference issues

                # Release grab result immediately to free memory
                grab_result.Release()
                grab_result = None

                if len(img.shape) == 2:  # Černobílý obrázek
                    pass  # `Mono8` je v pořádku
                elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB/BGR obrázek
                    pass  # RGB8 je také v pořádku
                else:
                    REX.TraceError("Formát obrázku není podporován OpenCV.")
                    return None

                savepath = os.path.join(self.save_path, filename)
                success = cv2.imwrite(savepath, img)
                if not success:
                    REX.TraceError(f"Ukládání obrázku selhalo pro soubor: {savepath}")
                    REX.TraceError(f"Datový typ obrázku: {img.dtype}, tvar: {img.shape}")
                else:
                    REX.TraceInfo(f"Snímek uložen do souboru: {savepath}")
            else:
                REX.TraceError("Nepodařilo se zachytit snímek.")
                if grab_result is not None:
                    grab_result.Release()
                    grab_result = None
        except Exception as e:
            REX.TraceError(f"Chyba při pořizování snímku: {e}")
        finally:
            # Clean up resources
            if grab_result is not None:
                grab_result.Release()

            # Close camera
            if self.camera is not None and self.camera.IsOpen():
                self.camera.Close()

            # Clean up image data
            if img is not None:
                del img

            # Force garbage collection
            gc.collect()

    def release(self):
        """ Uvolní kameru, pokud by byla stále otevřená. """
        try:
            if self.camera is not None:
                if self.camera.IsOpen():
                    self.camera.Close()
                # Set to None to help garbage collection
                self.camera = None
                REX.TraceInfo("Kamera uvolněna.")
        except Exception as e:
            REX.TraceError(f"Chyba při uvolňování kamery: {e}")
        finally:
            # Force garbage collection
            gc.collect()

    def create_unique_folder(self, base_folder):
        folder = base_folder
        counter = 1
        while os.path.exists(folder):
            folder = f"{base_folder}{counter}"
            counter += 1
        os.makedirs(folder)
        return folder

    def create_folder_if_not_exists(self, base_folder):
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        return base_folder

