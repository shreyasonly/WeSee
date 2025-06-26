"""
Real-Time Face Registration System
=================================

This system allows users to register their face using their device camera
with a local dialog box interface for Phase 1 registration.
"""

import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import os
import time
import threading
from datetime import datetime
import mediapipe as mp
import numpy as np
import tkinter.font as tkfont
from PIL import Image, ImageTk, ImageDraw

class FaceRegistrationSystem:
    """
    Real-time face registration system with camera and dialog interface.
    """
    
    def __init__(self, known_faces_dir="known_faces"):
        """
        Initialize the face registration system.
        
        Args:
            known_faces_dir (str): Directory to save registered faces
        """
        self.known_faces_dir = known_faces_dir
        self.cap = None
        self.is_capturing = False
        self.registration_window = None
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create known faces directory
        os.makedirs(known_faces_dir, exist_ok=True)
        
        # Initialize face detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7
        )
        
        print("Face Registration System initialized!")
    
    def start_camera(self):
        """
        Start the camera for face registration.
        """
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return False
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def stop_camera(self):
        """
        Stop the camera.
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def detect_face_in_frame(self, frame):
        """
        Detect faces in a frame and return the best face for registration.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (face_detected, face_bbox, face_quality_score)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        if not results.detections:
            return False, None, 0
        
        # Get the best face (largest and most centered)
        best_face = None
        best_score = 0
        
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            # Convert to absolute coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Calculate face quality score
            face_area = width * height
            face_center_x = x + width // 2
            face_center_y = y + height // 2
            
            # Distance from center (closer is better)
            center_distance = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
            
            # Quality score based on size and centering
            quality_score = face_area - (center_distance * 0.1)
            
            if quality_score > best_score:
                best_score = quality_score
                best_face = (x, y, width, height)
        
        return True, best_face, best_score
    
    def create_registration_dialog(self):
        """
        Create a step-by-step registration wizard with a rich, modern color theme and embedded camera feed.
        Use Toplevel() if a root window exists, otherwise Tk().
        """
        # Color palette
        PRIMARY = '#1a237e'  # Deep blue
        ACCENT = '#ffd600'   # Gold
        BG = '#f7fafc'       # Light background
        CARD = '#ffffff'     # Card background
        SHADOW = '#e0e1dd'   # Subtle shadow
        BTN = '#283593'      # Button blue
        BTN_HOVER = '#3949ab'
        BTN_ACCENT = '#ffd600'
        BTN_ACCENT_HOVER = '#ffea00'
        TEXT = '#22223b'
        SUBTEXT = '#4a4e69'
        SUCCESS = '#43a047'
        ERROR = '#d32f2f'

        try:
            self.registration_window = tk.Toplevel()
            self.registration_window.transient()  # Optional: keep on top
        except Exception:
            self.registration_window = tk.Tk()
        self.registration_window.title("Face Registration Wizard")
        self.registration_window.geometry("520x600")
        self.registration_window.configure(bg=BG)
        self.registration_window.resizable(False, False)
        
        # Center the window
        if isinstance(self.registration_window, tk.Tk):
            self.registration_window.eval('tk::PlaceWindow . center')
        else:
            # Manual centering for Toplevel
            self.registration_window.update_idletasks()
            w = self.registration_window.winfo_width()
            h = self.registration_window.winfo_height()
            ws = self.registration_window.winfo_screenwidth()
            hs = self.registration_window.winfo_screenheight()
            x = (ws // 2) - (w // 2)
            y = (hs // 2) - (h // 2)
            self.registration_window.geometry(f'+{x}+{y}')

        # Fonts
        title_font = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        step_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        label_font = tkfont.Font(family="Segoe UI", size=12)
        button_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        status_font = tkfont.Font(family="Segoe UI", size=11, slant="italic")

        # Wizard state
        self._wizard_step = 0
        self._wizard_name = ""
        self._wizard_capture_imgtk = None
        self._wizard_camera_running = False
        self._wizard_capture_frame = None
        self._wizard_success = False

        # Main card frame
        self._card = tk.Frame(self.registration_window, bg=CARD, bd=0, highlightthickness=0)
        self._card.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=440, height=520)
        self._card_shadow = tk.Frame(self.registration_window, bg=SHADOW, bd=0, highlightthickness=0)
        self._card_shadow.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=450, height=530)
        self._card.lift()

        # Progress indicator
        self._progress = tk.Canvas(self._card, width=320, height=18, bg=CARD, highlightthickness=0)
        self._progress.place(x=60, y=18)
        self._draw_progress(0)

        # Step container
        self._step_frame = tk.Frame(self._card, bg=CARD)
        self._step_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=400, height=440)

        # Status bar
        self.status_label = tk.Label(
            self.registration_window,
            text="Welcome to the Face Registration Wizard",
            font=status_font,
            bg=SHADOW,
            fg=TEXT,
            anchor='w',
            relief=tk.FLAT,
            padx=10
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 0))

        # Start wizard
        self._show_wizard_step(0)
        self.registration_window.protocol("WM_DELETE_WINDOW", self.exit_system)

    def _draw_progress(self, step):
        # Draw a 4-step progress bar
        self._progress.delete("all")
        colors = ['#1a237e', '#283593', '#3949ab', '#ffd600']
        for i in range(4):
            fill = colors[i] if i <= step else '#e0e1dd'
            self._progress.create_oval(10 + i*100, 2, 28 + i*100, 20, fill=fill, outline=fill)
            if i < 3:
                self._progress.create_rectangle(28 + i*100, 9, 82 + i*100, 13, fill=fill if i < step else '#e0e1dd', outline='')

    def _show_wizard_step(self, step):
        # Clear previous widgets
        for widget in self._step_frame.winfo_children():
            widget.destroy()
        self._draw_progress(step)
        if step == 0:
            self._wizard_step0()
        elif step == 1:
            self._wizard_step1()
        elif step == 2:
            self._wizard_step2()
        elif step == 3:
            self._wizard_step3()
        self._wizard_step = step

    def _wizard_step0(self):
        # Welcome step
        avatar_img = Image.new('RGBA', (90, 90), (26, 35, 126, 255))
        draw = ImageDraw.Draw(avatar_img)
        draw.ellipse((0, 0, 90, 90), fill=(255, 214, 0, 255))
        draw.ellipse((20, 20, 70, 70), fill=(26, 35, 126, 255))
        avatar = ImageTk.PhotoImage(avatar_img)
        avatar_label = tk.Label(self._step_frame, image=avatar, bg='#ffffff')
        avatar_label.image = avatar
        avatar_label.pack(pady=(30, 10))
        title = tk.Label(self._step_frame, text="Welcome!", font=("Segoe UI", 20, "bold"), bg='#ffffff', fg='#1a237e')
        title.pack(pady=(0, 6))
        subtitle = tk.Label(self._step_frame, text="Let's get you registered for face recognition.", font=("Segoe UI", 12), bg='#ffffff', fg='#4a4e69')
        subtitle.pack(pady=(0, 18))
        next_btn = tk.Button(self._step_frame, text="Get Started →", command=lambda: self._show_wizard_step(1), font=("Segoe UI", 13, "bold"), bg='#ffd600', fg='#1a237e', activebackground='#ffea00', activeforeground='#1a237e', relief=tk.FLAT, bd=0, padx=18, pady=8, cursor='hand2')
        next_btn.pack(pady=(30, 0))

    def _wizard_step1(self):
        # Name entry step
        label = tk.Label(self._step_frame, text="Enter your full name", font=("Segoe UI", 15, "bold"), bg='#ffffff', fg='#1a237e')
        label.pack(pady=(40, 10))
        self._name_var = tk.StringVar()
        entry = tk.Entry(self._step_frame, textvariable=self._name_var, font=("Segoe UI", 13), width=24, relief=tk.FLAT, highlightthickness=2, highlightbackground="#e0e1dd", highlightcolor="#1a237e", bg="#f0f4f8")
        entry.pack(ipady=7)
        entry.focus_set()
        self._name_error = tk.Label(self._step_frame, text="", font=("Segoe UI", 10), fg='#d32f2f', bg='#ffffff')
        self._name_error.pack(pady=(6, 0))
        next_btn = tk.Button(self._step_frame, text="Next →", command=self._wizard_validate_name, font=("Segoe UI", 13, "bold"), bg='#ffd600', fg='#1a237e', activebackground='#ffea00', activeforeground='#1a237e', relief=tk.FLAT, bd=0, padx=18, pady=8, cursor='hand2')
        next_btn.pack(pady=(30, 0))

    def _wizard_validate_name(self):
        name = self._name_var.get().strip()
        if not name or len(name) < 2:
            self._name_error.config(text="Please enter a valid name.")
            return
        # Check for duplicate
        existing_file = os.path.join(self.known_faces_dir, f"{name}.jpg")
        if os.path.exists(existing_file):
            self._name_error.config(text=f"'{name}' is already registered.")
            return
        self._wizard_name = name
        self._show_wizard_step(2)

    def _wizard_step2(self):
        # Camera preview & capture step
        label = tk.Label(self._step_frame, text="Camera Preview", font=("Segoe UI", 15, "bold"), bg='#ffffff', fg='#1a237e')
        label.pack(pady=(18, 6))
        self._camera_canvas = tk.Canvas(self._step_frame, width=320, height=240, bg='#e0e1dd', bd=0, highlightthickness=0)
        self._camera_canvas.pack(pady=(0, 10))
        self._wizard_camera_running = True
        self._wizard_capture_imgtk = None
        self._wizard_capture_frame = None
        self._wizard_capture_countdown = 3
        self._wizard_capture_status = tk.Label(self._step_frame, text="Position your face in the frame", font=("Segoe UI", 11), bg='#ffffff', fg='#4a4e69')
        self._wizard_capture_status.pack(pady=(0, 10))
        capture_btn = tk.Button(self._step_frame, text="Capture Photo", command=self._wizard_capture_photo, font=("Segoe UI", 13, "bold"), bg='#ffd600', fg='#1a237e', activebackground='#ffea00', activeforeground='#1a237e', relief=tk.FLAT, bd=0, padx=18, pady=8, cursor='hand2')
        capture_btn.pack(pady=(10, 0))
        self._wizard_update_camera()

    def _wizard_update_camera(self):
        if not self._wizard_camera_running:
            return
        if self.cap is None:
            self.start_camera()
        ret, frame = self.cap.read() if self.cap else (False, None)
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((320, 240))
            imgtk = ImageTk.PhotoImage(image=img)
            self._camera_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self._wizard_capture_imgtk = imgtk
            self._wizard_capture_frame = frame
        self._step_frame.after(30, self._wizard_update_camera)

    def _wizard_capture_photo(self):
        self._wizard_camera_running = False
        if self.cap:
            self.cap.release()
        # Detect face in the captured frame and save
        frame = self._wizard_capture_frame
        name = self._wizard_name
        if frame is not None and name:
            face_detected, face_bbox, face_score = self.detect_face_in_frame(frame)
            if face_detected and face_bbox:
                self.save_registered_face(name, frame, face_bbox, is_fallback=False)
            else:
                self.registration_window.after(0, lambda: messagebox.showerror(
                    "Error", "No face detected in the captured photo. Please try again."
                ))
        self._show_wizard_step(3)

    def _wizard_step3(self):
        # Success step
        label = tk.Label(self._step_frame, text="Registration Complete!", font=("Segoe UI", 15, "bold"), bg='#ffffff', fg='#43a047')
        label.pack(pady=(40, 10))
        avatar_img = Image.new('RGBA', (90, 90), (67, 160, 71, 255))
        draw = ImageDraw.Draw(avatar_img)
        draw.ellipse((0, 0, 90, 90), fill=(255, 214, 0, 255))
        draw.ellipse((20, 20, 70, 70), fill=(67, 160, 71, 255))
        avatar = ImageTk.PhotoImage(avatar_img)
        avatar_label = tk.Label(self._step_frame, image=avatar, bg='#ffffff')
        avatar_label.image = avatar
        avatar_label.pack(pady=(0, 10))
        name_label = tk.Label(self._step_frame, text=f"Welcome, {self._wizard_name}!", font=("Segoe UI", 13), bg='#ffffff', fg='#1a237e')
        name_label.pack(pady=(0, 10))
        done_btn = tk.Button(self._step_frame, text="Finish", command=self.exit_system, font=("Segoe UI", 13, "bold"), bg='#ffd600', fg='#1a237e', activebackground='#ffea00', activeforeground='#1a237e', relief=tk.FLAT, bd=0, padx=18, pady=8, cursor='hand2')
        done_btn.pack(pady=(30, 0))

    def save_registered_face(self, name, frame, face_bbox, is_fallback=False):
        """
        Save the registered face to the database.
        
        Args:
            name (str): Name of the person
            frame (np.ndarray): Frame containing the face
            face_bbox (tuple): Face bounding box (x, y, w, h)
            is_fallback (bool): Whether the capture is a fallback
        """
        try:
            x, y, w, h = face_bbox
            
            # Extract face region with some padding
            padding = int(min(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_img = frame[y1:y2, x1:x2]
            
            # Save face image
            filename = f"{name}_{'fallback' if is_fallback else 'main'}.jpg"
            filepath = os.path.join(self.known_faces_dir, filename)
            cv2.imwrite(filepath, face_img)
            
            # Show success message based on capture type
            if is_fallback:
                message = f"Face registered with fallback capture for '{name}'!\nQuality was below 60% for 15 seconds.\nSaved as: {filename}\nNote: Recognition accuracy may be lower."
            else:
                message = f"Face registered successfully for '{name}'!\nSaved as: {filename}"
            
            self.registration_window.after(0, lambda: messagebox.showinfo(
                "Success", message
            ))
            
            print(f"Face registered for: {name} ({'fallback' if is_fallback else 'normal'})")
            
        except Exception as e:
            self.registration_window.after(0, lambda: messagebox.showerror(
                "Error", f"Failed to save face: {str(e)}"
            ))
    
    def view_registered_faces(self):
        """
        Show a dialog with all registered faces.
        """
        if not os.path.exists(self.known_faces_dir):
            messagebox.showinfo("Info", "No registered faces found.")
            return
        
        registered_faces = []
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                registered_faces.append(name)
        
        if not registered_faces:
            messagebox.showinfo("Info", "No registered faces found.")
            return
        
        # Create registered faces window
        faces_window = tk.Toplevel(self.registration_window)
        faces_window.title("Registered Faces")
        faces_window.geometry("300x400")
        faces_window.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(
            faces_window,
            text="Registered Faces",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=10)
        
        # Create scrollable list
        listbox_frame = tk.Frame(faces_window, bg='#f0f0f0')
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            font=("Arial", 12),
            selectmode=tk.SINGLE
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=listbox.yview)
        
        # Add registered faces to list
        for i, name in enumerate(registered_faces):
            listbox.insert(tk.END, f"{i+1}. {name}")
        
        # Buttons frame
        button_frame = tk.Frame(faces_window, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        # Delete button
        delete_button = tk.Button(
            button_frame,
            text="Delete Selected",
            command=lambda: self.delete_face(listbox, registered_faces, faces_window),
            font=("Arial", 10),
            bg='#f44336',
            fg='white',
            padx=15,
            pady=5,
            relief=tk.FLAT
        )
        delete_button.pack(side=tk.LEFT, padx=5)
        
        # Close button
        close_button = tk.Button(
            button_frame,
            text="Close",
            command=faces_window.destroy,
            font=("Arial", 10),
            bg='#666666',
            fg='white',
            padx=15,
            pady=5,
            relief=tk.FLAT
        )
        close_button.pack(side=tk.LEFT, padx=5)
    
    def add_photo_manually(self):
        """
        Allow users to add photos manually by selecting files from their computer.
        """
        # Ask for the person's name
        name = simpledialog.askstring("Add Photo", "Enter the person's name:")
        if not name or not name.strip():
            return
        
        name = name.strip()
        
        # Check if name already exists
        existing_file = os.path.join(self.known_faces_dir, f"{name}.jpg")
        if os.path.exists(existing_file):
            response = messagebox.askyesno(
                "Name Exists",
                f"A person named '{name}' is already registered.\nDo you want to update their photo?"
            )
            if not response:
                return
        
        # Open file dialog to select photo
        file_path = filedialog.askopenfilename(
            title="Select a photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Could not read the selected image file.")
                return
            
            # Detect face in the image
            face_detected, face_bbox, face_score = self.detect_face_in_frame(image)
            
            if not face_detected or not face_bbox:
                messagebox.showerror("Error", "No face detected in the selected image.\nPlease select an image with a clear, front-facing face.")
                return
            
            # Check face quality
            quality_percentage = min(100, int(face_score / 1000))
            if quality_percentage < 50:
                response = messagebox.askyesno(
                    "Low Quality",
                    f"Face quality is low ({quality_percentage}%).\nThis may affect recognition accuracy.\nDo you want to continue anyway?"
                )
                if not response:
                    return
            
            # Save the face
            self.save_registered_face(name, image, face_bbox)
            
            # Update status
            self.status_label.config(text=f"Photo added for '{name}'! Ready for next registration.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def delete_face(self, listbox, registered_faces, window):
        """
        Delete a selected registered face.
        """
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a face to delete.")
            return
        
        index = selection[0]
        name = registered_faces[index]
        
        response = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete '{name}' from registered faces?"
        )
        
        if response:
            try:
                filename = f"{name}.jpg"
                filepath = os.path.join(self.known_faces_dir, filename)
                os.remove(filepath)
                
                # Update list
                listbox.delete(index)
                registered_faces.pop(index)
                
                messagebox.showinfo("Success", f"'{name}' has been deleted.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete face: {str(e)}")
    
    def exit_system(self):
        """
        Exit the registration system.
        """
        if self.is_capturing:
            self.is_capturing = False
            self.stop_camera()
        
        if self.registration_window:
            self.registration_window.destroy()
        
        print("Face Registration System closed.")
    
    def run(self):
        """
        Run the face registration system.
        """
        self.create_registration_dialog()
        self.registration_window.mainloop()


def main():
    """
    Main function to run the face registration system.
    """
    print("Starting Face Registration System...")
    print("This system allows you to register faces using your camera.")
    print("=" * 50)
    
    # Create and run the registration system
    registration_system = FaceRegistrationSystem()
    registration_system.run()


if __name__ == "__main__":
    main() 