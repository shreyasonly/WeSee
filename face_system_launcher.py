"""
Face System Launcher
===================

This launcher provides a menu to choose between:
1. Real-time face registration (Phase 1)
2. Enhanced face recognition for group photos
"""

import tkinter as tk
from tkinter import messagebox
import os
import sys
import tkinter.font as tkfont
from PIL import Image, ImageTk, ImageDraw

class FaceSystemLauncher:
    """
    Launcher for the face recognition system with menu options.
    """
    
    def __init__(self):
        """
        Initialize the launcher.
        """
        self.root = None
        self.create_launcher_window()
    
    def create_launcher_window(self):
        """
        Create the main launcher window with a rich, modern, maximizable UI.
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

        self.root = tk.Tk()
        self.root.title("Face Recognition System Launcher")
        self.root.geometry("700x520")
        self.root.configure(bg=BG)
        self.root.minsize(600, 420)
        self.root.resizable(True, True)

        # Center the window (optional, only on first open)
        self.root.eval('tk::PlaceWindow . center')

        # Fonts
        title_font = tkfont.Font(family="Segoe UI", size=24, weight="bold")
        subtitle_font = tkfont.Font(family="Segoe UI", size=13)
        button_font = tkfont.Font(family="Segoe UI", size=15, weight="bold")
        status_font = tkfont.Font(family="Segoe UI", size=11, slant="italic")

        # Card shadow
        card_shadow = tk.Frame(self.root, bg=SHADOW, bd=0, highlightthickness=0)
        card_shadow.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=480, height=420)
        # Main card
        card = tk.Frame(self.root, bg=CARD, bd=0, highlightthickness=0)
        card.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=460, height=400)

        # Avatar/graphic at the top
        avatar_img = Image.new('RGBA', (80, 80), (26, 35, 126, 255))
        draw = ImageDraw.Draw(avatar_img)
        draw.ellipse((0, 0, 80, 80), fill=(255, 214, 0, 255))
        draw.ellipse((18, 18, 62, 62), fill=(26, 35, 126, 255))
        avatar = ImageTk.PhotoImage(avatar_img)
        avatar_label = tk.Label(card, image=avatar, bg=CARD)
        avatar_label.image = avatar
        avatar_label.pack(pady=(28, 8))

        # Title
        title_label = tk.Label(
            card,
            text="Face Recognition System",
            font=title_font,
            bg=CARD,
            fg=PRIMARY
        )
        title_label.pack(pady=(0, 4))

        # Subtitle
        subtitle_label = tk.Label(
            card,
            text="Choose an option to get started",
            font=subtitle_font,
            bg=CARD,
            fg=SUBTEXT
        )
        subtitle_label.pack(pady=(0, 18))

        # Buttons frame
        button_frame = tk.Frame(card, bg=CARD)
        button_frame.pack(pady=10)

        def style_button(btn, bg, fg, hover_bg):
            btn.configure(bg=bg, fg=fg, activebackground=hover_bg, activeforeground=fg, relief=tk.FLAT, bd=0, font=button_font, cursor='hand2', highlightthickness=0)
            btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
            btn.bind("<Leave>", lambda e: btn.config(bg=bg))

        # Registration button
        registration_button = tk.Button(
            button_frame,
            text="Phase 1: Register New Faces",
            command=self.start_registration,
            padx=30, pady=15
        )
        style_button(registration_button, BTN, 'white', BTN_HOVER)
        registration_button.pack(pady=8, fill=tk.X, expand=True)

        # Video Recognition button
        video_recognition_button = tk.Button(
            button_frame,
            text="Phase 2: Real-Time Video Recognition",
            command=self.start_video_recognition,
            padx=30, pady=15
        )
        style_button(video_recognition_button, PRIMARY, 'white', BTN_HOVER)
        video_recognition_button.pack(pady=8, fill=tk.X, expand=True)

        # Attendance Viewer button
        attendance_button = tk.Button(
            button_frame,
            text="View Attendance Records",
            command=self.view_attendance,
            padx=20, pady=10
        )
        style_button(attendance_button, '#607D8B', 'white', '#455A64')
        attendance_button.pack(pady=8, fill=tk.X, expand=True)

        # View Registered Faces button
        view_button = tk.Button(
            button_frame,
            text="View Registered Faces",
            command=self.view_registered,
            padx=20, pady=10
        )
        style_button(view_button, ACCENT, PRIMARY, BTN_ACCENT_HOVER)
        view_button.pack(pady=8, fill=tk.X, expand=True)

        # Add 'Check Attendance Report' button to main menu
        check_attendance_btn = tk.Button(
            button_frame,
            text="Check Attendance Report",
            command=self.view_attendance,
            padx=30, pady=15
        )
        style_button(check_attendance_btn, BTN_ACCENT, PRIMARY, BTN_ACCENT_HOVER)
        check_attendance_btn.pack(pady=8, fill=tk.X, expand=True)

        # Exit button
        exit_button = tk.Button(
            button_frame,
            text="Exit",
            command=self.exit_system,
            padx=20, pady=10
        )
        style_button(exit_button, '#d32f2f', 'white', '#b71c1c')
        exit_button.pack(pady=8, fill=tk.X, expand=True)

        # Status bar at the bottom
        self.status_label = tk.Label(
            self.root,
            text="Ready to use face recognition system",
            font=status_font,
            bg=SHADOW,
            fg=TEXT,
            anchor='w',
            relief=tk.FLAT,
            padx=10
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 0))

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.exit_system)
    
    def start_registration(self):
        """
        Start the face registration system.
        """
        try:
            self.status_label.config(text="Starting face registration system...")
            self.root.update()
            
            # Import and start registration system
            from real_time_face_registration import FaceRegistrationSystem
            
            registration_system = FaceRegistrationSystem()
            registration_system.run()
            
            # Update status after registration
            self.status_label.config(text="Registration completed. Ready for recognition.")
            
        except ImportError as e:
            messagebox.showerror("Error", f"Could not import registration system: {str(e)}")
            self.status_label.config(text="Error loading registration system")
        except Exception as e:
            messagebox.showerror("Error", f"Error starting registration: {str(e)}")
            self.status_label.config(text="Error in registration system")
    
    def start_video_recognition(self):
        """
        Start the real-time video recognition system.
        """
        try:
            self.status_label.config(text="Starting real-time video recognition...")
            self.root.update()
            
            # Check if there are registered faces
            known_faces_dir = "known_faces"
            if not os.path.exists(known_faces_dir) or not os.listdir(known_faces_dir):
                response = messagebox.askyesno(
                    "No Registered Faces",
                    "No faces have been registered yet.\nWould you like to register faces first?"
                )
                if response:
                    self.start_registration()
                    return
                else:
                    self.status_label.config(text="No registered faces found")
                    return
            
            # Import and start video recognition system
            from real_time_video_recognition import RealTimeVideoRecognition
            
            video_recognition = RealTimeVideoRecognition()
            video_recognition.run()
            
            # Update status after video recognition
            self.status_label.config(text="Video recognition completed. Ready for next operation.")
            
        except ImportError as e:
            messagebox.showerror("Error", f"Could not import video recognition system: {str(e)}")
            self.status_label.config(text="Error loading video recognition system")
        except Exception as e:
            messagebox.showerror("Error", f"Error starting video recognition: {str(e)}")
            self.status_label.config(text="Error in video recognition system")
    
    def view_registered(self):
        """
        View registered faces.
        """
        known_faces_dir = "known_faces"
        
        if not os.path.exists(known_faces_dir):
            messagebox.showinfo("Info", "No registered faces found.")
            return
        
        registered_faces = []
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                registered_faces.append(name)
        
        if not registered_faces:
            messagebox.showinfo("Info", "No registered faces found.")
            return
        
        # Create registered faces window
        faces_window = tk.Toplevel(self.root)
        faces_window.title("Registered Faces")
        faces_window.geometry("400x500")
        faces_window.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(
            faces_window,
            text="Registered Faces",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=20)
        
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
        
        # Close button
        close_button = tk.Button(
            button_frame,
            text="Close",
            command=faces_window.destroy,
            font=("Arial", 12),
            bg='#666666',
            fg='white',
            padx=20,
            pady=10,
            relief=tk.FLAT
        )
        close_button.pack()
    
    def view_attendance(self):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        import csv
        import glob
        import os
        from datetime import date
        # Color palette
        PRIMARY = '#1a237e'
        ACCENT = '#ffd600'
        BG = '#f7fafc'
        CARD = '#ffffff'
        SHADOW = '#e0e1dd'
        BTN = '#283593'
        BTN_HOVER = '#3949ab'
        BTN_ACCENT = '#ffd600'
        BTN_ACCENT_HOVER = '#ffea00'
        TEXT = '#22223b'
        SUBTEXT = '#4a4e69'
        # Attendance Viewer Window
        att_win = tk.Toplevel(self.root)
        att_win.title("Attendance Records")
        att_win.geometry("800x540")
        att_win.configure(bg=BG)
        att_win.minsize(600, 400)
        # Card shadow
        card_shadow = tk.Frame(att_win, bg=SHADOW, bd=0, highlightthickness=0)
        card_shadow.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=700, height=420)
        card = tk.Frame(att_win, bg=CARD, bd=0, highlightthickness=0)
        card.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=680, height=400)
        # Title
        title_label = tk.Label(card, text="Attendance Records", font=("Segoe UI", 18, "bold"), bg=CARD, fg=PRIMARY)
        title_label.pack(pady=(24, 8))
        # Date selection
        date_label = tk.Label(card, text="Select Date:", font=("Segoe UI", 12), bg=CARD, fg=SUBTEXT)
        date_label.pack(pady=(0, 2))
        files = sorted(glob.glob("attendance_*.csv"))
        dates = [f.split("_")[1].split(".")[0] for f in files]
        selected_date = tk.StringVar(value=date.today().isoformat() if date.today().isoformat() in dates else (dates[-1] if dates else ""))
        date_combo = ttk.Combobox(card, values=dates, textvariable=selected_date, font=("Segoe UI", 12), state="readonly")
        date_combo.pack(pady=4)
        # Table
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', background=CARD, fieldbackground=CARD, foreground=TEXT, rowheight=28, font=("Segoe UI", 11))
        style.configure('Treeview.Heading', font=("Segoe UI", 12, "bold"), background=PRIMARY, foreground='white')
        tree = ttk.Treeview(card, columns=("Name", "Date", "Time"), show="headings", height=8)
        tree.heading("Name", text="Name")
        tree.heading("Date", text="Date")
        tree.heading("Time", text="Time")
        tree.column("Name", width=200, anchor=tk.CENTER)
        tree.column("Date", width=120, anchor=tk.CENTER)
        tree.column("Time", width=120, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        # Export button
        def style_button(btn, bg, fg, hover_bg):
            btn.configure(bg=bg, fg=fg, activebackground=hover_bg, activeforeground=fg, relief=tk.FLAT, bd=0, font=("Segoe UI", 12, "bold"), cursor='hand2', highlightthickness=0)
            btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
            btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        def export_csv():
            if not tree.get_children():
                messagebox.showinfo("Export", "No records to export.")
                return
            file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if file:
                with open(file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Name", "Date", "Time"])
                    for row in tree.get_children():
                        writer.writerow(tree.item(row)['values'])
                messagebox.showinfo("Export", f"Attendance exported to {file}")
        export_btn = tk.Button(card, text="Export as CSV", command=export_csv, padx=18, pady=8)
        style_button(export_btn, BTN_ACCENT, PRIMARY, BTN_ACCENT_HOVER)
        export_btn.pack(pady=(0, 10))
        # Export all attendance button
        def export_all_csv():
            import glob
            all_files = sorted(glob.glob("attendance_*.csv"))
            if not all_files:
                messagebox.showinfo("Export", "No attendance records to export.")
                return
            file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], title="Save All Attendance As")
            if file:
                with open(file, 'w', newline='') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(["Name", "Date", "Time"])
                    for att_file in all_files:
                        with open(att_file, 'r') as f_in:
                            reader = csv.reader(f_in)
                            next(reader, None)  # skip header
                            for row in reader:
                                writer.writerow(row)
                messagebox.showinfo("Export", f"All attendance exported to {file}")
        export_all_btn = tk.Button(card, text="Export All Attendance (CSV)", command=export_all_csv, padx=18, pady=8)
        style_button(export_all_btn, BTN, 'white', BTN_HOVER)
        export_all_btn.pack(pady=(0, 10))
        # Load records
        def load_records(*_):
            tree.delete(*tree.get_children())
            sel = selected_date.get()
            if not sel:
                return
            file = f"attendance_{sel}.csv"
            if os.path.exists(file):
                with open(file, 'r') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        tree.insert('', tk.END, values=row)
        date_combo.bind('<<ComboboxSelected>>', load_records)
        load_records()
        # Add close button
        close_btn = tk.Button(card, text="Close", command=att_win.destroy, padx=18, pady=8)
        style_button(close_btn, '#d32f2f', 'white', '#b71c1c')
        close_btn.pack(pady=(0, 18))
    
    def exit_system(self):
        """
        Exit the launcher system.
        """
        if self.root:
            self.root.destroy()
        print("Face System Launcher closed.")
    
    def run(self):
        """
        Run the launcher.
        """
        self.root.mainloop()


def main():
    """
    Main function to run the face system launcher.
    """
    print("Starting Face Recognition System Launcher...")
    print("This launcher provides access to both registration and recognition systems.")
    print("=" * 60)
    
    # Create and run the launcher
    launcher = FaceSystemLauncher()
    launcher.run()


if __name__ == "__main__":
    main() 