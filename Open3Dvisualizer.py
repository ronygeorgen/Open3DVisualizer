import open3d as o3d
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
import numpy as np
import os
import sys
from PIL import Image, ImageTk
import threading
import tempfile
import time
import sys



class PointCloudViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Viewer")
        self.root.geometry("1200x800")
        
        #point cloud variables
        self.current_point_cloud = None
        self.vis = None
        self.running = False
        
        self.bg_color = [1.0, 1.0, 1.0]  
        self.point_color = [0.9, 0.9, 0.9]  
        self.point_size = 3
        self.show_axes = False
        self.show_skymap = False
        
        self.create_menu_bar()
        
        # main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create visualization area
        self.viz_frame = ttk.Frame(self.main_frame, borderwidth=2, relief="sunken")
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the canvas for Open3D rendering
        self.canvas = tk.Canvas(self.viz_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create control panel
        self.control_panel = ttk.Frame(self.main_frame, width=300)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add control sections
        self.create_view_controls()
        self.create_material_settings()
        
        # Initialize Open3D visualizer
        self.init_open3d()
        
        
        
    def create_menu_bar(self):
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open...", command=self.open_file)
        file_menu.add_command(label="Open Samples...", command=self.open_samples)
        file_menu.add_command(label="Export Current Image...", command=self.export_image)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit_application)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Settings menu
        settings_menu = tk.Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="General Settings", command=self.open_general_settings)
        settings_menu.add_command(label="AI Settings", command=self.open_ai_settings)
        menu_bar.add_cascade(label="Settings", menu=settings_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
    
    def create_view_controls(self):
        # View controls section
        view_frame = ttk.LabelFrame(self.control_panel, text="View controls")
        view_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Mouse controls
        ttk.Label(view_frame, text="Mouse controls").pack(anchor=tk.W, padx=5)
        mouse_frame = ttk.Frame(view_frame)
        mouse_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(mouse_frame, text="Arcball", command=self.set_arcball_mode).pack(side=tk.LEFT, padx=2)
        ttk.Button(mouse_frame, text="Fly", command=self.set_fly_mode).pack(side=tk.LEFT, padx=2)
        ttk.Button(mouse_frame, text="Model", command=self.set_model_mode).pack(side=tk.LEFT, padx=2)
        
        # Environment controls
        env_frame = ttk.Frame(view_frame)
        env_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(env_frame, text="Sun", command=self.toggle_sun).pack(side=tk.LEFT, padx=2)
        ttk.Button(env_frame, text="Environment", command=self.toggle_environment).pack(side=tk.LEFT, padx=2)
        
        # Show skymap
        skymap_frame = ttk.Frame(view_frame)
        skymap_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.skymap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(skymap_frame, text="Show skymap", variable=self.skymap_var, command=self.toggle_skymap).pack(anchor=tk.W)
        
        # Background color
        bg_frame = ttk.Frame(view_frame)
        bg_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(bg_frame, text="BG Color").pack(side=tk.LEFT, padx=2)
        
        self.bg_r_entry = ttk.Entry(bg_frame, width=5, justify=tk.CENTER)
        self.bg_r_entry.insert(0, "R:255")
        self.bg_r_entry.pack(side=tk.LEFT, padx=2)
        
        self.bg_g_entry = ttk.Entry(bg_frame, width=5, justify=tk.CENTER)
        self.bg_g_entry.insert(0, "G:255")
        self.bg_g_entry.pack(side=tk.LEFT, padx=2)
        
        self.bg_b_entry = ttk.Entry(bg_frame, width=5, justify=tk.CENTER)
        self.bg_b_entry.insert(0, "B:255")
        self.bg_b_entry.pack(side=tk.LEFT, padx=2)
        
        self.bg_color_button = ttk.Button(bg_frame, text="", width=2, command=self.choose_bg_color)
        self.bg_color_button.pack(side=tk.LEFT, padx=2)
        
        # Apply color button
        ttk.Button(bg_frame, text="Apply", command=self.apply_bg_color).pack(side=tk.LEFT, padx=5)
        
        # Show axes
        self.axes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(view_frame, text="Show axes", variable=self.axes_var, command=self.toggle_axes).pack(anchor=tk.W, padx=5)
        
        # Lighting profiles
        lighting_frame = ttk.Frame(view_frame)
        lighting_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(lighting_frame, text="Lighting profiles").pack(anchor=tk.W)
        self.lighting_combo = ttk.Combobox(lighting_frame, values=["Bright day with sun at +Y [default]", "Cloudy day", "Night", "Custom"])
        self.lighting_combo.current(0)
        self.lighting_combo.pack(fill=tk.X, pady=2)
        self.lighting_combo.bind("<<ComboboxSelected>>", self.change_lighting)
        
        # Advanced lighting (able to collapse)
        self.advanced_lighting_var = tk.BooleanVar(value=False)
        self.advanced_lighting_button = ttk.Button(view_frame, text="▶ Advanced lighting", command=self.toggle_advanced_lighting)
        self.advanced_lighting_button.pack(anchor=tk.W, padx=5, pady=2)
        
        # Advanced lighting frame (hidden initially)
        self.advanced_lighting_frame = ttk.Frame(view_frame)
        
    def create_material_settings(self):
        # Material settings section
        material_frame = ttk.LabelFrame(self.control_panel, text="Material settings")
        material_frame.pack(fill=tk.X, padx=5, pady=5)
        
        type_frame = ttk.Frame(material_frame)
        type_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(type_frame, text="Type", width=10).pack(side=tk.LEFT)
        self.type_combo = ttk.Combobox(type_frame, values=["Lit", "Unlit", "Normal map", "Depth"])
        self.type_combo.current(0)
        self.type_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.type_combo.bind("<<ComboboxSelected>>", self.change_material_type)
        
        # Material
        material_frame_row = ttk.Frame(material_frame)
        material_frame_row.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(material_frame_row, text="Material", width=10).pack(side=tk.LEFT)
        self.material_combo = ttk.Combobox(material_frame_row, values=["Polished ceramic [default]","Metal", "Plastic", "Custom"])
        self.material_combo.current(0)
        self.material_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.material_combo.bind("<<ComboboxSelected>>", self.change_material)
        
        # Color
        color_frame = ttk.Frame(material_frame)
        color_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(color_frame, text="Color", width=10).pack(side=tk.LEFT)
        
        self.color_r_entry = ttk.Entry(color_frame, width=5, justify=tk.CENTER)
        self.color_r_entry.insert(0, "R:230")
        self.color_r_entry.pack(side=tk.LEFT, padx=2)
        
        self.color_g_entry = ttk.Entry(color_frame, width=5, justify=tk.CENTER)
        self.color_g_entry.insert(0, "G:230")
        self.color_g_entry.pack(side=tk.LEFT, padx=2)
        
        self.color_b_entry = ttk.Entry(color_frame, width=5, justify=tk.CENTER)
        self.color_b_entry.insert(0, "B:230")
        self.color_b_entry.pack(side=tk.LEFT, padx=2)
        
        self.color_button = ttk.Button(color_frame, text="", width=2, command=self.choose_point_color)
        self.color_button.pack(side=tk.LEFT, padx=2)
        
        # Apply color button
        ttk.Button(color_frame, text="Apply", command=self.apply_point_color).pack(side=tk.LEFT, padx=5)
        
        # Point size
        point_frame = ttk.Frame(material_frame)
        point_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(point_frame, text="Point size", width=10).pack(side=tk.LEFT)
        self.point_scale = ttk.Scale(point_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=self.update_point_size)
        self.point_scale.set(3)
        self.point_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.point_value = ttk.Label(point_frame, text="3")
        self.point_value.pack(side=tk.LEFT, padx=5)
        print("Created point_value label")

    def init_open3d(self):
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(visible=False, width=800, height=600)
            
            self.opt = self.vis.get_render_option()
            self.opt.background_color = np.array(self.bg_color)
            self.opt.point_size = self.point_size
            
            # Set up view control
            self.view_control = self.vis.get_view_control()
            
            # Start visualization thread
            self.running = True
            self.vis_thread = threading.Thread(target=self.visualization_loop)
            self.vis_thread.daemon = True
            self.vis_thread.start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize Open3D: {str(e)}")
            print(f"Open3D initialization error: {e}")
        
    def visualization_loop(self):
        time.sleep(0.5)  
        
        temp_dir = tempfile.gettempdir()
        img_path = os.path.join(temp_dir, "open3d_render.png")
        
        while self.running:
            try:
                if self.vis:
                    # Render and capture the image
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    self.vis.capture_screen_image(img_path, do_render=True)
                    
                    try:
                        # Check if canvas exists and has non-zero dimensions
                        canvas_width = self.canvas.winfo_width()
                        canvas_height = self.canvas.winfo_height()
                        
                        if canvas_width > 0 and canvas_height > 0:
                            # Load the image and display in tkinter
                            img = Image.open(img_path)
                            img = img.resize((canvas_width, canvas_height), Image.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            
                            # Update canvas with new image
                            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                            self.canvas.image = photo  # Keep reference to avoid garbage collection
                    
                    except Exception as e:
                        print(f"Error updating canvas: {e}")
                    
            except Exception as e:
                print(f"Error in visualization loop: {e}")
            
            # Add small sleep to prevent excessive CPU usage
            time.sleep(0.05)
        
    def load_demo_point_cloud(self):
        try:
            # Create a demo point cloud (bunny, sphere, or cube)
            pcd = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            pcd = pcd.sample_points_uniformly(number_of_points=5000)
            pcd.paint_uniform_color([0.9, 0.9, 0.9])  # Light gray
            
            # Add to the visualizer
            self.current_point_cloud = pcd
            self.vis.add_geometry(pcd)
            self.view_control.set_zoom(0.8)
            
            self.status_bar.config(text="Demo point cloud loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load demo point cloud: {str(e)}")
    
    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Point Cloud Files", "*.pcd *.ply *.xyz *.pts"), ("All Files", "*.*")]
        )
        if file_path:
            self.load_point_cloud(file_path)
    
    def load_point_cloud(self, file_path):
        try:
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Load the point cloud based on file type
            if file_ext == '.pcd':
                pcd = o3d.io.read_point_cloud(file_path)
            elif file_ext == '.ply':
                pcd = o3d.io.read_point_cloud(file_path)
            elif file_ext == '.xyz' or file_ext == '.pts':
                pcd = o3d.io.read_point_cloud(file_path, format='xyz')
            else:
                messagebox.showerror("Error", "Unsupported file format")
                return
            
            # Check if we have points
            if not pcd.has_points():
                messagebox.showerror("Error", "The file contains no points")
                return
                
            # Remove previous point cloud if exists
            if self.current_point_cloud is not None:
                self.vis.remove_geometry(self.current_point_cloud)
            
            # Assign colors if none exist
            if not pcd.has_colors():
                pcd.paint_uniform_color(self.point_color)
                
            # Add new point cloud
            self.current_point_cloud = pcd
            self.vis.add_geometry(pcd)
            
            # Reset view
            self.view_control.reset_camera_to_default()
            self.view_control.set_zoom(0.8)
            
            self.status_bar.config(text=f"Loaded point cloud from {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load point cloud: {str(e)}")
    
    def open_samples(self):
        # Create a samples dialog
        samples_dialog = tk.Toplevel(self.root)
        samples_dialog.title("Sample Point Clouds")
        samples_dialog.geometry("300x200")
        samples_dialog.resizable(False, False)
        
        # Sample options
        ttk.Label(samples_dialog, text="Select a sample point cloud:").pack(padx=10, pady=10)
        
        samples = ["Sphere", "Cube", "Cylinder", "Bunny", "Armadillo"]
        sample_var = tk.StringVar()
        sample_var.set(samples[0])
        
        for sample in samples:
            ttk.Radiobutton(samples_dialog, text=sample, variable=sample_var, value=sample).pack(anchor=tk.W, padx=20)
        
        # Load button
        def load_sample():
            sample = sample_var.get()
            self.load_sample_point_cloud(sample)
            samples_dialog.destroy()
            
        ttk.Button(samples_dialog, text="Load Sample", command=load_sample).pack(pady=10)
    
    def load_sample_point_cloud(self, sample_name):
        try:
            # Remove previous point cloud if exists
            if self.current_point_cloud is not None:
                self.vis.remove_geometry(self.current_point_cloud)
            
            # Create the sample point cloud
            if sample_name == "Sphere":
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                pcd = mesh.sample_points_uniformly(number_of_points=5000)
            elif sample_name == "Cube":
                mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
                pcd = mesh.sample_points_uniformly(number_of_points=5000)
            elif sample_name == "Cylinder":
                mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=2.0)
                pcd = mesh.sample_points_uniformly(number_of_points=5000)
            elif sample_name == "Bunny":
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                pcd = mesh.sample_points_uniformly(number_of_points=5000)
            elif sample_name == "Armadillo":
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                pcd = mesh.sample_points_uniformly(number_of_points=5000)
            else:
                messagebox.showerror("Error", "Unknown sample")
                return
            
            # Paint with default color
            pcd.paint_uniform_color(self.point_color)
            
            # Add to visualizer
            self.current_point_cloud = pcd
            self.vis.add_geometry(pcd)
            
            # Reset view
            self.view_control.reset_camera_to_default()
            self.view_control.set_zoom(0.8)
            
            self.status_bar.config(text=f"Loaded sample point cloud: {sample_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample point cloud: {str(e)}")
    
    def export_image(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.vis.capture_screen_image(file_path, True)
                self.status_bar.config(text=f"Image saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export image: {str(e)}")
    
    def set_arcball_mode(self):
        # Set camera control to arcball mode
        if self.view_control:
            self.view_control.change_field_of_view(step=0) 
            self.status_bar.config(text="Camera control: Arcball")
    
    def set_fly_mode(self):
        # Set camera control to fly mode
        if self.view_control:
            self.status_bar.config(text="Camera control: Fly (not implemented)")
    
    def set_model_mode(self):
        # Set camera control to model mode
        if self.view_control:
            self.status_bar.config(text="Camera control: Model (not implemented)")
    
    def toggle_sun(self):
        # Toggle sun lighting
        if self.opt:
            # Placeholder for sun lighting toggle
            self.status_bar.config(text="Sun lighting toggled")
    
    def toggle_environment(self):
        # Toggle environment lighting
        if self.opt:
            # Placeholder for environment lighting toggle
            self.status_bar.config(text="Environment lighting toggled")
    
    def toggle_skymap(self):
        # Toggle skymap display
        if self.opt:
            self.show_skymap = self.skymap_var.get()
            self.status_bar.config(text=f"Skymap {'enabled' if self.show_skymap else 'disabled'}")
    
    def toggle_axes(self):
        # Toggle coordinate axes
        self.show_axes = self.axes_var.get()
        if self.vis:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0])
            
            if self.show_axes:
                self.vis.add_geometry(mesh_frame, reset_bounding_box=False)
            else:
                self.vis.remove_geometry(mesh_frame, reset_bounding_box=False)
                
            self.status_bar.config(text=f"Coordinate axes {'shown' if self.show_axes else 'hidden'}")
    
    def choose_bg_color(self):
        color = colorchooser.askcolor(title="Choose Background Color")
        if color[1]: 
            # Update color entries
            r, g, b = [int(c) for c in color[0]]
            self.bg_r_entry.delete(0, tk.END)
            self.bg_r_entry.insert(0, f"R:{r}")
            self.bg_g_entry.delete(0, tk.END)
            self.bg_g_entry.insert(0, f"G:{g}")
            self.bg_b_entry.delete(0, tk.END)
            self.bg_b_entry.insert(0, f"B:{b}")
            
            # Update button color
            self.bg_color_button.configure(background=color[1])
            
            # Apply color
            self.apply_bg_color()
    
    def apply_bg_color(self):
        try:
            # Parse color values from entries
            r = int(self.bg_r_entry.get().split(":")[1]) / 255.0
            g = int(self.bg_g_entry.get().split(":")[1]) / 255.0
            b = int(self.bg_b_entry.get().split(":")[1]) / 255.0
            
            # Update background color
            self.bg_color = [r, g, b]
            if self.opt:
                self.opt.background_color = np.array(self.bg_color)
                self.status_bar.config(text=f"Background color updated to RGB({int(r*255)},{int(g*255)},{int(b*255)})")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid color format: {str(e)}")
    
    def choose_point_color(self):
        color = colorchooser.askcolor(title="Choose Point Color")
        if color[1]:  
            # Update color entries
            r, g, b = [int(c) for c in color[0]]
            self.color_r_entry.delete(0, tk.END)
            self.color_r_entry.insert(0, f"R:{r}")
            self.color_g_entry.delete(0, tk.END)
            self.color_g_entry.insert(0, f"G:{g}")
            self.color_b_entry.delete(0, tk.END)
            self.color_b_entry.insert(0, f"B:{b}")
            
            # Update button color
            self.color_button.configure(background=color[1])
            
            # Apply color
            self.apply_point_color()
    
    def apply_point_color(self):
        try:
            # Parse color values from entries
            r = int(self.color_r_entry.get().split(":")[1]) / 255.0
            g = int(self.color_g_entry.get().split(":")[1]) / 255.0
            b = int(self.color_b_entry.get().split(":")[1]) / 255.0
            
            # Update point color
            self.point_color = [r, g, b]
            
            if self.current_point_cloud:
                # Apply color to point cloud
                self.current_point_cloud.paint_uniform_color(self.point_color)
                # Update geometry
                self.vis.update_geometry(self.current_point_cloud)
                self.status_bar.config(text=f"Point color updated to RGB({int(r*255)},{int(g*255)},{int(b*255)})")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid color format: {str(e)}")
    
    def update_point_size(self, value):
        try:
            size = float(value)
            self.point_size = size
            if hasattr(self, 'point_value'):
                self.point_value.config(text=f"{int(size)}")
            else:
                print("Warning: point_value attribute not found")
            
            if self.opt:
                self.opt.point_size = size
                self.status_bar.config(text=f"Point size updated to {int(size)}")
        except Exception as e:
            print(f"Error updating point size: {e}")
    
    def toggle_advanced_lighting(self):
        # Toggle advanced lighting options
        if self.advanced_lighting_var.get():
            self.advanced_lighting_var.set(False)
            self.advanced_lighting_button.config(text="▶ Advanced lighting")
            self.advanced_lighting_frame.pack_forget()
        else:
            self.advanced_lighting_var.set(True)
            self.advanced_lighting_button.config(text="▼ Advanced lighting")
            self.advanced_lighting_frame.pack(fill=tk.X, padx=5, pady=2)
    
    def change_lighting(self, event):
        # Change lighting profile based on selection
        profile = self.lighting_combo.get()
        
        if profile == "Bright day with sun at +Y [default]":
            # Default bright lighting
            pass
        elif profile == "Cloudy day":
            # Softer, more diffuse lighting
            pass
        elif profile == "Night":
            # Dark lighting
            pass
        elif profile == "Custom":
            # Custom lighting settings
            pass
            
        self.status_bar.config(text=f"Lighting profile changed to: {profile}")
    
    def change_material_type(self, event):
        # Change material type based on selection
        material_type = self.type_combo.get()
        self.status_bar.config(text=f"Material type changed to: {material_type}")
    
    def change_material(self, event):
        # Change material based on selection
        material = self.material_combo.get()
        self.status_bar.config(text=f"Material changed to: {material}")
    
    def open_general_settings(self):
        # Open general settings dialog
        settings_dialog = tk.Toplevel(self.root)
        settings_dialog.title("General Settings")
        settings_dialog.geometry("400x300")
        settings_dialog.transient(self.root)
        settings_dialog.grab_set()
        
        # Create notebook for settings tabs
        notebook = ttk.Notebook(settings_dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Display tab
        display_frame = ttk.Frame(notebook)
        notebook.add(display_frame, text="Display")
        
        # Display settings
        ttk.Label(display_frame, text="Default point size:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        default_point_size = ttk.Scale(display_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        default_point_size.set(3)
        default_point_size.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(display_frame, text="Default background color:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        bg_color_button = ttk.Button(display_frame, text="Choose Color")
        bg_color_button.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(display_frame, text="Show axes on startup:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        show_axes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(display_frame, variable=show_axes_var).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Performance tab
        performance_frame = ttk.Frame(notebook)
        notebook.add(performance_frame, text="Performance")
        
        ttk.Label(performance_frame, text="Rendering quality:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        quality_combo = ttk.Combobox(performance_frame, values=["Low", "Medium", "High", "Ultra"])
        quality_combo.current(1)
        quality_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(performance_frame, text="Max points to render:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        max_points_entry = ttk.Entry(performance_frame)
        max_points_entry.insert(0, "1000000")
        max_points_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Controls tab
        controls_frame = ttk.Frame(notebook)
        notebook.add(controls_frame, text="Controls")
        
        ttk.Label(controls_frame, text="Mouse sensitivity:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        sensitivity_scale = ttk.Scale(controls_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        sensitivity_scale.set(5)
        sensitivity_scale.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(controls_frame, text="Invert mouse Y:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        invert_y_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_frame, variable=invert_y_var).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Button frame
        button_frame = ttk.Frame(settings_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=settings_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="OK", command=settings_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
    def open_ai_settings(self):
        # Open AI settings dialog
        ai_dialog = tk.Toplevel(self.root)
        ai_dialog.title("AI Settings")
        ai_dialog.geometry("400x300")
        ai_dialog.transient(self.root)
        ai_dialog.grab_set()
        
        # Create notebook for AI settings tabs
        notebook = ttk.Notebook(ai_dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Segmentation tab
        segmentation_frame = ttk.Frame(notebook)
        notebook.add(segmentation_frame, text="Segmentation")
        
        ttk.Label(segmentation_frame, text="Segmentation method:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        method_combo = ttk.Combobox(segmentation_frame, values=["RANSAC", "Region Growing", "Cluster Extraction"])
        method_combo.current(0)
        method_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(segmentation_frame, text="Distance threshold:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        distance_entry = ttk.Entry(segmentation_frame)
        distance_entry.insert(0, "0.05")
        distance_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Registration tab
        registration_frame = ttk.Frame(notebook)
        notebook.add(registration_frame, text="Registration")
        
        ttk.Label(registration_frame, text="Registration method:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        reg_method_combo = ttk.Combobox(registration_frame, values=["ICP", "Feature-based", "Global"])
        reg_method_combo.current(0)
        reg_method_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(registration_frame, text="Max iterations:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        iterations_entry = ttk.Entry(registration_frame)
        iterations_entry.insert(0, "100")
        iterations_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Processing tab
        processing_frame = ttk.Frame(notebook)
        notebook.add(processing_frame, text="Processing")
        
        ttk.Label(processing_frame, text="Voxel size:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        voxel_entry = ttk.Entry(processing_frame)
        voxel_entry.insert(0, "0.01")
        voxel_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(processing_frame, text="Statistical outlier removal:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        outlier_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(processing_frame, variable=outlier_var).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Button frame
        button_frame = ttk.Frame(ai_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=ai_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=ai_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="OK", command=ai_dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def show_documentation(self):
        # Show documentation dialog
        doc_dialog = tk.Toplevel(self.root)
        doc_dialog.title("Documentation")
        doc_dialog.geometry("600x400")
        doc_dialog.transient(self.root)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(doc_dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=text_widget.yview)
        
        # Insert documentation text
        documentation = """
        # Point Cloud Viewer Documentation
        
        ## Overview
        This application allows you to view and manipulate 3D point cloud data using Open3D.
        
        ## Features
        - Load point cloud data from various file formats (.pcd, .ply, .xyz, .pts)
        - Visualize point clouds with adjustable colors and point sizes
        - Control the camera with different navigation modes
        - Adjust lighting and material properties
        - Export visualizations as images
        
        ## Controls
        - **Mouse Controls**:
            - Left Click + Drag: Rotate
            - Right Click + Drag: Pan
            - Scroll Wheel: Zoom
        
        ## File Formats
        The application supports the following file formats:
        - .pcd: Point Cloud Data format
        - .ply: Polygon File Format
        - .xyz: ASCII format with X, Y, Z coordinates
        - .pts: Points format
        
        ## Settings
        You can adjust various settings to customize the visualization:
        - View controls: Camera navigation, lighting, background color
        - Material settings: Point color, size, and material properties
        
        ## AI Features
        The application includes AI-powered features for point cloud processing:
        - Segmentation: Identify and separate different objects in the point cloud
        - Registration: Align multiple point clouds
        - Processing: Clean and downsample point clouds
        """
        
        text_widget.insert(tk.END, documentation)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Close button
        ttk.Button(doc_dialog, text="Close", command=doc_dialog.destroy).pack(pady=10)
    
    def show_about(self):
        # Show about dialog
        about_dialog = tk.Toplevel(self.root)
        about_dialog.title("About")
        about_dialog.geometry("400x300")
        about_dialog.transient(self.root)
        about_dialog.resizable(False, False)
        
        # App title
        ttk.Label(about_dialog, text="Point Cloud Viewer", font=("Arial", 16, "bold")).pack(pady=(20, 10))
        
        # Version info
        ttk.Label(about_dialog, text="Version 1.0.0").pack()
        
        # Description
        description = """
        A powerful point cloud visualization tool built with Python and Open3D.
        Developed for Proto Corp as a UI demonstration project.
        """
        ttk.Label(about_dialog, text=description, wraplength=350, justify=tk.CENTER).pack(pady=10)
        
        # Libraries used
        ttk.Label(about_dialog, text="Libraries Used:", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        ttk.Label(about_dialog, text="Open3D, Tkinter, NumPy, PIL").pack()
        
        # Copyright
        ttk.Label(about_dialog, text="© 2025 Proto Corp. All rights reserved.").pack(pady=(20, 10))
        
        # Close button
        ttk.Button(about_dialog, text="Close", command=about_dialog.destroy).pack(pady=10)
    
    def quit_application(self):
        # Clean up and exit
        self.running = False
        
        # Wait for visualization thread to finish 
        if hasattr(self, 'vis_thread') and self.vis_thread.is_alive():
            self.vis_thread.join(1.0)  # Wait up to 1 second
        
        # Destroy the visualizer
        if hasattr(self, 'vis') and self.vis:
            try:
                self.vis.destroy_window()
            except Exception as e:
                print(f"Error destroying visualizer: {e}")
        
        self.root.quit()
        self.root.destroy()
        
        # Force exit if needed
        sys.exit(0)

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = PointCloudViewer(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_application)
    root.mainloop()