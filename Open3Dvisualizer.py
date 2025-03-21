import open3d as o3d
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
import numpy as np
import os
import sys
from PIL import Image, ImageTk
import multiprocessing
import tempfile
import time
import sys
import queue


def visualization_worker(render_queue, result_queue):
    """Worker function for visualization process"""
    vis = None
    cloud = None
    running = True
    temp_dir = tempfile.mkdtemp()
    
    # list to store selected points
    selected_points = []
    view_control = None
    
    try:
        # Initialize visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=600)
        
        # Set default rendering options
        opt = vis.get_render_option()
        opt.point_size = 3
        opt.background_color = np.array([1.0, 1.0, 1.0])
        
        # Get view control
        view_control = vis.get_view_control()
        
        # Main loop
        while running:
            # Process commands from render queue
            try:
                if not render_queue.empty():
                    command = render_queue.get(block=False)
                    
                    # Handle load_file command
                    if command['command'] == 'load_file':
                        file_path = command['file_path']
                        file_ext = command['file_ext']
                        
                        result_queue.put({
                            'type': 'status',
                            'message': f"Loading file: {os.path.basename(file_path)}"
                        })
                        
                        try:
                            # Load the file based on its extension
                            if file_ext.lower() in ['.ply', '.pcd', '.xyz', '.pts']:
                                # Load as point cloud
                                cloud = o3d.io.read_point_cloud(file_path)
                                
                                # If point cloud is empty, try to load as mesh and sample points
                                if len(cloud.points) == 0 and file_ext.lower() in ['.ply', '.obj']:
                                    mesh = o3d.io.read_triangle_mesh(file_path)
                                    cloud = mesh.sample_points_uniformly(number_of_points=100000)
                            elif file_ext.lower() == '.obj':
                                # Load as mesh then convert to point cloud
                                mesh = o3d.io.read_triangle_mesh(file_path)
                                cloud = mesh.sample_points_uniformly(number_of_points=100000)
                            else:
                                result_queue.put({
                                    'type': 'error',
                                    'message': f"Unsupported file format: {file_ext}"
                                })
                                continue
                            
                            # Check if the point cloud is valid
                            if cloud is None or len(cloud.points) == 0:
                                result_queue.put({
                                    'type': 'error',
                                    'message': "Failed to load point cloud or mesh"
                                })
                                continue
                            
                            # Clear existing geometries and add new point cloud
                            vis.clear_geometries()
                            vis.add_geometry(cloud)
                            
                            # Reset view to look at the point cloud
                            vis.reset_view_point(True)
                            
                            # Update and render
                            vis.poll_events()
                            vis.update_renderer()
                            
                            # Capture screenshot
                            img_path = os.path.join(temp_dir, 'render.png')
                            vis.capture_screen_image(img_path, do_render=True)
                            
                            result_queue.put({
                                'type': 'image',
                                'image_path': img_path
                            })
                            
                            result_queue.put({
                                'type': 'status',
                                'message': f"Loaded {os.path.basename(file_path)} with {len(cloud.points)} points"
                            })
                            
                        except Exception as e:
                            result_queue.put({
                                'type': 'error',
                                'message': f"Error loading file: {str(e)}"
                            })
                    
                    
                    elif command['command'] == 'pick_point':
                        if cloud is not None:
                            viewport_x = command['viewport_x']
                            viewport_y = command['viewport_y']
                            
                            # Get the view parameters
                            params = view_control.convert_to_pinhole_camera_parameters()
                            intrinsic = params.intrinsic.intrinsic_matrix
                            extrinsic = params.extrinsic
                            width = 800  # Visualizer window width
                            height = 600  # Visualizer window height
                            
                            # Convert viewport coordinates to pixel coordinates
                            x = int(viewport_x * width)
                            y = int(viewport_y * height)
                            
                            # Get ray direction in camera space
                            fx = intrinsic[0, 0]
                            fy = intrinsic[1, 1]
                            cx = intrinsic[0, 2]
                            cy = intrinsic[1, 2]
                            
                            x_cam = (x - cx) / fx
                            y_cam = (y - cy) / fy
                            z_cam = 1.0  # Forward direction in camera space
                            
                            ray_camera = np.array([x_cam, y_cam, z_cam])
                            ray_camera = ray_camera / np.linalg.norm(ray_camera)
                            
                            # Convert ray to world space
                            camera_pose = np.linalg.inv(extrinsic)
                            rotation = camera_pose[:3, :3]
                            camera_pos = camera_pose[:3, 3]
                            
                            ray_world = rotation @ ray_camera
                            
                            # Find closest point to ray
                            points = np.asarray(cloud.points)
                            
                            # Compute distance from each point to ray
                            v = points - camera_pos.reshape(1, 3)
                            dist = np.cross(v, ray_world.reshape(1, 3))
                            dist = np.linalg.norm(dist, axis=1)
                            
                            # Find closest point
                            closest_idx = np.argmin(dist)
                            closest_point = points[closest_idx]
                            
                            # Add to selected points
                            if len(selected_points) >= 2:
                                selected_points.clear()
                            
                            selected_points.append(closest_point)
                            
                            # Create a marker for the picked point
                            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                            marker.translate(closest_point)
                            marker.paint_uniform_color([1, 0, 0])  # Red marker
                            
                            # Clear previous markers and re-add point cloud
                            vis.clear_geometries()
                            vis.add_geometry(cloud)
                            
                            # Add all selected point markers
                            for point in selected_points:
                                m = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                                m.translate(point)
                                m.paint_uniform_color([1, 0, 0])
                                vis.add_geometry(m)
                            
                            # If we have two points, calculate distance
                            if len(selected_points) == 2:
                                p1 = selected_points[0]
                                p2 = selected_points[1]
                                distance = np.linalg.norm(p1 - p2)
                                
                                # Draw line between points
                                line_points = np.vstack([p1, p2])
                                line_indices = np.array([[0, 1]])
                                line_set = o3d.geometry.LineSet()
                                line_set.points = o3d.utility.Vector3dVector(line_points)
                                line_set.lines = o3d.utility.Vector2iVector(line_indices)
                                line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green line
                                vis.add_geometry(line_set)
                                
                                # Update UI with distance
                                result_queue.put({
                                    'type': 'distance',
                                    'distance': distance,
                                    'points': selected_points.copy()
                                })
                            
                            vis.update_renderer()
                            
                            # Capture and send rendered image
                            img_path = os.path.join(temp_dir, 'render.png')
                            vis.capture_screen_image(img_path, do_render=True)
                            
                            result_queue.put({
                                'type': 'image',
                                'image_path': img_path
                            })
                            
                            # Send the points back to the main process
                            result_queue.put({
                                'type': 'selected_point',
                                'point': closest_point.tolist(),
                                'total_points': len(selected_points)
                            })
                    
                    elif command['command'] == 'clear_markers':
                        # Clear all markers by reloading the point cloud
                        if cloud is not None:
                            selected_points.clear()
                            vis.clear_geometries()
                            vis.add_geometry(cloud)
                            vis.update_renderer()
                            
                            # Capture and send rendered image
                            img_path = os.path.join(temp_dir, 'render.png')
                            vis.capture_screen_image(img_path, do_render=True)
                            
                            result_queue.put({
                                'type': 'image',
                                'image_path': img_path
                            })
                            
                            result_queue.put({
                                'type': 'status',
                                'message': "Point markers cleared"
                            })
                    
                    # Add other commands for background color, point size, etc.
                    elif command['command'] == 'set_bg_color':
                        color = command['color']
                        opt = vis.get_render_option()
                        opt.background_color = np.array(color)
                        vis.update_renderer()
                        
                        # Capture and send rendered image
                        if cloud is not None:
                            img_path = os.path.join(temp_dir, 'render.png')
                            vis.capture_screen_image(img_path, do_render=True)
                            
                            result_queue.put({
                                'type': 'image',
                                'image_path': img_path
                            })
                    
                    elif command['command'] == 'set_point_size':
                        size = command['size']
                        opt = vis.get_render_option()
                        opt.point_size = size
                        vis.update_renderer()
                        
                        # Capture and send rendered image
                        if cloud is not None:
                            img_path = os.path.join(temp_dir, 'render.png')
                            vis.capture_screen_image(img_path, do_render=True)
                            
                            result_queue.put({
                                'type': 'image',
                                'image_path': img_path
                            })
                    
                    elif command['command'] == 'set_point_color':
                        if cloud is not None:
                            color = command['color']
                            cloud.paint_uniform_color(color)
                            vis.update_geometry(cloud)
                            vis.update_renderer()
                            
                            # Capture and send rendered image
                            img_path = os.path.join(temp_dir, 'render.png')
                            vis.capture_screen_image(img_path, do_render=True)
                            
                            result_queue.put({
                                'type': 'image',
                                'image_path': img_path
                            })
                    
                    elif command['command'] == 'set_view_mode':
                        mode = command['mode']
                        if mode == 'arcball':
                            view_control.set_zoom(0.7)
                            view_control.set_front([-0.0, -0.0, -1.0])
                            view_control.set_lookat([0.0, 0.0, 0.0])
                            view_control.set_up([0.0, 1.0, 0.0])
                        elif mode == 'fly':
                            # Custom fly mode settings
                            pass
                        elif mode == 'model':
                            view_control.set_zoom(0.7)
                            view_control.set_front([0.0, 0.0, 1.0])
                            view_control.set_lookat([0.0, 0.0, 0.0])
                            view_control.set_up([0.0, 1.0, 0.0])
                        
                        vis.update_renderer()
                        
                        # Capture and send rendered image
                        if cloud is not None:
                            img_path = os.path.join(temp_dir, 'render.png')
                            vis.capture_screen_image(img_path, do_render=True)
                            
                            result_queue.put({
                                'type': 'image',
                                'image_path': img_path
                            })
                    
                    elif command['command'] == 'set_lighting':
                        profile = command['profile']
                        opt = vis.get_render_option()
                        
                        if profile == "Bright day with sun at +Y [default]":
                            opt.light_on = True
                            # Default lighting settings
                        elif profile == "Cloudy day":
                            opt.light_on = True
                            # Cloudy day lighting settings
                        elif profile == "Night":
                            opt.light_on = True
                            # Night lighting settings
                        elif profile == "Custom":
                            # Custom lighting settings
                            pass
                        
                        vis.update_renderer()
                        
                        # Capture and send rendered image
                        if cloud is not None:
                            img_path = os.path.join(temp_dir, 'render.png')
                            vis.capture_screen_image(img_path, do_render=True)
                            
                            result_queue.put({
                                'type': 'image',
                                'image_path': img_path
                            })
                    
                    # commands for rotation and zoom
                    elif command['command'] == 'rotate':
                        if cloud is not None and view_control is not None:
                            dx = command['dx']
                            dy = command['dy']
                            
                            
                            # Get current view parameters
                            params = view_control.convert_to_pinhole_camera_parameters()
                            
                            # Rotate view using view_control methods
                            view_control.rotate(dx, dy)
                            
                            vis.update_renderer()
                            
                            # Capture and send rendered image
                            img_path = os.path.join(temp_dir, 'render.png')
                            vis.capture_screen_image(img_path, do_render=True)
                            
                            result_queue.put({
                                'type': 'image',
                                'image_path': img_path
                            })
                    
                    elif command['command'] == 'zoom':
                        if cloud is not None and view_control is not None:
                            zoom_factor = command['factor']
                            
                            # Apply zoom based on direction
                            view_control.scale(zoom_factor)
                            
                            # Update visualization
                            vis.poll_events()
                            vis.update_renderer()
                            
                            # Capture and send rendered image
                            img_path = os.path.join(temp_dir, 'render.png')
                            vis.capture_screen_image(img_path, do_render=True)
                            
                            result_queue.put({
                                'type': 'image',
                                'image_path': img_path
                            })
                
            except queue.Empty:
                pass
            except Exception as e:
                result_queue.put({
                    'type': 'error',
                    'message': f"Visualization process error: {str(e)}"
                })
            
            # Render and update at least once per second
            time.sleep(0.1)
            
            # Update visualization if cloud is loaded
            if cloud is not None:
                vis.poll_events()
                vis.update_renderer()
                
                # Capture and send rendered image occasionally
                if np.random.random() < 0.05:  # ~5% chance each iteration
                    img_path = os.path.join(temp_dir, 'render.png')
                    vis.capture_screen_image(img_path, do_render=True)
                    
                    result_queue.put({
                        'type': 'image',
                        'image_path': img_path
                    })
    
    except Exception as e:
        result_queue.put({
            'type': 'error',
            'message': f"Visualization process error: {str(e)}"
        })
    
    finally:
        # Clean up
        if vis is not None:
            vis.destroy_window()
        # Clean up temp directory
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

class PointCloudViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Viewer")
        self.root.geometry("1200x800")

        # Replace threading lock with multiprocessing
        self.gl_lock = multiprocessing.RLock()
        
        # Communication queues for multiprocessing
        self.render_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        
        #point cloud variables
        self.current_point_cloud = None
        self.vis = None
        self.running = False
        
        self.bg_color = [1.0, 1.0, 1.0]  
        self.point_color = [0.9, 0.9, 0.9]  
        self.point_size = 3
        self.show_axes = False
        self.show_skymap = False
        
        # Variables for rotation and zoom
        self.is_rotating = False
        self.last_x = 0
        self.last_y = 0
        self.zoom_scale = 1.0
        
        self.create_menu_bar()
        
        # main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create visualization area
        self.viz_frame = ttk.Frame(self.main_frame, borderwidth=2, relief="sunken")
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the canvas for Open3D rendering
        self.canvas = tk.Canvas(self.viz_frame, bg="white")
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
        
        # Initialize Open3D visualizer in a separate process
        self.init_open3d()

        self.selected_points = []
        self.point_picking_mode = False
        
        # Bind mouse events for rotation and zoom
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B2-Motion>", self.on_rotate_drag)  # Middle mouse button for rotation
        self.canvas.bind("<Button-2>", self.on_rotate_start)
        self.canvas.bind("<ButtonRelease-2>", self.on_rotate_stop)
        
        # Bind right mouse button for rotation as well
        self.canvas.bind("<Button-3>", self.on_rotate_start)
        self.canvas.bind("<B3-Motion>", self.on_rotate_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_rotate_stop)
        
        # Bind mouse wheel for zoom
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        
        # Set up periodic UI update from result queue
        self.root.after(100, self.check_result_queue)
        
    def on_rotate_start(self, event):
        """Start rotation when middle mouse button is pressed"""
        if not self.point_picking_mode:
            self.is_rotating = True
            self.last_x = event.x
            self.last_y = event.y
            self.status_bar.config(text="Rotating view (hold and drag)")
            
    def on_rotate_stop(self, event):
        """Stop rotation when middle mouse button is released"""
        self.is_rotating = False
        self.status_bar.config(text="Ready")
        
    def on_rotate_drag(self, event):
        """Handle rotation by sending the delta to visualization process"""
        if self.is_rotating:
            # Calculate the motion difference
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            
            # Update last position
            self.last_x = event.x
            self.last_y = event.y
            
            # Send rotation command to visualization process
            self.render_queue.put({
                'command': 'rotate',
                'dx': dx,
                'dy': dy
            })
    
    def on_mouse_wheel(self, event):
        """Handle zoom with mouse wheel on Windows"""
        if not self.point_picking_mode:
            # Get zoom direction from event.delta
            zoom_factor = 1.0
            direction = 'none'
            if event.delta > 0:
                zoom_factor = 1.1  # Zoom in
                direction = 'in'
            else:
                zoom_factor = 0.9  # Zoom out
                direction = 'out'
                
            # Send zoom command to visualization process
            self.render_queue.put({
                'command': 'zoom',
                'factor': zoom_factor,
                'direction': direction
            })
    
        
    def create_menu_bar(self):
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open...", command=self.open_file)
        # file_menu.add_command(label="Open Samples...", command=self.open_samples)
        # file_menu.add_command(label="Export Current Image...", command=self.export_image)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit_application)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Settings menu
        settings_menu = tk.Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="General Settings", command=self.open_general_settings)
        # settings_menu.add_command(label="AI Settings", command=self.open_ai_settings)
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
        
        # Add rotation and zoom help text
        rotation_frame = ttk.Frame(view_frame)
        rotation_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(rotation_frame, text="Rotate: Middle/Right mouse button", font=("Arial", 8)).pack(anchor=tk.W)
        ttk.Label(rotation_frame, text="Zoom: Mouse wheel", font=("Arial", 8)).pack(anchor=tk.W)
        
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

        # Point picking section
        point_picking_frame = ttk.Frame(view_frame)
        point_picking_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(point_picking_frame, text="Distance Measurement").pack(anchor=tk.W)
        ttk.Button(point_picking_frame, text="Toggle Point Selection", command=self.toggle_point_picking_mode).pack(anchor=tk.W, pady=2)
        ttk.Button(point_picking_frame, text="Clear Selected Points", command=self.clear_point_markers).pack(anchor=tk.W, pady=2)
        
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
        # Create the point_value label first
        self.point_value = ttk.Label(point_frame, text="3")
        self.point_value.pack(side=tk.LEFT, padx=5)
        
        # Then create and set the scale
        self.point_scale = ttk.Scale(point_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=self.update_point_size)
        self.point_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.point_scale.set(3)
        
        print("Created point_value label")

    def init_open3d(self):
        try:
            # Start a new process for handling Open3D rendering
            self.visualization_process = multiprocessing.Process(
                target=visualization_worker,
                args=(self.render_queue, self.result_queue)
            )
            self.visualization_process.daemon = True
            self.visualization_process.start()
            
            # Send initialization command
            self.render_queue.put({
                'command': 'init',
                'bg_color': self.bg_color,
                'point_size': self.point_size,
                'show_axes': self.show_axes
            })
            
            self.running = True
            self.status_bar.config(text="Open3D initialized in separate process")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize Open3D: {str(e)}")
            print(f"Open3D initialization error: {e}")
            
    def check_result_queue(self):
        """Process any results from the visualization process"""
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get(block=False)
                
                if result['type'] == 'image':
                    # Update the canvas with the new render
                    img_path = result['image_path']
                    if os.path.exists(img_path):
                        try:
                            # robust image loading method
                            from PIL import ImageFile
                            ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow truncated images
                            
                            # Add a short delay to ensure file is fully written
                            time.sleep(0.05)
                            
                            img = Image.open(img_path)
                            if self.canvas.winfo_width() > 1 and self.canvas.winfo_height() > 1:
                                img = img.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.LANCZOS)
                            else:
                                img = img.resize((800, 600), Image.LANCZOS)
                            self.photo = ImageTk.PhotoImage(img)
                            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                        except Exception as e:
                            print(f"Error displaying image: {e}")
                            # Create a default image instead
                            self.create_default_preview()
                    else:
                        print(f"Image file not found: {img_path}")
                        self.create_default_preview()
                
                elif result['type'] == 'selected_point':
                    # Handle selected point
                    point = result['point']
                    self.selected_points.append(point)
                    total_points = result['total_points']
                    self.status_bar.config(text=f"Selected point {total_points}/2: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
                
                elif result['type'] == 'distance':
                    # Handle distance calculation
                    distance = result['distance']
                    points = result['points']
                    self.status_bar.config(text=f"Distance: {distance:.4f} units")
                    
                    # Show distance in a dialog
                    self.show_distance_dialog(distance, points)
                
                elif result['type'] == 'status':
                    # Update status message
                    self.status_bar.config(text=result['message'])
                    
                elif result['type'] == 'error':
                    # Show error message
                    messagebox.showerror("Error", result['message'])
        except Exception as e:
            print(f"Error checking result queue: {e}")
        
        # Schedule the next check
        self.root.after(100, self.check_result_queue)

    def create_default_preview(self):
        """Create a default preview image when rendering fails"""
        width = self.canvas.winfo_width() or 800
        height = self.canvas.winfo_height() or 600
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw a simple placeholder
        self.canvas.create_rectangle(0, 0, width, height, fill="white")
        self.canvas.create_text(width/2, height/2, 
                            text="Point cloud visualization\n(Preview not available)",
                            font=("Arial", 14),
                            fill="gray",
                            justify=tk.CENTER)
    
    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("3D Model Files", "*.obj *.ply *.pcd *.xyz *.pts"), 
                    ("Mesh Files", "*.obj *.ply"),
                    ("Point Cloud Files", "*.pcd *.ply *.xyz *.pts"), 
                    ("All Files", "*.*")]
        )
        if file_path:
            file_ext = os.path.splitext(file_path)[1].lower()
            # Send command to visualization process to load file
            self.render_queue.put({
                'command': 'load_file',
                'file_path': file_path,
                'file_ext': file_ext
            })
            self.status_bar.config(text=f"Loading {os.path.basename(file_path)}...")

    def toggle_point_picking_mode(self):
        self.point_picking_mode = not self.point_picking_mode
        if self.point_picking_mode:
            self.selected_points = []
            self.clear_point_markers()
            self.status_bar.config(text="Point picking mode: ON. Click to select points (max 2)")
        else:
            self.status_bar.config(text="Point picking mode: OFF")

    def clear_point_markers(self):
        self.render_queue.put({
            'command': 'clear_markers'
        })
        self.point_markers = []
        self.selected_points = []
        
    def on_canvas_click(self, event):
        if not self.point_picking_mode:
            return
            
        if len(self.selected_points) >= 2:
            # If we already have 2 points, reset selection
            self.selected_points = []
            self.clear_point_markers()
            
        # Convert canvas coordinates to viewport coordinates
        viewport_x = event.x / self.canvas.winfo_width()
        viewport_y = event.y / self.canvas.winfo_height()
        
        # Send point picking command to visualization process
        self.render_queue.put({
            'command': 'pick_point',
            'viewport_x': viewport_x,
            'viewport_y': viewport_y
        })

    def show_distance_dialog(self, distance, points):
        # Create a dialog to show the distance
        distance_dialog = tk.Toplevel(self.root)
        distance_dialog.title("Distance Measurement")
        distance_dialog.geometry("300x150")
        distance_dialog.transient(self.root)
        
        # Distance information
        ttk.Label(distance_dialog, text="Distance between selected points:", font=("Arial", 12)).pack(pady=(20, 5))
        ttk.Label(distance_dialog, text=f"{distance:.4f} units", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Point information
        if len(points) >= 2:
            point1 = points[0]
            point2 = points[1]
            
            info_text = f"Point 1: ({point1[0]:.3f}, {point1[1]:.3f}, {point1[2]:.3f})\n"
            info_text += f"Point 2: ({point2[0]:.3f}, {point2[1]:.3f}, {point2[2]:.3f})"
            
            ttk.Label(distance_dialog, text=info_text, justify=tk.LEFT).pack(pady=10)
        
        # Close button
        ttk.Button(distance_dialog, text="Close", command=distance_dialog.destroy).pack(pady=10)
    
    

    def update_point_size(self, value):
        try:
            size = int(float(value))
            self.point_size = size
            self.point_value.config(text=str(size))
            
            # Send command to update point size in visualization
            self.render_queue.put({
                'command': 'set_point_size',
                'size': size
            })
        except Exception as e:
            print(f"Error updating point size: {e}")

    
    
    
    def toggle_advanced_lighting(self):
        if self.advanced_lighting_var.get():
            self.advanced_lighting_var.set(False)
            self.advanced_lighting_frame.pack_forget()
            self.advanced_lighting_button.config(text="▶ Advanced lighting")
        else:
            self.advanced_lighting_var.set(True)
            self.advanced_lighting_frame.pack(fill=tk.X, padx=5, pady=2)
            self.advanced_lighting_button.config(text="▼ Advanced lighting")
            
    def change_material_type(self, event):
        material_type = self.type_combo.get()
        self.render_queue.put({
            'command': 'set_material_type',
            'type': material_type
        })
    
    def change_material(self, event):
        material = self.material_combo.get()
        self.render_queue.put({
            'command': 'set_material',
            'material': material
        })
    
    def choose_point_color(self):
        color = colorchooser.askcolor(title="Choose Point Color")
        if color[1]:  # If not canceled
            r, g, b = [int(c) for c in color[0]]
            self.color_r_entry.delete(0, tk.END)
            self.color_r_entry.insert(0, f"R:{r}")
            self.color_g_entry.delete(0, tk.END)
            self.color_g_entry.insert(0, f"G:{g}")
            self.color_b_entry.delete(0, tk.END)
            self.color_b_entry.insert(0, f"B:{b}")
    
    def apply_point_color(self):
        try:
            r = int(self.color_r_entry.get().split(':')[1])
            g = int(self.color_g_entry.get().split(':')[1])
            b = int(self.color_b_entry.get().split(':')[1])
            self.point_color = [r/255, g/255, b/255]
            self.render_queue.put({
                'command': 'set_point_color',
                'color': self.point_color
            })
        except Exception as e:
            print(f"Error applying point color: {e}")
    
    def set_arcball_mode(self):
        self.render_queue.put({
            'command': 'set_view_mode',
            'mode': 'arcball'
        })
        
    def set_fly_mode(self):
        self.render_queue.put({
            'command': 'set_view_mode',
            'mode': 'fly'
        })
        # self.status_bar.config(text="View mode: Fly")
        
    def set_model_mode(self):
        self.render_queue.put({
            'command': 'set_view_mode',
            'mode': 'model'
        })
        # self.status_bar.config(text="View mode: Model")
    
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
            r = int(self.bg_r_entry.get().split(':')[1]) / 255.0
            g = int(self.bg_g_entry.get().split(':')[1]) / 255.0
            b = int(self.bg_b_entry.get().split(':')[1]) / 255.0
            
            self.bg_color = [r, g, b]
            
            # Send to visualization process
            self.render_queue.put({
                'command': 'set_bg_color',
                'color': self.bg_color
            })
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid color format: {str(e)}")
    
    def change_lighting(self, event):
        # Change lighting profile based on selection
        profile = self.lighting_combo.get()
        
        # Send to visualization process
        self.render_queue.put({
            'command': 'set_lighting',
            'profile': profile
        })
            
        # self.status_bar.config(text=f"Lighting profile changed to: {profile}")

    
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
        
    def show_documentation(self):
        messagebox.showinfo("Documentation", "Documentation is available at: https://pointcloudviewer.docs.example.com")
    
    def show_about(self):
        messagebox.showinfo("About Point Cloud Viewer", "Point Cloud Viewer\nVersion 1.0\n\nA tool for visualizing and analyzing 3D point clouds.")

    
    
    def quit_application(self):
        # Send quit command to visualization process
        if self.running and self.visualization_process.is_alive():
            self.render_queue.put({'command': 'quit'})
            # Give time for the process to terminate 
            time.sleep(0.5)
        self.running = False
        self.root.destroy()
        sys.exit()


if __name__ == "__main__":
    # here we check if Open3D is installed
    try:
        import open3d as o3d
    except ImportError:
        print("Error: Open3D is not installed. Please install it using:")
        print("pip install open3d")
        sys.exit(1)
    
    # Start the application
    root = tk.Tk()
    app = PointCloudViewer(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_application)
    root.mainloop()