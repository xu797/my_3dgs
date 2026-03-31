#!/usr/bin/env python3
"""
Embedded 3D Gaussian Splatting Viewer using Python rendering
"""

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time

# Add gaussian-splatting to path
sys.path.insert(0, "/home/eric/xujixian/work/code/gaussian-splatting")

try:
    import torch
    from scene import Scene
    from gaussian_renderer import render, GaussianModel  # 新增导入GaussianModel
    from argparse import Namespace
    from scene.cameras_back import Camera as GSCamera
    from utils.graphics_utils import focal2fov
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the gaussian-splatting conda environment")


class EmbeddedGaussianViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Gaussian Splatting Viewer")
        self.root.geometry("1200x800")
        
        self.model_path = "/home/eric/xujixian/work/code/gaussian-splatting/data/ggbond/output"
        self.scene = None
        self.gaussians = None  # 改为先初始化GaussianModel，再传入Scene
        self.rendering = False
        self.current_camera_idx = 0
        
        # Camera parameters for interactive viewing
        self.camera_distance = 4.0
        self.camera_azimuth = 0.0
        self.camera_elevation = 20.0
        
        self.create_widgets()
    
    def create_widgets(self):
        # Top control panel
        control_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        control_frame.pack_propagate(False)
        
        # Title
        title = tk.Label(control_frame, text="3D Gaussian Splatting Viewer", 
                        font=("Arial", 14, "bold"), bg="#2c3e50", fg="white")
        title.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Load button
        self.load_btn = tk.Button(control_frame, text="Load Model",
                                  command=self.load_model,
                                  font=("Arial", 11, "bold"),
                                  bg="#27ae60", fg="white",
                                  width=12, height=1,
                                  cursor="hand2")
        self.load_btn.pack(side=tk.LEFT, padx=10)
        
        # Render button
        self.render_btn = tk.Button(control_frame, text="Render",
                                    command=self.render_view,
                                    font=("Arial", 11, "bold"),
                                    bg="#3498db", fg="white",
                                    width=12, height=1,
                                    cursor="hand2",
                                    state=tk.DISABLED)
        self.render_btn.pack(side=tk.LEFT, padx=10)
        
        # Next camera button
        self.next_cam_btn = tk.Button(control_frame, text="Next Camera",
                                      command=self.next_camera,
                                      font=("Arial", 11, "bold"),
                                      bg="#9b59b6", fg="white",
                                      width=12, height=1,
                                      cursor="hand2",
                                      state=tk.DISABLED)
        self.next_cam_btn.pack(side=tk.LEFT, padx=10)
        
        # Status
        self.status = tk.Label(control_frame, text="Ready", 
                              font=("Arial", 10), bg="#2c3e50", fg="#ecf0f1")
        self.status.pack(side=tk.RIGHT, padx=20)
        
        # Canvas for rendering
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Info label
        self.info_label = tk.Label(self.canvas, 
                                   text="Click 'Load Model' to start",
                                   font=("Arial", 12),
                                   bg="black", fg="white")
        self.info_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"Model path not found:\n{self.model_path}")
            return
        
        self.status.config(text="Loading model...", fg="#f39c12")
        self.info_label.config(text="Loading Gaussian Splatting model...")
        self.root.update()
        
        try:
            # Load model in a thread
            thread = threading.Thread(target=self._load_model_thread)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{str(e)}")
            self.status.config(text="Load failed", fg="#e74c3c")
    
    def _load_model_thread(self):
        error_msg = None
        try:
            # Create args namespace
            args = Namespace(
                model_path=self.model_path,
                sh_degree=3,
                images="images",
                resolution=-1,
                white_background=False,
                data_device="cuda",
                eval=True
            )
            
            # 核心修复：先创建GaussianModel对象
            self.gaussians = GaussianModel(sh_degree=args.sh_degree)
            # 修复Scene初始化：传入gaussians参数
            self.scene = Scene(args, self.gaussians, load_iteration=None, shuffle=False)
            
            self.root.after(0, self._model_loaded)
            
        except Exception as ex:
            error_msg = str(ex)
            import traceback
            error_msg += "\n" + traceback.format_exc()
            self.root.after(0, lambda: self._load_error(error_msg))
    
    def _model_loaded(self):
        # 检查相机数量，避免空列表
        cam_count = len(self.scene.getTrainCameras()) if self.scene else 0
        self.status.config(text=f"Model loaded ({cam_count} cameras)", fg="#27ae60")
        self.info_label.config(text="Model loaded! Click 'Render' to visualize")
        self.render_btn.config(state=tk.NORMAL)
        self.next_cam_btn.config(state=tk.NORMAL if cam_count > 1 else tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
    
    def _load_error(self, error):
        messagebox.showerror("Error", f"Failed to load model:\n{error}")
        self.status.config(text="Load failed", fg="#e74c3c")
        self.info_label.config(text=f"Error loading model")
    
    def render_view(self):
        if self.scene is None or self.gaussians is None:
            return
        
        self.status.config(text="Rendering...", fg="#f39c12")
        self.root.update()
        
        try:
            # Get camera
            cameras = self.scene.getTrainCameras()
            if not cameras:
                messagebox.showwarning("Warning", "No cameras found in scene!")
                self.status.config(text="No cameras", fg="#e67e22")
                return
                
            if self.current_camera_idx >= len(cameras):
                self.current_camera_idx = 0
        
            viewpoint_cam = cameras[self.current_camera_idx]
        
            # Render (禁用梯度计算加速)
            with torch.no_grad():
                render_pkg = render(viewpoint_cam, self.gaussians, 
                                   background=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
                image = render_pkg["render"]
        
            # Convert to PIL Image (修正维度转换逻辑)
            image_np = image.cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))  # CHW -> HWC
            image_np = np.clip(image_np, 0, 1)  # 防止数值溢出
            image_np = (image_np * 255).astype(np.uint8)
        
            # Resize to fit canvas (处理canvas尺寸为0的情况)
            canvas_width = self.canvas.winfo_width() or 800
            canvas_height = self.canvas.winfo_height() or 600
        
            pil_image = Image.fromarray(image_np)
            # Maintain aspect ratio
            img_ratio = pil_image.width / pil_image.height
            canvas_ratio = canvas_width / canvas_height
        
            if img_ratio > canvas_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)
        
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
            # Display on canvas
            self.photo = ImageTk.PhotoImage(pil_image)
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
        
            self.info_label.place_forget()
            self.status.config(text=f"Rendered camera {self.current_camera_idx + 1}/{len(cameras)}", fg="#27ae60")
        
        except Exception as e:
            messagebox.showerror("Error", f"Rendering failed:\n{str(e)}")
            self.status.config(text="Render failed", fg="#e74c3c")
            import traceback
            traceback.print_exc()
    
    def next_camera(self):
        if self.scene is None:
            return
        
        cameras = self.scene.getTrainCameras()
        if not cameras:
            return
            
        self.current_camera_idx = (self.current_camera_idx + 1) % len(cameras)
        self.render_view()
    
    def on_closing(self):
        self.rendering = False
        # 安全释放GPU内存
        if self.scene is not None:
            del self.scene
        if self.gaussians is not None:
            del self.gaussians
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.root.destroy()


if __name__ == "__main__":
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print("CUDA is available - using GPU")
    else:
        print("Warning: CUDA not available - using CPU (slow!)")
    
    root = tk.Tk()
    app = EmbeddedGaussianViewer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()