#!/usr/bin/env python3
"""
GUI Launcher for dashboard_ros.py
Provides a user-friendly interface to configure and run the ROS2 Multi-Agent Dashboard.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import subprocess
import os
import sys
import json

# Presets file location (same directory as this script)
PRESETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_presets.json")

class DashboardLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Dashboard Launcher")
        self.root.geometry("600x700")
        self.root.resizable(True, True)
        
        # Create main scrollable frame
        self.canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Bind mousewheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        
        # Variables
        self.mode_var = tk.StringVar(value="2d")
        self.agents_var = tk.StringVar(value="")
        self.max_points_var = tk.StringVar(value="1000")
        self.ekf_agent_var = tk.StringVar(value="")
        self.blue_var = tk.BooleanVar(value=False)
        self.pos_inverted_var = tk.StringVar(value="")
        self.z_min_var = tk.StringVar(value="")
        self.z_max_var = tk.StringVar(value="")
        self.wing_trails_var = tk.BooleanVar(value=False)
        self.wing_span_var = tk.StringVar(value="10.0")
        self.skip_points_var = tk.StringVar(value="1")
        self.follow_agent_var = tk.StringVar(value="")
        self.camera_distance_var = tk.StringVar(value="200")
        self.no_axis_var = tk.BooleanVar(value=False)
        self.fancy_var = tk.BooleanVar(value=False)
        
        # Preset management
        self.presets = self.load_presets()
        self.current_preset_var = tk.StringVar(value="")
        
        self.create_widgets()
        self.update_mode_options()
        
    def _on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def create_widgets(self):
        main_frame = self.scrollable_frame
        
        # Title
        title_label = ttk.Label(main_frame, text="ROS2 Multi-Agent Dashboard Launcher", 
                                font=('Helvetica', 14, 'bold'))
        title_label.pack(pady=10)
        
        # ========== Presets Section ==========
        preset_frame = ttk.LabelFrame(main_frame, text="Presets", padding=10)
        preset_frame.pack(fill="x", padx=10, pady=5)
        
        # Preset dropdown row
        preset_select_frame = ttk.Frame(preset_frame)
        preset_select_frame.pack(fill="x", pady=2)
        
        ttk.Label(preset_select_frame, text="Load Preset:").pack(side="left")
        self.preset_combo = ttk.Combobox(preset_select_frame, textvariable=self.current_preset_var,
                                          state="readonly", width=25)
        self.preset_combo.pack(side="left", padx=5)
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset_selected)
        self.update_preset_list()
        
        ttk.Button(preset_select_frame, text="Load", command=self.load_selected_preset).pack(side="left", padx=2)
        ttk.Button(preset_select_frame, text="Delete", command=self.delete_preset).pack(side="left", padx=2)
        
        # Save preset row
        preset_save_frame = ttk.Frame(preset_frame)
        preset_save_frame.pack(fill="x", pady=5)
        
        ttk.Button(preset_save_frame, text="💾 Save Current as New Preset", 
                   command=self.save_as_new_preset).pack(side="left", padx=2)
        ttk.Button(preset_save_frame, text="📝 Update Selected Preset", 
                   command=self.update_current_preset).pack(side="left", padx=2)
        
        # ========== Mode Selection ==========
        mode_frame = ttk.LabelFrame(main_frame, text="Visualization Mode", padding=10)
        mode_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Radiobutton(mode_frame, text="2D View (Default)", variable=self.mode_var, 
                        value="2d", command=self.update_mode_options).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="3D View", variable=self.mode_var, 
                        value="3d", command=self.update_mode_options).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Top View Only (Trajectory only)", variable=self.mode_var, 
                        value="top", command=self.update_mode_options).pack(anchor="w")
        
        # ========== Common Options ==========
        common_frame = ttk.LabelFrame(main_frame, text="Common Options", padding=10)
        common_frame.pack(fill="x", padx=10, pady=5)
        
        # Agents filter
        agents_frame = ttk.Frame(common_frame)
        agents_frame.pack(fill="x", pady=2)
        ttk.Label(agents_frame, text="Filter Agents (IDs, space-separated):").pack(side="left")
        ttk.Entry(agents_frame, textvariable=self.agents_var, width=20).pack(side="right", expand=True, fill="x", padx=5)
        
        # Max path points
        max_points_frame = ttk.Frame(common_frame)
        max_points_frame.pack(fill="x", pady=2)
        ttk.Label(max_points_frame, text="Max Path Points:").pack(side="left")
        ttk.Entry(max_points_frame, textvariable=self.max_points_var, width=10).pack(side="right", padx=5)
        
        # Skip points
        skip_frame = ttk.Frame(common_frame)
        skip_frame.pack(fill="x", pady=2)
        ttk.Label(skip_frame, text="Skip Points (keep every Nth):").pack(side="left")
        ttk.Entry(skip_frame, textvariable=self.skip_points_var, width=10).pack(side="right", padx=5)
        
        # Position inverted agents
        pos_inv_frame = ttk.Frame(common_frame)
        pos_inv_frame.pack(fill="x", pady=2)
        ttk.Label(pos_inv_frame, text="NED Agents (pos_inverted IDs):").pack(side="left")
        ttk.Entry(pos_inv_frame, textvariable=self.pos_inverted_var, width=20).pack(side="right", expand=True, fill="x", padx=5)
        
        # Blue first agent checkbox
        ttk.Checkbutton(common_frame, text="Force Agent 1 to Blue (--blue)", 
                        variable=self.blue_var).pack(anchor="w", pady=2)
        
        # ========== 2D/Standard Options ==========
        self.standard_frame = ttk.LabelFrame(main_frame, text="2D Mode Options", padding=10)
        self.standard_frame.pack(fill="x", padx=10, pady=5)
        
        # EKF agent
        ekf_frame = ttk.Frame(self.standard_frame)
        ekf_frame.pack(fill="x", pady=2)
        ttk.Label(ekf_frame, text="EKF Agent ID (for target estimates):").pack(side="left")
        ttk.Entry(ekf_frame, textvariable=self.ekf_agent_var, width=10).pack(side="right", padx=5)
        
        # ========== 3D Options ==========
        self.options_3d_frame = ttk.LabelFrame(main_frame, text="3D Mode Options", padding=10)
        self.options_3d_frame.pack(fill="x", padx=10, pady=5)
        
        # Z bounds
        z_bounds_frame = ttk.Frame(self.options_3d_frame)
        z_bounds_frame.pack(fill="x", pady=2)
        ttk.Label(z_bounds_frame, text="Z Bounds (min, max):").pack(side="left")
        ttk.Entry(z_bounds_frame, textvariable=self.z_max_var, width=8).pack(side="right", padx=2)
        ttk.Entry(z_bounds_frame, textvariable=self.z_min_var, width=8).pack(side="right", padx=2)
        
        # Wing trails
        wing_frame = ttk.Frame(self.options_3d_frame)
        wing_frame.pack(fill="x", pady=2)
        self.wing_check = ttk.Checkbutton(wing_frame, text="Enable Wing Trails", 
                                           variable=self.wing_trails_var, command=self.update_wing_span_state)
        self.wing_check.pack(side="left")
        ttk.Label(wing_frame, text="Wingspan (m):").pack(side="left", padx=(20, 5))
        self.wing_span_entry = ttk.Entry(wing_frame, textvariable=self.wing_span_var, width=8)
        self.wing_span_entry.pack(side="left")
        
        # Follow agent
        follow_frame = ttk.Frame(self.options_3d_frame)
        follow_frame.pack(fill="x", pady=2)
        ttk.Label(follow_frame, text="Follow Agent ID:").pack(side="left")
        self.follow_entry = ttk.Entry(follow_frame, textvariable=self.follow_agent_var, width=10)
        self.follow_entry.pack(side="right", padx=5)
        
        # Camera distance
        cam_frame = ttk.Frame(self.options_3d_frame)
        cam_frame.pack(fill="x", pady=2)
        ttk.Label(cam_frame, text="Camera Distance:").pack(side="left")
        self.cam_dist_entry = ttk.Entry(cam_frame, textvariable=self.camera_distance_var, width=10)
        self.cam_dist_entry.pack(side="right", padx=5)
        
        # No axis checkbox
        self.no_axis_check = ttk.Checkbutton(self.options_3d_frame, text="Hide Axis (--no-axis)", 
                                              variable=self.no_axis_var)
        self.no_axis_check.pack(anchor="w", pady=2)
        
        # Fancy mode checkbox
        self.fancy_check = ttk.Checkbutton(self.options_3d_frame, text="Fancy Publication Mode (--fancy)", 
                                            variable=self.fancy_var)
        self.fancy_check.pack(anchor="w", pady=2)
        
        # ========== Command Preview ==========
        preview_frame = ttk.LabelFrame(main_frame, text="Command Preview", padding=10)
        preview_frame.pack(fill="x", padx=10, pady=5)
        
        self.command_text = tk.Text(preview_frame, height=4, wrap="word", state="disabled",
                                     font=('Courier', 9))
        self.command_text.pack(fill="x", pady=5)
        
        # Update button
        ttk.Button(preview_frame, text="Update Preview", 
                   command=self.update_command_preview).pack(pady=2)
        
        # ========== Action Buttons ==========
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Copy Command", 
                   command=self.copy_command).pack(side="left", padx=5)
        
        run_btn = ttk.Button(button_frame, text="🚀 Run Dashboard", 
                             command=self.run_dashboard)
        run_btn.pack(side="right", padx=5)
        
        # Style the run button
        style = ttk.Style()
        style.configure('Run.TButton', font=('Helvetica', 10, 'bold'))
        
        # Initial command preview
        self.update_command_preview()
    
    def update_mode_options(self):
        """Enable/disable options based on selected mode"""
        mode = self.mode_var.get()
        
        # 3D options are only available in 3D mode
        is_3d = (mode == "3d")
        state_3d = "normal" if is_3d else "disabled"
        
        for child in self.options_3d_frame.winfo_children():
            self._set_widget_state(child, state_3d)
        
        # Standard options (EKF) not available in top_view_only or 3D
        is_standard = (mode == "2d")
        state_standard = "normal" if is_standard else "disabled"
        
        for child in self.standard_frame.winfo_children():
            self._set_widget_state(child, state_standard)
        
        self.update_wing_span_state()
        self.update_command_preview()
    
    def _set_widget_state(self, widget, state):
        """Recursively set state for widget and its children"""
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_widget_state(child, state)
    
    def update_wing_span_state(self):
        """Enable/disable wingspan entry based on checkbox"""
        if self.mode_var.get() == "3d":
            state = "normal" if self.wing_trails_var.get() else "disabled"
            self.wing_span_entry.configure(state=state)
    
    def build_command(self):
        """Build the command from current options"""
        cmd = ["python3", "dashboard_ros.py"]
        
        mode = self.mode_var.get()
        
        # Mode flags
        if mode == "3d":
            cmd.append("--3d")
        elif mode == "top":
            cmd.append("--top_view_only")
        
        # Common options
        agents = self.agents_var.get().strip()
        if agents:
            cmd.extend(["--agents"] + agents.split())
        
        max_points = self.max_points_var.get().strip()
        if max_points and max_points != "1000":
            cmd.extend(["--max-path-points", max_points])
        
        skip_points = self.skip_points_var.get().strip()
        if skip_points and skip_points != "1":
            cmd.extend(["--skip-points", skip_points])
        
        pos_inverted = self.pos_inverted_var.get().strip()
        if pos_inverted:
            cmd.extend(["--pos_inverted"] + pos_inverted.split())
        
        if self.blue_var.get():
            cmd.append("--blue")
        
        # 2D-specific options
        if mode == "2d":
            ekf_agent = self.ekf_agent_var.get().strip()
            if ekf_agent:
                cmd.extend(["--ekf-agent", ekf_agent])
        
        # 3D-specific options
        if mode == "3d":
            z_min = self.z_min_var.get().strip()
            z_max = self.z_max_var.get().strip()
            if z_min and z_max:
                cmd.extend(["--z-bounds", z_min, z_max])
            
            if self.wing_trails_var.get():
                wingspan = self.wing_span_var.get().strip()
                if wingspan and wingspan != "10.0":
                    cmd.extend(["--wing-trails", wingspan])
                else:
                    cmd.append("--wing-trails")
            
            follow_agent = self.follow_agent_var.get().strip()
            if follow_agent:
                cmd.extend(["--follow", follow_agent])
            
            cam_dist = self.camera_distance_var.get().strip()
            if cam_dist and cam_dist != "200":
                cmd.extend(["--camera-distance", cam_dist])
            
            if self.no_axis_var.get():
                cmd.append("--no-axis")
            
            if self.fancy_var.get():
                cmd.append("--fancy")
        
        return cmd
    
    def update_command_preview(self):
        """Update the command preview text"""
        cmd = self.build_command()
        cmd_str = " ".join(cmd)
        
        self.command_text.configure(state="normal")
        self.command_text.delete("1.0", tk.END)
        self.command_text.insert("1.0", cmd_str)
        self.command_text.configure(state="disabled")
    
    def copy_command(self):
        """Copy command to clipboard"""
        cmd = self.build_command()
        cmd_str = " ".join(cmd)
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd_str)
        messagebox.showinfo("Copied", "Command copied to clipboard!")
    
    def run_dashboard(self):
        """Run the dashboard with current configuration"""
        cmd = self.build_command()
        
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            # Run in a new terminal or subprocess
            print(f"Running: {' '.join(cmd)}")
            subprocess.Popen(cmd, cwd=script_dir)
            
            # Optionally close the launcher
            # self.root.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run dashboard:\n{e}")
    
    # ========== Preset Management Methods ==========
    
    def load_presets(self):
        """Load presets from JSON file"""
        if os.path.exists(PRESETS_FILE):
            try:
                with open(PRESETS_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load presets: {e}")
                return {}
        return {}
    
    def save_presets(self):
        """Save presets to JSON file"""
        try:
            with open(PRESETS_FILE, 'w') as f:
                json.dump(self.presets, f, indent=2)
        except IOError as e:
            messagebox.showerror("Error", f"Could not save presets:\n{e}")
    
    def update_preset_list(self):
        """Update the preset dropdown with current presets"""
        preset_names = sorted(self.presets.keys())
        self.preset_combo['values'] = preset_names
        if preset_names and not self.current_preset_var.get():
            pass  # Don't auto-select
    
    def get_current_config(self):
        """Get current configuration as a dictionary"""
        return {
            'mode': self.mode_var.get(),
            'agents': self.agents_var.get(),
            'max_points': self.max_points_var.get(),
            'ekf_agent': self.ekf_agent_var.get(),
            'blue': self.blue_var.get(),
            'pos_inverted': self.pos_inverted_var.get(),
            'z_min': self.z_min_var.get(),
            'z_max': self.z_max_var.get(),
            'wing_trails': self.wing_trails_var.get(),
            'wing_span': self.wing_span_var.get(),
            'skip_points': self.skip_points_var.get(),
            'follow_agent': self.follow_agent_var.get(),
            'camera_distance': self.camera_distance_var.get(),
            'no_axis': self.no_axis_var.get(),
            'fancy': self.fancy_var.get(),
        }
    
    def apply_config(self, config):
        """Apply a configuration dictionary to the UI"""
        self.mode_var.set(config.get('mode', '2d'))
        self.agents_var.set(config.get('agents', ''))
        self.max_points_var.set(config.get('max_points', '1000'))
        self.ekf_agent_var.set(config.get('ekf_agent', ''))
        self.blue_var.set(config.get('blue', False))
        self.pos_inverted_var.set(config.get('pos_inverted', ''))
        self.z_min_var.set(config.get('z_min', ''))
        self.z_max_var.set(config.get('z_max', ''))
        self.wing_trails_var.set(config.get('wing_trails', False))
        self.wing_span_var.set(config.get('wing_span', '10.0'))
        self.skip_points_var.set(config.get('skip_points', '1'))
        self.follow_agent_var.set(config.get('follow_agent', ''))
        self.camera_distance_var.set(config.get('camera_distance', '200'))
        self.no_axis_var.set(config.get('no_axis', False))
        self.fancy_var.set(config.get('fancy', False))
        
        # Update UI state based on mode
        self.update_mode_options()
    
    def on_preset_selected(self, event=None):
        """Handle preset selection from dropdown"""
        # Just update selection, don't auto-load
        pass
    
    def load_selected_preset(self):
        """Load the selected preset into the UI"""
        preset_name = self.current_preset_var.get()
        if not preset_name:
            messagebox.showwarning("No Preset Selected", "Please select a preset to load.")
            return
        
        if preset_name in self.presets:
            self.apply_config(self.presets[preset_name])
            self.update_command_preview()
            messagebox.showinfo("Preset Loaded", f"Loaded preset: {preset_name}")
        else:
            messagebox.showerror("Error", f"Preset '{preset_name}' not found.")
    
    def save_as_new_preset(self):
        """Save current configuration as a new preset"""
        preset_name = simpledialog.askstring("Save Preset", 
                                              "Enter a name for this preset:",
                                              parent=self.root)
        if not preset_name:
            return
        
        preset_name = preset_name.strip()
        if not preset_name:
            messagebox.showwarning("Invalid Name", "Preset name cannot be empty.")
            return
        
        # Check if preset already exists
        if preset_name in self.presets:
            if not messagebox.askyesno("Overwrite?", 
                                        f"Preset '{preset_name}' already exists. Overwrite?"):
                return
        
        # Save the preset
        self.presets[preset_name] = self.get_current_config()
        self.save_presets()
        self.update_preset_list()
        self.current_preset_var.set(preset_name)
        messagebox.showinfo("Saved", f"Preset '{preset_name}' saved successfully!")
    
    def update_current_preset(self):
        """Update the currently selected preset with current configuration"""
        preset_name = self.current_preset_var.get()
        if not preset_name:
            messagebox.showwarning("No Preset Selected", 
                                   "Please select a preset to update, or use 'Save as New Preset'.")
            return
        
        if preset_name not in self.presets:
            messagebox.showerror("Error", f"Preset '{preset_name}' not found.")
            return
        
        if messagebox.askyesno("Update Preset?", 
                               f"Update preset '{preset_name}' with current settings?"):
            self.presets[preset_name] = self.get_current_config()
            self.save_presets()
            messagebox.showinfo("Updated", f"Preset '{preset_name}' updated successfully!")
    
    def delete_preset(self):
        """Delete the selected preset"""
        preset_name = self.current_preset_var.get()
        if not preset_name:
            messagebox.showwarning("No Preset Selected", "Please select a preset to delete.")
            return
        
        if preset_name not in self.presets:
            messagebox.showerror("Error", f"Preset '{preset_name}' not found.")
            return
        
        if messagebox.askyesno("Delete Preset?", 
                               f"Are you sure you want to delete preset '{preset_name}'?"):
            del self.presets[preset_name]
            self.save_presets()
            self.update_preset_list()
            self.current_preset_var.set("")
            messagebox.showinfo("Deleted", f"Preset '{preset_name}' deleted.")


def main():
    root = tk.Tk()
    app = DashboardLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
