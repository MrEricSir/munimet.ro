#!/usr/bin/env python3
"""
Segment-level evaluation tool for the station delay detection algorithm.

Displays images with algorithm predictions overlaid, lets the user correct
false positives (uncheck) and false negatives (add missed delays), then
saves corrected evaluations for parameter fitting.

Usage:
    python scripts/evaluate_segments.py
"""

import os
import json
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageTk
import cv2
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from scripts.detect_stations import (
        analyze_image_detailed, PUBLIC_STATIONS, SUBWAY_CODES, CENTRAL_CODES,
        ICON_COUNT_THRESHOLD, TOTAL_ICON_THRESHOLD, RED_TRACK_THRESHOLD,
    )
except ModuleNotFoundError:
    from detect_stations import (
        analyze_image_detailed, PUBLIC_STATIONS, SUBWAY_CODES, CENTRAL_CODES,
        ICON_COUNT_THRESHOLD, TOTAL_ICON_THRESHOLD, RED_TRACK_THRESHOLD,
    )

# Configuration
IMAGE_FOLDER = "artifacts/training_data/images"
EVALUATIONS_FILE = "artifacts/training_data/station_evaluations.json"
MAX_IMAGE_WIDTH = 1200
MAX_IMAGE_HEIGHT = 800  # Full image height (will be scrollable)
IMAGE_DISPLAY_HEIGHT = 350  # Visible viewport height

# Direction abbreviations for display
DIR_ABBREV = {
    'Westbound': 'WB',
    'Eastbound': 'EB',
    'Northbound': 'NB',
    'Southbound': 'SB',
}


def dir_abbrev(direction):
    """Get standard abbreviation for direction."""
    return DIR_ABBREV.get(direction, direction[:2])


class SegmentEvaluator:
    def __init__(self, root):
        self.root = root
        self.root.title("Station Delay Evaluation Tool")

        # Data
        self.image_files = []
        self.current_index = 0
        self.evaluations = {}  # image_path -> evaluation dict
        self.current_analysis = None  # cached analyze_image_detailed() result
        self.segment_vars = []  # list of (BooleanVar, segment_info) for checkboxes
        self.added_delays = []  # manually added delays for current image

        # Image display state
        self.original_photo = None  # PIL PhotoImage of original
        self.debug_photo = None     # PIL PhotoImage with overlay
        self.show_overlay = None    # BooleanVar for overlay toggle

        # Load existing evaluations
        self.load_evaluations()

        # Load image files
        self.load_image_files()

        # Setup UI
        self.setup_ui()

        # Display first unevaluated image
        if self.image_files:
            self.jump_to_first_unevaluated()
            self.display_image()
        else:
            messagebox.showinfo("No Images", f"No images found in {IMAGE_FOLDER}")
            self.root.quit()

    def load_evaluations(self):
        """Load existing evaluations from JSON file."""
        if os.path.exists(EVALUATIONS_FILE):
            try:
                with open(EVALUATIONS_FILE, 'r') as f:
                    data = json.load(f)
                self.evaluations = {
                    item['image_path']: item
                    for item in data.get('evaluations', [])
                }
                print(f"Loaded {len(self.evaluations)} existing evaluations")
            except Exception as e:
                print(f"Error loading evaluations: {e}")
                self.evaluations = {}
        else:
            self.evaluations = {}

    def save_evaluations(self):
        """Save evaluations to JSON file."""
        try:
            eval_list = sorted(
                self.evaluations.values(),
                key=lambda x: x['image_path']
            )
            data = {
                'evaluations': eval_list,
                'total_evaluated': len(eval_list),
                'last_updated': datetime.now().isoformat(),
                'thresholds_at_evaluation': {
                    'ICON_COUNT_THRESHOLD': ICON_COUNT_THRESHOLD,
                    'TOTAL_ICON_THRESHOLD': TOTAL_ICON_THRESHOLD,
                    'RED_TRACK_THRESHOLD': RED_TRACK_THRESHOLD,
                },
            }

            os.makedirs(os.path.dirname(EVALUATIONS_FILE), exist_ok=True)
            with open(EVALUATIONS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save evaluations: {e}")
            return False

    def load_image_files(self):
        """Load jpg image files from the image folder."""
        if not os.path.exists(IMAGE_FOLDER):
            print(f"Image folder not found: {IMAGE_FOLDER}")
            return

        image_dir = Path(IMAGE_FOLDER)
        jpg_files = set(image_dir.glob("*.jpg"))
        JPG_files = set(image_dir.glob("*.JPG"))
        self.image_files = sorted([
            str(f).replace("\\", "/") for f in jpg_files | JPG_files
        ])

        evaluated_count = sum(
            1 for f in self.image_files if f in self.evaluations
        )
        print(f"Found {len(self.image_files)} images, {evaluated_count} evaluated")

    def is_evaluated(self, image_path):
        """Check if an image has been evaluated."""
        return image_path in self.evaluations

    def setup_ui(self):
        """Setup the user interface."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Progress and filename header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        header_frame.columnconfigure(1, weight=1)

        self.progress_label = ttk.Label(
            header_frame, text="", font=('Arial', 12, 'bold')
        )
        self.progress_label.grid(row=0, column=0, sticky=tk.W)

        self.filename_label = ttk.Label(
            header_frame, text="", font=('Arial', 11), foreground="darkblue"
        )
        self.filename_label.grid(row=0, column=1, sticky=tk.E, padx=20)

        # Detection quality indicator
        self.quality_label = ttk.Label(
            header_frame, text="", font=('Arial', 10)
        )
        self.quality_label.grid(row=0, column=2, sticky=tk.E, padx=10)

        # Scrollable image display
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        image_frame.columnconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(
            image_frame, height=IMAGE_DISPLAY_HEIGHT, highlightthickness=1,
            highlightbackground="gray"
        )
        image_scrollbar = ttk.Scrollbar(
            image_frame, orient="vertical", command=self.image_canvas.yview
        )
        self.image_canvas.configure(yscrollcommand=image_scrollbar.set)

        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        image_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Store reference for the image on canvas
        self.canvas_image_id = None
        self.current_img_width = 0
        self.current_img_height = 0

        # Bind resize to recenter image
        self.image_canvas.bind('<Configure>', self._on_canvas_resize)

        # Bind click for segment inspection
        self.image_canvas.bind('<Button-1>', self._on_canvas_click)

        # Overlay toggle and click info on same row
        controls_frame = ttk.Frame(image_frame)
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)

        self.show_overlay = tk.BooleanVar(value=True)
        overlay_check = ttk.Checkbutton(
            controls_frame, text="Show detection overlay",
            variable=self.show_overlay, command=self._toggle_overlay
        )
        overlay_check.pack(side=tk.LEFT, padx=5)

        ttk.Label(controls_frame, text="|").pack(side=tk.LEFT, padx=5)

        self.click_info_label = ttk.Label(
            controls_frame, text="Click image to inspect regions",
            foreground="gray"
        )
        self.click_info_label.pack(side=tk.LEFT, padx=5)

        # Color legend
        ttk.Label(controls_frame, text="|").pack(side=tk.LEFT, padx=5)
        legend_frame = ttk.Frame(controls_frame)
        legend_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="Legend:", font=('Arial', 9)).pack(side=tk.LEFT)
        tk.Label(legend_frame, text=" Green=station ", bg='green', fg='white',
                font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text=" Gray=segment ", bg='gray', fg='white',
                font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text=" Orange=train ", bg='orange', fg='black',
                font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text=" Red=delay ", bg='red', fg='white',
                font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text=" Magenta=outage ", bg='magenta', fg='white',
                font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text=" Cyan=bunched ", bg='cyan', fg='black',
                font=('Arial', 8)).pack(side=tk.LEFT, padx=2)

        # Predictions and corrections frame with scrollbar
        pred_frame = ttk.LabelFrame(
            main_frame, text="Segment Predictions", padding="5"
        )
        pred_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        pred_frame.columnconfigure(0, weight=1)

        # Canvas + scrollbar for segment checkboxes
        self.segments_canvas = tk.Canvas(pred_frame, height=120, highlightthickness=0)
        segments_scrollbar = ttk.Scrollbar(
            pred_frame, orient="vertical", command=self.segments_canvas.yview
        )
        self.segments_frame = ttk.Frame(self.segments_canvas)

        self.segments_frame.bind(
            "<Configure>",
            lambda e: self.segments_canvas.configure(
                scrollregion=self.segments_canvas.bbox("all")
            )
        )
        self.segments_canvas.create_window(
            (0, 0), window=self.segments_frame, anchor="nw"
        )
        self.segments_canvas.configure(yscrollcommand=segments_scrollbar.set)

        self.segments_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        segments_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Info label for no-delay images
        self.no_delay_label = ttk.Label(
            pred_frame, text="", foreground="gray", font=('Arial', 10)
        )
        self.no_delay_label.grid(row=1, column=0, sticky=tk.W, padx=5)

        # Added delays display
        self.added_frame = ttk.LabelFrame(
            main_frame, text="Manually Added Delays", padding="5"
        )
        self.added_frame.grid(row=3, column=0, pady=2, sticky=(tk.W, tk.E))
        self.added_delays_label = ttk.Label(
            self.added_frame, text="None", foreground="gray"
        )
        self.added_delays_label.grid(row=0, column=0, sticky=tk.W)

        # Add delay controls
        add_frame = ttk.Frame(main_frame)
        add_frame.grid(row=4, column=0, pady=5, sticky=tk.W)

        ttk.Label(add_frame, text="Add delay:").grid(row=0, column=0, padx=2)

        self.add_direction_var = tk.StringVar(value="Westbound")
        dir_combo = ttk.Combobox(
            add_frame, textvariable=self.add_direction_var,
            values=["Westbound", "Eastbound", "Northbound", "Southbound"],
            state="readonly", width=10
        )
        dir_combo.grid(row=0, column=1, padx=2)

        ttk.Label(add_frame, text="from").grid(row=0, column=2, padx=2)

        # Use only public stations (excludes maintenance platforms)
        station_names = [name for _, name in PUBLIC_STATIONS]
        self.add_from_var = tk.StringVar(value=station_names[0])
        from_combo = ttk.Combobox(
            add_frame, textvariable=self.add_from_var,
            values=station_names, state="readonly", width=14
        )
        from_combo.grid(row=0, column=3, padx=2)

        ttk.Label(add_frame, text="to").grid(row=0, column=4, padx=2)

        self.add_to_var = tk.StringVar(value=station_names[1])
        to_combo = ttk.Combobox(
            add_frame, textvariable=self.add_to_var,
            values=station_names, state="readonly", width=14
        )
        to_combo.grid(row=0, column=5, padx=2)

        add_button = ttk.Button(
            add_frame, text="+ Add", command=self.add_delay
        )
        add_button.grid(row=0, column=6, padx=5)

        clear_added_button = ttk.Button(
            add_frame, text="Clear Added", command=self.clear_added_delays
        )
        clear_added_button.grid(row=0, column=7, padx=5)

        # Notes
        notes_frame = ttk.Frame(main_frame)
        notes_frame.grid(row=5, column=0, pady=2, sticky=(tk.W, tk.E))
        notes_frame.columnconfigure(1, weight=1)

        ttk.Label(notes_frame, text="Notes:").grid(row=0, column=0, padx=2)
        self.notes_entry = tk.Entry(notes_frame, font=('Arial', 10))
        self.notes_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=2)

        # Navigation buttons
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=6, column=0, pady=10)

        # Index jump
        index_frame = ttk.LabelFrame(nav_frame, text='Index')
        index_frame.grid(row=0, column=0, padx=5, sticky=tk.W)

        self.jump_entry = ttk.Entry(index_frame, width=8)
        self.jump_entry.grid(row=0, column=0, padx=3, pady=3)
        ttk.Button(
            index_frame, text="Go", command=self.jump_to_index
        ).grid(row=0, column=1, padx=3, pady=3)

        # Navigation
        self.prev_uneval_btn = ttk.Button(
            nav_frame, text="<< Prev Unevaluated",
            command=self.prev_unevaluated
        )
        self.prev_uneval_btn.grid(row=0, column=1, padx=3)

        self.prev_btn = ttk.Button(
            nav_frame, text="< Prev", command=self.prev_image
        )
        self.prev_btn.grid(row=0, column=2, padx=3)

        self.accept_btn = ttk.Button(
            nav_frame, text="Accept All (A)",
            command=self.accept_all
        )
        self.accept_btn.grid(row=0, column=3, padx=3)

        self.save_next_btn = ttk.Button(
            nav_frame, text="Save & Next (S)",
            command=self.save_and_next
        )
        self.save_next_btn.grid(row=0, column=4, padx=3)

        self.next_btn = ttk.Button(
            nav_frame, text="Next >", command=self.next_image
        )
        self.next_btn.grid(row=0, column=5, padx=3)

        self.next_uneval_btn = ttk.Button(
            nav_frame, text="Next Unevaluated >>",
            command=self.next_unevaluated
        )
        self.next_uneval_btn.grid(row=0, column=6, padx=3)

        # Status bar
        self.status_label = ttk.Label(
            main_frame, text="", foreground="gray"
        )
        self.status_label.grid(row=7, column=0, pady=2)

        # Keyboard shortcuts
        shortcuts_text = (
            "A=Accept All | S=Save & Next | Ctrl+Left/Right=Prev/Next | "
            "Ctrl+Shift+Left/Right=Prev/Next Unevaluated | D=Add Delay focus | "
            "Ctrl+G=Jump"
        )
        ttk.Label(
            main_frame, text=shortcuts_text, foreground="gray",
            font=('Arial', 9)
        ).grid(row=8, column=0, pady=2)

        # Bind keyboard shortcuts
        self.root.bind('a', lambda e: self.accept_all()
                       if not self._in_entry(e) else None)
        self.root.bind('s', lambda e: self.save_and_next()
                       if not self._in_entry(e) else None)
        self.root.bind('<Control-Return>', lambda e: self.save_and_next())
        self.root.bind('<Control-Right>', lambda e: self.next_image())
        self.root.bind('<Control-Left>', lambda e: self.prev_image())
        self.root.bind('<Control-Shift-Right>', lambda e: self.next_unevaluated())
        self.root.bind('<Control-Shift-Left>', lambda e: self.prev_unevaluated())
        self.root.bind('<Control-g>', lambda e: self.jump_entry.focus())
        self.root.bind('d', lambda e: self._focus_add_delay()
                       if not self._in_entry(e) else None)
        self.jump_entry.bind('<Return>', lambda e: self.jump_to_index())

    def _in_entry(self, event):
        """Check if the event originated in a text entry widget."""
        return isinstance(event.widget, (tk.Entry, ttk.Entry))

    def _focus_add_delay(self):
        """Focus the add-delay from combobox."""
        # Focus the first combo in the add-delay row
        for child in self.root.winfo_children():
            pass  # Can't easily access nested combos; just skip for now

    def display_image(self):
        """Display the current image with algorithm predictions."""
        if not self.image_files or self.current_index >= len(self.image_files):
            return

        image_path = self.image_files[self.current_index]

        # Update progress and filename
        evaluated_count = sum(
            1 for f in self.image_files if self.is_evaluated(f)
        )
        self.progress_label.config(
            text=f"Image {self.current_index + 1} of {len(self.image_files)} | "
                 f"{evaluated_count} evaluated"
        )

        # Show filename with detection quality indicator
        filename = Path(image_path).name
        self.filename_label.config(text=filename)

        # Detection quality will be updated after analysis
        self.quality_label.config(text="...", foreground="gray")

        # Run detection algorithm
        try:
            self.current_analysis = analyze_image_detailed(image_path)
        except Exception as e:
            self.status_label.config(
                text=f"Error analyzing image: {e}", foreground="red"
            )
            self.quality_label.config(text="Error", foreground="red")
            self.current_analysis = None
            return

        # Update detection quality indicator
        meta = self.current_analysis['result']['detection_meta']
        upper_found = meta['upper_labels_found']
        lower_found = meta['lower_labels_found']
        expected = meta['expected_per_row']
        total_found = upper_found + lower_found
        total_expected = expected * 2
        trains = self.current_analysis['total_train_icons']
        bunched = len(self.current_analysis.get('bunched_stations', []))

        if meta['layout_ok']:
            quality_text = f"✓ {total_found}/{total_expected} stations | {trains} trains"
            quality_color = "green"
        else:
            quality_text = f"⚠ {total_found}/{total_expected} stations | {trains} trains"
            quality_color = "orange"

        if bunched > 0:
            quality_text += f" | {bunched} bunched"

        self.quality_label.config(text=quality_text, foreground=quality_color)

        # Load both original and debug images
        debug_img = self.current_analysis['debug_img']
        original_img = cv2.imread(image_path)

        # Convert to PIL and create PhotoImages for both
        debug_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        debug_pil = Image.fromarray(debug_rgb)
        debug_pil.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        self.debug_photo = ImageTk.PhotoImage(debug_pil)

        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)
        original_pil.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        self.original_photo = ImageTk.PhotoImage(original_pil)

        # Store dimensions for resize handling
        self.current_img_width, self.current_img_height = debug_pil.size

        # Update canvas - center image horizontally
        self._update_canvas_image()

        # Center view vertically on the track area (~50-60% of image)
        if self.current_img_height > IMAGE_DISPLAY_HEIGHT:
            center_y = self.current_img_height * 0.55  # Center on track region
            scroll_fraction = max(0, (center_y - IMAGE_DISPLAY_HEIGHT / 2)) / self.current_img_height
            self.image_canvas.yview_moveto(scroll_fraction)

        # Build segment checkbox list
        self._build_segment_checkboxes(image_path)

        # Load notes if previously evaluated
        self.notes_entry.delete(0, tk.END)
        if image_path in self.evaluations:
            notes = self.evaluations[image_path].get('notes', '')
            if notes:
                self.notes_entry.insert(0, notes)

        # Update navigation button states
        self.prev_btn.config(
            state=tk.NORMAL if self.current_index > 0 else tk.DISABLED
        )
        self.next_btn.config(
            state=tk.NORMAL if self.current_index < len(self.image_files) - 1
            else tk.DISABLED
        )

        # Update status
        if image_path in self.evaluations:
            eval_data = self.evaluations[image_path]
            delay_count = len(eval_data.get('segments', []))
            self.status_label.config(
                text=f"Previously evaluated: {delay_count} delay(s) | "
                     f"{eval_data.get('evaluated_at', 'unknown')}",
                foreground="green"
            )
        else:
            self.status_label.config(
                text="Not yet evaluated", foreground="orange"
            )

        # Update jump entry
        self.jump_entry.delete(0, tk.END)
        self.jump_entry.insert(0, str(self.current_index))

    def _update_canvas_image(self):
        """Update canvas with current image, centered horizontally."""
        if self.debug_photo is None or self.original_photo is None:
            return

        photo = self.debug_photo if self.show_overlay.get() else self.original_photo
        canvas_width = self.image_canvas.winfo_width()
        if canvas_width < 100:
            canvas_width = MAX_IMAGE_WIDTH

        center_x = canvas_width // 2

        self.image_canvas.delete("all")
        self.canvas_image_id = self.image_canvas.create_image(
            center_x, 0, anchor="n", image=photo
        )
        # Keep references to BOTH photos to prevent garbage collection
        self.image_canvas.debug_photo = self.debug_photo
        self.image_canvas.original_photo = self.original_photo
        self.image_canvas.config(
            scrollregion=(0, 0, canvas_width, self.current_img_height)
        )

    def _toggle_overlay(self):
        """Toggle between original and debug overlay image."""
        self._update_canvas_image()

    def _on_canvas_resize(self, event):
        """Recenter image when canvas is resized."""
        if self.canvas_image_id is not None:
            self._update_canvas_image()

    def _on_canvas_click(self, event):
        """Handle click on image canvas to inspect regions."""
        if self.current_analysis is None:
            return

        # Convert canvas coordinates to image coordinates
        canvas_x = self.image_canvas.canvasx(event.x)
        canvas_y = self.image_canvas.canvasy(event.y)

        # Account for image centering
        canvas_width = self.image_canvas.winfo_width()
        img_offset_x = (canvas_width - self.current_img_width) // 2
        img_x = int(canvas_x - img_offset_x)
        img_y = int(canvas_y)

        # Check if click is within image bounds
        if img_x < 0 or img_x >= self.current_img_width:
            self.click_info_label.config(text="Click outside image", foreground="gray")
            return
        if img_y < 0 or img_y >= self.current_img_height:
            self.click_info_label.config(text="Click outside image", foreground="gray")
            return

        # Check for train icon click
        train_icons = self.current_analysis.get('train_icons', [])
        route_codes = self.current_analysis.get('route_codes', {})
        for i, icon in enumerate(train_icons):
            if (icon['x'] <= img_x <= icon['x'] + icon['w'] and
                icon['y'] <= img_y <= icon['y'] + icon['h']):
                route = route_codes.get(i, None)
                route_str = f" (route: {route})" if route else ""
                self.click_info_label.config(
                    text=f"Train icon at ({icon['cx']}, {icon['cy']}){route_str}",
                    foreground="blue"
                )
                return

        # Check for station bunching
        bunched = self.current_analysis.get('bunched_stations', [])
        for b in bunched:
            dist = ((img_x - b['station_x'])**2 + (img_y - b['station_y'])**2) ** 0.5
            if dist < 50:
                self.click_info_label.config(
                    text=f"BUNCHING at {b['station_name']}: {b['train_count']} trains",
                    foreground="red"
                )
                return

        # Check for segment region click
        upper_regions = self.current_analysis.get('upper_regions', [])
        lower_regions = self.current_analysis.get('lower_regions', [])
        for region in upper_regions + lower_regions:
            if (region['x_min'] <= img_x <= region['x_max'] and
                region['y_min'] <= img_y <= region['y_max']):
                # Find matching segment info
                for seg in self.current_analysis['segments']:
                    if (seg['from_code'] == region['from_code'] and
                        seg['to_code'] == region['to_code'] and
                        seg['direction'] == region['direction']):
                        delay_str = "DELAY" if seg['predicted_delay'] else "OK"
                        self.click_info_label.config(
                            text=f"{seg['direction']}: {seg['from_name']} -> {seg['to_name']} | "
                                 f"icons: {seg['icon_count']}, red: {seg['red_ratio']:.1%} | {delay_str}",
                            foreground="green" if not seg['predicted_delay'] else "orange"
                        )
                        return

        # Check for station label click
        upper_stations = self.current_analysis.get('upper_stations', [])
        lower_stations = self.current_analysis.get('lower_stations', [])
        for code, name, sx, sy in upper_stations + lower_stations:
            if abs(img_x - sx) < 30 and abs(img_y - sy) < 20:
                self.click_info_label.config(
                    text=f"Station: {name} ({code})",
                    foreground="purple"
                )
                return

        self.click_info_label.config(text=f"Position: ({img_x}, {img_y})", foreground="gray")

    def _build_segment_checkboxes(self, image_path):
        """Build the checkbox list of predicted delays."""
        # Clear existing checkboxes
        for widget in self.segments_frame.winfo_children():
            widget.destroy()
        self.segment_vars = []
        self.added_delays = []
        self.added_delays_label.config(text="None", foreground="gray")

        if self.current_analysis is None:
            return

        segments = self.current_analysis['segments']
        predicted_delays = [s for s in segments if s['predicted_delay']]
        total_icons = self.current_analysis['total_train_icons']

        # Check for system-wide spread delay
        has_spread = (
            total_icons >= TOTAL_ICON_THRESHOLD and not predicted_delays
        )

        # Load previous evaluation to restore corrections
        prev_eval = self.evaluations.get(image_path)
        prev_delay_set = set()
        if prev_eval:
            for seg in prev_eval.get('segments', []):
                key = (seg['from'], seg['to'], seg['direction'])
                prev_delay_set.add(key)

        if not predicted_delays and not has_spread:
            self.no_delay_label.config(
                text=f"No delays predicted ({total_icons} train icons detected). "
                     f"Use 'Add delay' below if a delay was missed."
            )
        else:
            self.no_delay_label.config(text="")

        row = 0

        # Show header with summary and thresholds
        delay_count = len(predicted_delays)
        spread_text = " + spread delay" if has_spread else ""
        header_text = (
            f"Predicted: {delay_count} segment delay(s){spread_text}  |  "
            f"Train icons: {total_icons}  |  "
            f"Thresholds: icons≥{ICON_COUNT_THRESHOLD}, "
            f"total≥{TOTAL_ICON_THRESHOLD}, "
            f"red≥{RED_TRACK_THRESHOLD:.0%}"
        )
        header = ttk.Label(
            self.segments_frame,
            text=header_text,
            font=('Arial', 9), foreground="gray"
        )
        header.grid(row=row, column=0, sticky=tk.W, padx=5)
        row += 1

        # Add checkboxes for each predicted delay
        for seg in predicted_delays:
            var = tk.BooleanVar(value=True)

            # If previously evaluated, restore the user's correction
            if prev_eval:
                key = (seg['from_name'], seg['to_name'], seg['direction'])
                var.set(key in prev_delay_set)

            reason_text = {
                'icon_cluster': f"{seg['icon_count']} icons",
                'red_outage': f"red {seg['red_ratio']:.1%}",
            }.get(seg['reason'], seg['reason'] or '')

            label_text = (
                f"{dir_abbrev(seg['direction'])}: {seg['from_name']} -> "
                f"{seg['to_name']} ({reason_text})"
            )

            cb = tk.Checkbutton(
                self.segments_frame, text=label_text, variable=var,
                font=('Arial', 10), anchor=tk.W
            )
            cb.grid(row=row, column=0, sticky=tk.W, padx=10)
            self.segment_vars.append((var, seg))
            row += 1

        # Show spread delay if applicable
        if has_spread:
            var = tk.BooleanVar(value=True)
            if prev_eval:
                key = ('Multiple', 'stations', 'Both')
                var.set(key in prev_delay_set)

            cb = tk.Checkbutton(
                self.segments_frame,
                text=f"System-wide spread delay ({total_icons} total icons)",
                variable=var, font=('Arial', 10), anchor=tk.W
            )
            cb.grid(row=row, column=0, sticky=tk.W, padx=10)
            self.segment_vars.append((var, {
                'from_name': 'Multiple', 'to_name': 'stations',
                'direction': 'Both', 'reason': 'high_total_count',
                'icon_count': total_icons,
            }))
            row += 1

        # Also show non-delay segments that were previously marked as delays
        # (i.e., the user previously added a delay the algorithm missed)
        if prev_eval:
            algo_delay_keys = set()
            for seg in predicted_delays:
                algo_delay_keys.add(
                    (seg['from_name'], seg['to_name'], seg['direction'])
                )
            if has_spread:
                algo_delay_keys.add(('Multiple', 'stations', 'Both'))

            for prev_seg in prev_eval.get('segments', []):
                key = (prev_seg['from'], prev_seg['to'], prev_seg['direction'])
                if key not in algo_delay_keys:
                    # This was a manually added delay from a previous evaluation
                    self.added_delays.append({
                        'from': prev_seg['from'],
                        'to': prev_seg['to'],
                        'direction': prev_seg['direction'],
                    })

            self._update_added_delays_display()

    def add_delay(self):
        """Add a manually identified delay."""
        from_name = self.add_from_var.get()
        to_name = self.add_to_var.get()
        direction = self.add_direction_var.get()

        if from_name == to_name:
            messagebox.showwarning(
                "Invalid Segment", "From and To stations must be different."
            )
            return

        # Check for duplicates
        for d in self.added_delays:
            if (d['from'] == from_name and d['to'] == to_name
                    and d['direction'] == direction):
                messagebox.showinfo("Duplicate", "This delay is already added.")
                return

        # Also check against algorithm predictions
        for var, seg in self.segment_vars:
            if (seg['from_name'] == from_name and seg['to_name'] == to_name
                    and seg['direction'] == direction):
                # Already predicted - just ensure it's checked
                var.set(True)
                self.status_label.config(
                    text="Already predicted by algorithm - checkbox checked",
                    foreground="blue"
                )
                return

        self.added_delays.append({
            'from': from_name,
            'to': to_name,
            'direction': direction,
        })
        self._update_added_delays_display()

    def clear_added_delays(self):
        """Clear all manually added delays."""
        self.added_delays = []
        self._update_added_delays_display()

    def _update_added_delays_display(self):
        """Update the display of manually added delays."""
        if not self.added_delays:
            self.added_delays_label.config(text="None", foreground="gray")
        else:
            parts = []
            for d in self.added_delays:
                parts.append(
                    f"{dir_abbrev(d['direction'])}: {d['from']} -> {d['to']}"
                )
            self.added_delays_label.config(
                text=" | ".join(parts), foreground="blue"
            )

    def _collect_current_evaluation(self):
        """Collect the current evaluation from checkboxes + added delays."""
        image_path = self.image_files[self.current_index]

        # Collect checked algorithm predictions
        delay_segments = []
        for var, seg in self.segment_vars:
            if var.get():
                delay_segments.append({
                    'from': seg.get('from_name', seg.get('from', '')),
                    'to': seg.get('to_name', seg.get('to', '')),
                    'direction': seg['direction'],
                    'has_delay': True,
                })

        # Add manually added delays
        for d in self.added_delays:
            delay_segments.append({
                'from': d['from'],
                'to': d['to'],
                'direction': d['direction'],
                'has_delay': True,
            })

        notes = self.notes_entry.get().strip()

        return {
            'image_path': image_path,
            'evaluated_at': datetime.now().isoformat(),
            'segments': delay_segments,
            'notes': notes if notes else None,
            'algorithm_meta': {
                'total_train_icons': (
                    self.current_analysis['total_train_icons']
                    if self.current_analysis else 0
                ),
                'predicted_delay_count': sum(
                    1 for s in (self.current_analysis or {}).get('segments', [])
                    if s.get('predicted_delay')
                ),
            },
        }

    def save_current(self):
        """Save the current evaluation."""
        if not self.image_files:
            return False

        evaluation = self._collect_current_evaluation()
        self.evaluations[evaluation['image_path']] = evaluation
        return self.save_evaluations()

    def accept_all(self):
        """Accept all predictions as-is and move to next."""
        # All checkboxes are already checked by default, so just save
        if self.save_current():
            self.status_label.config(
                text="Accepted all predictions", foreground="green"
            )
            if self.current_index < len(self.image_files) - 1:
                self._advance_to_next_unevaluated()
            else:
                messagebox.showinfo("Done", "Last image reached.")

    def save_and_next(self):
        """Save current evaluation and move to next."""
        if self.save_current():
            self.status_label.config(text="Saved", foreground="green")
            if self.current_index < len(self.image_files) - 1:
                self._advance_to_next_unevaluated()
            else:
                messagebox.showinfo("Done", "Last image reached.")

    def _advance_to_next_unevaluated(self):
        """Move to next unevaluated image, or next image if all evaluated."""
        for i in range(self.current_index + 1, len(self.image_files)):
            if not self.is_evaluated(self.image_files[i]):
                self.current_index = i
                self.display_image()
                return
        # All remaining are evaluated - just go to next
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_image()

    def next_image(self):
        """Move to next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_image()

    def prev_image(self):
        """Move to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()

    def next_unevaluated(self):
        """Skip to next unevaluated image."""
        for i in range(self.current_index + 1, len(self.image_files)):
            if not self.is_evaluated(self.image_files[i]):
                self.current_index = i
                self.display_image()
                return
        messagebox.showinfo(
            "All Evaluated",
            "No unevaluated images found after this one."
        )

    def prev_unevaluated(self):
        """Skip to previous unevaluated image."""
        for i in range(self.current_index - 1, -1, -1):
            if not self.is_evaluated(self.image_files[i]):
                self.current_index = i
                self.display_image()
                return
        messagebox.showinfo(
            "All Evaluated",
            "No unevaluated images found before this one."
        )

    def jump_to_first_unevaluated(self):
        """Jump to the first unevaluated image."""
        for i in range(len(self.image_files)):
            if not self.is_evaluated(self.image_files[i]):
                self.current_index = i
                print(f"Starting at first unevaluated image: "
                      f"{i + 1}/{len(self.image_files)}")
                return
        self.current_index = 0
        print("All images evaluated! Starting at first image.")

    def jump_to_index(self):
        """Jump to a specific image index."""
        try:
            index = int(self.jump_entry.get().strip())
            if 0 <= index < len(self.image_files):
                self.current_index = index
                self.display_image()
            else:
                messagebox.showerror(
                    "Invalid Index",
                    f"Index must be 0-{len(self.image_files) - 1}"
                )
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter a valid number.")


def main():
    root = tk.Tk()

    # Start maximized (cross-platform)
    try:
        # macOS
        root.attributes('-zoomed', True)
    except tk.TclError:
        try:
            # Windows/Linux
            root.state('zoomed')
        except tk.TclError:
            # Fallback: set to screen size
            root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")

    app = SegmentEvaluator(root)

    # Bring to front
    root.lift()
    root.focus_force()

    root.mainloop()


if __name__ == "__main__":
    main()
