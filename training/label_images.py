#!/usr/bin/env python3
"""
Image labeling tool for creating training data.
Displays images and allows user to enter descriptions.
"""

import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from datetime import datetime
from pathlib import Path

# Configuration
IMAGE_FOLDER = "../artifacts/training_data/images"
LABELS_FILE = "../artifacts/training_data/labels.json"
OUTLIER_REPORT_FILE = "../artifacts/models/v1/outlier_report.json"
MAX_IMAGE_WIDTH = 1200
MAX_IMAGE_HEIGHT = 800


class ImageLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Muni Image Labeling Tool")

        # Data
        self.image_files = []
        self.all_image_files = []  # Store all images for switching modes
        self.current_index = 0
        self.labels = {}
        self.current_status = tk.StringVar(value="")  # Track selected status
        self.current_mode = "all"  # "all" or "outliers"
        self.outlier_data = {}  # Map image_path -> outlier info

        # Set up status change callback
        self.current_status.trace_add('write', self.on_status_change)

        # Load existing labels
        self.load_labels()

        # Load outlier report
        self.load_outlier_report()

        # Load image files
        self.load_image_files()

        # Setup UI
        self.setup_ui()

        # Display first unlabeled image (or first image if all are labeled)
        if self.image_files:
            self.jump_to_first_unlabeled()
            self.display_image()
        else:
            messagebox.showinfo("No Images", f"No images found in {IMAGE_FOLDER}")
            self.root.quit()

    def load_labels(self):
        """Load existing labels from JSON file."""
        if os.path.exists(LABELS_FILE):
            try:
                with open(LABELS_FILE, 'r') as f:
                    data = json.load(f)
                    self.labels = {item['image_path']: item for item in data.get('training_data', [])}
                print(f"Loaded {len(self.labels)} existing labels")
            except Exception as e:
                print(f"Error loading labels: {e}")
                self.labels = {}
        else:
            self.labels = {}

    def load_outlier_report(self):
        """Load outlier report from JSON file (read-only)."""
        if not os.path.exists(OUTLIER_REPORT_FILE):
            print(f"No outlier report found at {OUTLIER_REPORT_FILE}")
            return

        try:
            with open(OUTLIER_REPORT_FILE, 'r') as f:
                report = json.load(f)

            # Process all outlier categories
            for item in report.get('misclassified', []):
                self.outlier_data[item['image_path']] = {
                    'type': 'misclassified',
                    'explanation': f"Predicted {item['predicted_status']} but should be {item['true_status']} (confidence: {item['confidence']*100:.1f}%)"
                }

            for item in report.get('low_confidence', []):
                status_emoji = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}.get(item['true_status'], '')
                self.outlier_data[item['image_path']] = {
                    'type': 'low_confidence',
                    'explanation': f"Correctly predicted {status_emoji} {item['true_status']} but with low confidence ({item['confidence']*100:.1f}%)"
                }

            for item in report.get('high_confidence_errors', []):
                self.outlier_data[item['image_path']] = {
                    'type': 'high_confidence_error',
                    'explanation': f"High confidence error: predicted {item['predicted_status']} instead of {item['true_status']} ({item['confidence']*100:.1f}%)"
                }

            print(f"Loaded {len(self.outlier_data)} outliers from report")
        except Exception as e:
            print(f"Error loading outlier report: {e}")
            self.outlier_data = {}

    def save_labels(self):
        """Save labels to JSON file."""
        try:
            training_data = list(self.labels.values())
            # Sort by image path for consistency
            training_data.sort(key=lambda x: x['image_path'])

            data = {
                'training_data': training_data,
                'total_images': len(training_data),
                'last_updated': datetime.now().isoformat()
            }

            with open(LABELS_FILE, 'w') as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save labels: {e}")
            return False

    def is_fully_labeled(self, image_path):
        """Check if an image has both description and status."""
        if image_path not in self.labels:
            return False

        label_data = self.labels[image_path]
        has_description = bool(label_data.get('description', '').strip())
        has_status = bool(label_data.get('status', ''))

        return has_description and has_status

    def load_image_files(self):
        """Load image files based on current mode."""
        if not os.path.exists(IMAGE_FOLDER):
            return

        # Get all jpg files (both .jpg and .JPG)
        # Use a set to deduplicate on case-insensitive filesystems (Windows)
        image_dir = Path(IMAGE_FOLDER)
        jpg_files = set(image_dir.glob("*.jpg"))
        JPG_files = set(image_dir.glob("*.JPG"))
        # Use forward slashes for cross-platform consistency with labels.json
        all_files = sorted([
            str(f).replace("\\", "/") for f in jpg_files | JPG_files
        ])

        # Store all images for mode switching
        self.all_image_files = all_files

        # Filter based on mode
        if self.current_mode == "outliers":
            # Only show images that are in the outlier report and NOT reviewed
            self.image_files = [
                f for f in all_files
                if f in self.outlier_data and not self.labels.get(f, {}).get('reviewed', False)
            ]
            total_outliers = len([f for f in all_files if f in self.outlier_data])
            reviewed_count = total_outliers - len(self.image_files)
            print(f"Found {len(self.image_files)} unreviewed outlier images ({reviewed_count} already reviewed)")
        else:
            # Show all images
            self.image_files = all_files
            fully_labeled = sum(1 for img in self.image_files if self.is_fully_labeled(img))
            partially_labeled = sum(1 for img in self.image_files if img in self.labels and not self.is_fully_labeled(img))
            print(f"Found {len(self.image_files)} total images")
            print(f"Fully labeled: {fully_labeled}")
            print(f"Partially labeled: {partially_labeled}")
            print(f"Not labeled: {len(self.image_files) - fully_labeled - partially_labeled}")

    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Mode tabs
        tab_frame = ttk.Frame(main_frame)
        tab_frame.grid(row=0, column=0, pady=5, sticky=tk.W)

        self.all_tab_button = ttk.Button(
            tab_frame,
            text="All Images",
            command=lambda: self.switch_mode("all"),
            style='Accent.TButton' if self.current_mode == "all" else 'TButton'
        )
        self.all_tab_button.grid(row=0, column=0, padx=5)

        self.outliers_tab_button = ttk.Button(
            tab_frame,
            text="Outliers",
            command=lambda: self.switch_mode("outliers")
        )
        self.outliers_tab_button.grid(row=0, column=1, padx=5)

        # Progress label
        self.progress_label = ttk.Label(main_frame, text="", font=('Arial', 12, 'bold'))
        self.progress_label.grid(row=1, column=0, pady=5, sticky=tk.W)

        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=2, column=0, pady=10)

        # Outlier explanation (only visible in outliers mode)
        self.outlier_frame = ttk.LabelFrame(main_frame, text='Outlier Details', labelanchor='w', padding="5")
        self.outlier_frame.grid(row=3, column=0, pady=5)
        self.outlier_frame.grid_remove()  # Hidden by default

        self.outlier_explanation = ttk.Label(
            self.outlier_frame,
            text="",
            wraplength=1200  # Allow wrapping but keep it to one line visually
        )
        self.outlier_explanation.pack(fill=tk.X, expand=True)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, pady=10)

        # Column 0: Index
        index_frame = ttk.LabelFrame(button_frame, text='Index')
        index_frame.grid(row=0, column=0, rowspan=3, padx=5, pady=5, sticky=tk.W)

        self.jump_entry = ttk.Entry(index_frame, width=10)
        self.jump_entry.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.jump_button = ttk.Button(index_frame, text="Jump To", command=self.jump_to_index)
        self.jump_button.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        # Row 1: Main navigation and save buttons
        self.green_radio = tk.Radiobutton(
            button_frame,
            text="🟢 Green (Normal)",
            variable=self.current_status,
            value="green",
            font=('Arial', 11),
            bg='#90EE90',
            activebackground='#90EE90',
            selectcolor='#90EE90'
        )
        self.green_radio.grid(row=0, column=3, padx=10)

        # Row 2: Yellow and description
        self.yellow_radio = tk.Radiobutton(
            button_frame,
            text="🟡 Yellow (Delays)",
            variable=self.current_status,
            value="yellow",
            font=('Arial', 11),
            bg='#FFFFE0',
            activebackground='#FFFFE0',
            selectcolor='#FFFFE0'
        )
        self.yellow_radio.grid(row=1, column=3, padx=10)

        # Text entry for description
        self.description_text = tk.Entry(
            button_frame,
            width=55,
            font=('Arial', 10)
        )
        self.description_text.grid(row=1, column=4, columnspan=3, pady=5)

        # Row 3: Red button
        self.red_radio = tk.Radiobutton(
            button_frame,
            text="🔴 Red (Issues)",
            variable=self.current_status,
            value="red",
            font=('Arial', 11),
            bg='#FFB6C6',
            activebackground='#FFB6C6',
            selectcolor='#FFB6C6'
        )
        self.red_radio.grid(row=2, column=3, padx=10)

        # Row 4: Navigation buttons
        self.prev_unlabeled_button = ttk.Button(button_frame, text="⏮ Previous Unlabeled", command=self.prev_unlabeled)
        self.prev_unlabeled_button.grid(row=3, column=1, padx=5, pady=5, sticky="E")

        self.prev_button = ttk.Button(button_frame, text="← Previous", command=self.prev_image)
        self.prev_button.grid(row=3, column=2, padx=5, sticky="E")

        self.clear_status_button = ttk.Button(
            button_frame,
            text="Clear Status",
            command=lambda: self.current_status.set("")
        )
        self.clear_status_button.grid(row=3, column=3, padx=10)

        self.save_button = ttk.Button(button_frame, text="Next →", command=self.save_and_next, style='Accent.TButton')
        self.save_button.grid(row=3, column=4, padx=5, sticky="W")

        self.next_button = ttk.Button(button_frame, text="Skip → →", command=self.next_image)
        self.next_button.grid(row=3, column=5, padx=5, sticky="W")

        self.next_unlabeled_button = ttk.Button(button_frame, text="Next Unlabeled ⏭", command=self.next_unlabeled)
        self.next_unlabeled_button.grid(row=3, column=6, padx=5, pady=5, sticky="W")

        # Column 7: Special
        special_frame = ttk.LabelFrame(button_frame, text='Special')
        special_frame.grid(row=0, column=7, rowspan=3, padx=5, pady=5, sticky=tk.W)

        self.save_all_button = ttk.Button(special_frame, text="💾 Save All", command=self.save_labels)
        self.save_all_button.grid(row=0, column=0, padx=20)

        # Delete image
        self.delete_button = ttk.Button(special_frame, text="🗑️ Delete Image", command=self.delete_current_image)
        self.delete_button.grid(row=1, column=0, padx=20, pady=5)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="", foreground="gray")
        self.status_label.grid(row=9, column=0, pady=5)

        # Keyboard shortcuts
        keyboard_1 = ttk.Label(main_frame, text="Save & Next: Ctrl+Enter | Prev or Skip: Ctrl+←/→ | Jump to Index: Ctrl+G")
        keyboard_1.grid(row=10, column=0, pady=5)
        keyboard_2 = ttk.Label(main_frame, text="Green: 1 | Yellow: 2 | Red: 3 | Clear: 0")
        keyboard_2.grid(row=11, column=0, pady=5)

        # Keyboard shortcuts - bind to root for global access
        self.root.bind('<Control-s>', lambda e: self.save_and_next())
        self.root.bind('<Control-Return>', lambda e: self.save_and_next())
        self.root.bind('<Control-Right>', lambda e: self.next_image())
        self.root.bind('<Control-Left>', lambda e: self.prev_image())
        self.root.bind('<Control-Shift-Right>', lambda e: self.next_unlabeled())
        self.root.bind('<Control-Shift-Left>', lambda e: self.prev_unlabeled())

        # Number key shortcuts for status - global bindings
        self.root.bind('1', self.set_green_status)
        self.root.bind('2', self.set_yellow_status)
        self.root.bind('3', self.set_red_status)
        self.root.bind('0', lambda e: self.current_status.set(""))

        # Delete key shortcut
        self.root.bind('<Delete>', lambda e: self.delete_current_image())
        self.root.bind('<Control-d>', lambda e: self.delete_current_image())

        # Jump to index shortcuts
        self.root.bind('<Control-g>', lambda e: self.jump_entry.focus())
        self.jump_entry.bind('<Return>', lambda e: self.jump_to_index())

    def set_green_status(self, event=None):
        """Set status to green and auto-fill 'Normal' if text is empty."""
        self.current_status.set("green")

    def set_yellow_status(self, event=None):
        """Set status to yellow."""
        self.current_status.set("yellow")

    def set_red_status(self, event=None):
        """Set status to red and auto-fill 'Offline' if text is empty."""
        self.current_status.set("red")

    def on_status_change(self, *args):
        """Called when status changes - auto-populate text for green/red if empty."""
        current_text = self.description_text.get().strip()
        new_status = self.current_status.get()

        # Only auto-populate if text box is empty
        if not current_text:
            if new_status == "green":
                self.description_text.delete(0, tk.END)
                self.description_text.insert(0, "Normal")
            elif new_status == "red":
                self.description_text.delete(0, tk.END)
                self.description_text.insert(0, "Offline")

    def display_image(self):
        """Display the current image and its label if exists."""
        if not self.image_files or self.current_index >= len(self.image_files):
            return

        image_path = self.image_files[self.current_index]

        # Update progress (count fully labeled images)
        unlabeled_count = len(self.image_files) - sum(1 for f in self.image_files if self.is_fully_labeled(f))
        self.progress_label.config(
            text=f"Image {self.current_index + 1} of {len(self.image_files)} | {unlabeled_count} unlabeled"
        )

        # Load and display image
        try:
            img = Image.open(image_path)

            # Resize if too large
            img.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference

            # Show/hide outlier explanation based on mode
            if self.current_mode == "outliers" and image_path in self.outlier_data:
                self.outlier_frame.grid()
                outlier_info = self.outlier_data[image_path]
                self.outlier_explanation.config(text=outlier_info['explanation'])
            else:
                self.outlier_frame.grid_remove()

            # Load existing description and status if available
            if image_path in self.labels:
                description = self.labels[image_path].get('description', '')
                status = self.labels[image_path].get('status', '')

                self.description_text.delete(0, tk.END)
                self.description_text.insert(0, description)
                self.current_status.set(status)

                # Check if fully labeled (has both description and status)
                has_description = bool(description.strip())
                has_status = bool(status)
                is_reviewed = self.labels[image_path].get('reviewed', False)

                if has_description and has_status:
                    status_text = f"✓ Labeled (last updated: {self.labels[image_path].get('labeled_at', 'unknown')})"
                    if is_reviewed:
                        status_text += f" | ✓ Reviewed: {self.labels[image_path].get('reviewed_at', 'unknown')}"
                    self.status_label.config(
                        text=status_text,
                        foreground="green"
                    )
                elif has_description or has_status:
                    self.status_label.config(
                        text=f"⚠ Partially labeled - missing {'status' if not has_status else 'description'}",
                        foreground="orange"
                    )
                else:
                    self.status_label.config(text="⚠ Not yet labeled", foreground="orange")
            else:
                self.description_text.delete(0, tk.END)
                self.current_status.set("")
                self.status_label.config(text="⚠ Not yet labeled", foreground="orange")

            # Update button states
            self.prev_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
            self.next_button.config(state=tk.NORMAL if self.current_index < len(self.image_files) - 1 else tk.DISABLED)

            # Update skip button states (check for fully labeled images)
            has_unlabeled_after = any(
                not self.is_fully_labeled(self.image_files[i])
                for i in range(self.current_index + 1, len(self.image_files))
            )
            has_unlabeled_before = any(
                not self.is_fully_labeled(self.image_files[i])
                for i in range(0, self.current_index)
            )
            self.next_unlabeled_button.config(state=tk.NORMAL if has_unlabeled_after else tk.DISABLED)
            self.prev_unlabeled_button.config(state=tk.NORMAL if has_unlabeled_before else tk.DISABLED)

            # Update jump index
            self.set_index(self.current_index)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def save_current_label(self):
        """Save the current image's label."""
        if not self.image_files:
            return False

        image_path = self.image_files[self.current_index]
        description = self.description_text.get().strip()
        status = self.current_status.get()

        if not description and not status:
            response = messagebox.askyesno(
                "Empty Label",
                "Both description and status are empty. Do you want to save anyway?"
            )
            if not response:
                return False

        # Create or update label
        label_data = {
            'image_path': image_path,
            'description': description,
            'status': status,
            'labeled_at': datetime.now().isoformat(),
            'image_size': os.path.getsize(image_path)
        }

        # Mark as reviewed if we're in outliers mode
        if self.current_mode == "outliers":
            label_data['reviewed'] = True
            label_data['reviewed_at'] = datetime.now().isoformat()

        # Preserve existing reviewed status if not in outliers mode
        if image_path in self.labels and 'reviewed' in self.labels[image_path]:
            label_data['reviewed'] = self.labels[image_path]['reviewed']
            if 'reviewed_at' in self.labels[image_path]:
                label_data['reviewed_at'] = self.labels[image_path]['reviewed_at']

        self.labels[image_path] = label_data

        return self.save_labels()

    def save_and_next(self):
        """Save current label and move to next image."""
        if self.save_current_label():
            self.status_label.config(text="✓ Saved successfully!", foreground="green")
            if self.current_index < len(self.image_files) - 1:
                self.next_image()
            else:
                # Check if all images are fully labeled
                labeled_count = sum(1 for f in self.image_files if self.is_fully_labeled(f))
                if labeled_count == len(self.image_files):
                    messagebox.showinfo("Complete", "All images have been fully labeled!")
                else:
                    messagebox.showinfo("Last Image", "This is the last image.")

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

    def jump_to_index(self):
        """Jump to a specific image by index."""
        try:
            index_str = self.jump_entry.get().strip()
            if not index_str:
                messagebox.showwarning("Invalid Index", "Please enter an index number.")
                return

            index = int(index_str)

            # Validate index range
            if index < 0 or index >= len(self.image_files):
                messagebox.showerror(
                    "Invalid Index",
                    f"Index must be between 0 and {len(self.image_files) - 1}.\n"
                    f"Total images: {len(self.image_files)}"
                )
                return

            # Jump to the index
            self.current_index = index
            self.display_image()
            self.status_label.config(text=f"Jumped to index {index}", foreground="blue")

            # Clear the entry field
            self.set_index(index)

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")

    def set_index(self, index):
        """Set the index entry to a numerical value."""
        self.jump_entry.delete(0, tk.END)
        self.jump_entry.insert(0, str(index))

    def next_unlabeled(self):
        """Skip to next unlabeled image (missing description and/or status)."""
        # Search from current position + 1 to end
        for i in range(self.current_index + 1, len(self.image_files)):
            if not self.is_fully_labeled(self.image_files[i]):
                self.current_index = i
                self.display_image()
                return

        # No unlabeled images found ahead
        messagebox.showinfo("No Unlabeled Images", "No unlabeled images found after this one.")

    def prev_unlabeled(self):
        """Skip to previous unlabeled image (missing description and/or status)."""
        # Search from current position - 1 to beginning
        for i in range(self.current_index - 1, -1, -1):
            if not self.is_fully_labeled(self.image_files[i]):
                self.current_index = i
                self.display_image()
                return

        # No unlabeled images found before
        messagebox.showinfo("No Unlabeled Images", "No unlabeled images found before this one.")

    def jump_to_first_unlabeled(self):
        """Jump to the first unlabeled image on startup."""
        # Search from the beginning for first unlabeled image
        for i in range(len(self.image_files)):
            if not self.is_fully_labeled(self.image_files[i]):
                self.current_index = i
                print(f"Starting at first unlabeled image: {i + 1}/{len(self.image_files)}")
                return

        # All images are labeled - start at beginning
        self.current_index = 0
        print("All images are labeled! Starting at first image.")

    def delete_current_image(self):
        """Delete the current image file after confirmation."""
        if not self.image_files or self.current_index >= len(self.image_files):
            return

        image_path = self.image_files[self.current_index]
        image_filename = os.path.basename(image_path)

        # Show confirmation dialog
        response = messagebox.askyesno(
            "Delete Image",
            f"Are you sure you want to delete this image?\n\n{image_filename}\n\nThis action cannot be undone.",
            icon='warning'
        )

        if not response:
            return

        try:
            # Delete the file from disk
            os.remove(image_path)
            print(f"Deleted: {image_path}")

            # Remove from labels if present
            if image_path in self.labels:
                del self.labels[image_path]
                self.save_labels()

            # Remove from image_files list
            self.image_files.pop(self.current_index)

            # Check if there are any images left
            if not self.image_files:
                messagebox.showinfo("No Images", "No more images to label.")
                self.root.quit()
                return

            # Adjust index if we deleted the last image
            if self.current_index >= len(self.image_files):
                self.current_index = len(self.image_files) - 1

            # Display the next (or previous) image
            self.display_image()
            self.status_label.config(text="✓ Image deleted successfully", foreground="red")

        except Exception as e:
            messagebox.showerror("Delete Error", f"Failed to delete image: {e}")

    def switch_mode(self, mode):
        """Switch between 'all' and 'outliers' modes."""
        if mode == self.current_mode:
            return  # Already in this mode

        self.current_mode = mode

        # Update button styles
        if mode == "all":
            self.all_tab_button.configure(style='Accent.TButton')
            self.outliers_tab_button.configure(style='TButton')
        else:
            self.all_tab_button.configure(style='TButton')
            self.outliers_tab_button.configure(style='Accent.TButton')

        # Reload image files for the new mode
        self.load_image_files()

        # Reset to first image
        self.current_index = 0
        if self.image_files:
            self.jump_to_first_unlabeled()
            self.display_image()
        else:
            if mode == "outliers":
                messagebox.showinfo("No Outliers", "No outlier images found. Make sure the outlier report exists.")
            else:
                messagebox.showinfo("No Images", f"No images found in {IMAGE_FOLDER}")


def main():
    root = tk.Tk()
    app = ImageLabeler(root)

    # Set window size
    root.geometry("1400x1000")

    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # Hack: Make window appear on top
    root.overrideredirect(True)
    root.overrideredirect(False)

    root.mainloop()


if __name__ == "__main__":
    main()
