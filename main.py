# imports
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont, ImageColor, ImageTk, ImageChops
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
from io import BytesIO
import os
import json
import ctypes
import platform

# High DPI Awareness (Windows)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Helper functions for dialogs using Tkinter directly
def get_file_path(title, save_as=False, file_types=None, default_extension=None):
    if tk._default_root is None:
        root = tk.Tk()
        root.withdraw()
        should_destroy = True
    else:
        root = tk._default_root
        should_destroy = False
    
    # Default file types if none provided
    if not file_types:
        file_types = [("All Files", "*.*")]

    if save_as:
        path = filedialog.asksaveasfilename(title=title, filetypes=file_types, defaultextension=default_extension)
    else:
        path = filedialog.askopenfilename(title=title, filetypes=file_types)
    
    if should_destroy:
        root.destroy()
    return path

def get_folder_path(title):
    if tk._default_root is None:
        root = tk.Tk()
        root.withdraw()
        should_destroy = True
    else:
        root = tk._default_root
        should_destroy = False
        
    path = filedialog.askdirectory(title=title)
    
    if should_destroy:
        root.destroy()
    return path

def show_message(title, message, is_error=False):
    if is_error:
        messagebox.showerror(title, message)
    else:
        messagebox.showinfo(title, message)

# function that applies effects to an image based on values
def apply_effects(original, values):
    image = original.copy()

    # Extract values
    blur = values.get('-BLUR-', 0)
    contrast = values.get('-CONTRAST-', 1.0)
    brightness = values.get('-BRIGHTNESS-', 1.0)
    color = values.get('-COLOR-', 1.0)
    sharpness = values.get('-SHARPNESS-', 1.0)
    edge_enhance = values.get('-EDGE-', False)
    detail = values.get('-DETAIL-', False)
    emboss = values.get('-EMBOSS-', False)
    contour = values.get('-CONTOUR-', False)
    flipx = values.get('-FLIPX-', False)
    flipy = values.get('-FLIPY-', False)
    rotation = values.get('-ROTATION-', 0)
    scale = values.get('-SCALE-', 100)
    crop_l = values.get('-CROP_L-', 0)
    crop_r = values.get('-CROP_R-', 0)
    crop_t = values.get('-CROP_T-', 0)
    crop_b = values.get('-CROP_B-', 0)
    invert = values.get('-INVERT-', False)
    sepia = values.get('-SEPIA-', 0)
    posterize = values.get('-POSTERIZE-', 8)
    solarize = values.get('-SOLARIZE-', 255)
    r_factor = values.get('-R_FACTOR-', 1.0)
    g_factor = values.get('-G_FACTOR-', 1.0)
    b_factor = values.get('-B_FACTOR-', 1.0)
    temperature = values.get('-TEMPERATURE-', 0)
    grayscale = values.get('-GRAYSCALE-', False)
    threshold = values.get('-THRESHOLD-', 255)
    pixelate = values.get('-PIXELATE-', 1)
    gamma = values.get('-GAMMA-', 1.0)
    hue = values.get('-HUE-', 0)
    vignette = values.get('-VIGNETTE-', 0)
    
    # New features v0.11
    auto_contrast = values.get('-AUTO_CONTRAST-', False)
    equalize = values.get('-EQUALIZE-', False)
    border_size = int(values.get('-BORDER_SIZE-', 0))
    border_color_hex = values.get('-BORDER_COLOR-', '#FFFFFF')

    # New features v0.12 (Merged)
    noise = int(values.get('-NOISE-', 0))
    unsharp = values.get('-UNSHARP-', 0)
    chromatic = int(values.get('-CHROMATIC-', 0))
    scanline = int(values.get('-SCANLINE-', 0))
    
    # Watermark values
    watermark_path = values.get('-WATERMARK_PATH-', '')
    watermark_opacity = int(values.get('-WATERMARK_OPACITY-', 255))
    watermark_scale = int(values.get('-WATERMARK_SCALE-', 100))
    watermark_x = values.get('-WATERMARK_X-', 50)
    watermark_y = values.get('-WATERMARK_Y-', 50)

    # Text values
    text_content = values.get('-TEXT_CONTENT-', '')
    text_size = int(values.get('-TEXT_SIZE-', 20))
    text_opacity = int(values.get('-TEXT_OPACITY-', 255))
    text_color_hex = values.get('-TEXT_COLOR-', '#FFFFFF')
    text_x_percent = values.get('-TEXT_X-', 50)
    text_y_percent = values.get('-TEXT_Y-', 50)

    # apply rotation
    if rotation != 0:
        image = image.rotate(rotation, expand=True)

    # apply flip
    if flipx:
        image = ImageOps.mirror(image)
    if flipy:
        image = ImageOps.flip(image)

    # apply crop
    if crop_l > 0 or crop_r > 0 or crop_t > 0 or crop_b > 0:
        w, h = image.size
        left = int(w * (crop_l / 100))
        top = int(h * (crop_t / 100))
        right = int(w * (1 - (crop_r / 100)))
        bottom = int(h * (1 - (crop_b / 100)))
        # Ensure valid crop box
        if right > left and bottom > top:
            image = image.crop((left, top, right, bottom))

    # apply scale
    if scale != 100:
        w, h = image.size
        new_w = int(w * (scale / 100))
        new_h = int(h * (scale / 100))
        # Use Resampling.LANCZOS if available, else fallback (handled by PIL usually)
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        image = image.resize((new_w, new_h), resample)

    # apply pixelate
    if pixelate > 1:
        w, h = image.size
        # Resize down
        small_w = max(1, int(w / pixelate))
        small_h = max(1, int(h / pixelate))
        image = image.resize((small_w, small_h), resample=Image.NEAREST)
        # Resize up
        image = image.resize((w, h), resample=Image.NEAREST)

    # apply noise reduction (Median Filter)
    if noise > 0:
        # Size must be odd: 1->3, 2->5, 3->7, etc.
        size = (noise * 2) + 1
        image = image.filter(ImageFilter.MedianFilter(size))

    # apply chromatic aberration
    if chromatic != 0:
        if image.mode != 'RGB' and image.mode != 'RGBA':
            image = image.convert('RGB')
        
        # Split channels
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
        else:
            r, g, b = image.split()
            a = None
            
        # Shift channels
        r = ImageChops.offset(r, chromatic, 0)
        b = ImageChops.offset(b, -chromatic, 0)
        
        # Merge back
        if a:
            image = Image.merge('RGBA', (r, g, b, a))
        else:
            image = Image.merge('RGB', (r, g, b))

    # apply filters
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    if edge_enhance:
        image = image.filter(ImageFilter.EDGE_ENHANCE())
    if detail:
        image = image.filter(ImageFilter.DETAIL())
    if emboss:
        image = image.filter(ImageFilter.EMBOSS())
    if contour:
        image = image.filter(ImageFilter.CONTOUR())

    # apply posterize
    if posterize < 8:
        # posterize requires an integer from 1 to 8
        # we ensure it is at least 1
        bits = max(1, int(posterize))
        if image.mode == 'RGBA':
            # posterize might lose alpha or not handle it well, let's separate alpha
            r, g, b, a = image.split()
            rgb = Image.merge('RGB', (r, g, b))
            rgb = ImageOps.posterize(rgb, bits)
            r, g, b = rgb.split()
            image = Image.merge('RGBA', (r, g, b, a))
        else:
            image = ImageOps.posterize(image, bits)

    # apply solarize
    if solarize < 255:
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rgb = Image.merge('RGB', (r, g, b))
            rgb = ImageOps.solarize(rgb, int(solarize))
            r, g, b = rgb.split()
            image = Image.merge('RGBA', (r, g, b, a))
        else:
            image = ImageOps.solarize(image, int(solarize))

    # apply color effects
    if invert:
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rgb_image = Image.merge('RGB', (r, g, b))
            inverted_image = ImageOps.invert(rgb_image)
            r2, g2, b2 = inverted_image.split()
            image = Image.merge('RGBA', (r2, g2, b2, a))
        else:
            image = ImageOps.invert(image)

    # apply sepia (now a slider)
    if sepia > 0:
        # Create sepia version
        if image.mode == 'RGBA':
            alpha = image.split()[3]
            gray = image.convert('L')
            sepia_img = ImageOps.colorize(gray, '#704214', '#C0C080')
            sepia_img.putalpha(alpha)
        else:
            gray = image.convert('L')
            sepia_img = ImageOps.colorize(gray, '#704214', '#C0C080')
        
        # Blend with original
        if image.mode != sepia_img.mode:
            sepia_img = sepia_img.convert(image.mode)
        image = Image.blend(image, sepia_img, sepia / 100.0)

    if grayscale:
        image = ImageOps.grayscale(image)
        # Convert back to RGB/RGBA so other filters work if needed, or keep as L
        # Usually for display it's fine, but subsequent operations might expect RGB
        if original.mode == 'RGBA':
             image = image.convert('RGBA')
        else:
             image = image.convert('RGB')

    # apply auto contrast
    if auto_contrast:
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rgb = Image.merge('RGB', (r, g, b))
            rgb = ImageOps.autocontrast(rgb)
            r, g, b = rgb.split()
            image = Image.merge('RGBA', (r, g, b, a))
        else:
            image = ImageOps.autocontrast(image)

    # apply equalize
    if equalize:
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rgb = Image.merge('RGB', (r, g, b))
            rgb = ImageOps.equalize(rgb)
            r, g, b = rgb.split()
            image = Image.merge('RGBA', (r, g, b, a))
        else:
            image = ImageOps.equalize(image)

    # apply threshold (black and white)
    if threshold < 255:
        # Convert to grayscale first
        gray_img = ImageOps.grayscale(image)
        # Apply threshold
        image = gray_img.point(lambda p: 255 if p > threshold else 0)
        if original.mode == 'RGBA':
             image = image.convert('RGBA')
        else:
             image = image.convert('RGB')

    # apply hue
    if hue != 0:
        # save alpha
        alpha = None
        if image.mode == 'RGBA':
            alpha = image.split()[3]
            img_rgb = image.convert('RGB')
        else:
            img_rgb = image.convert('RGB')
        
        # convert to HSV
        hsv_image = img_rgb.convert('HSV')
        h, s, v = hsv_image.split()
        
        # apply hue shift
        # hue is -180 to 180
        # 255 is 360 degrees
        shift = int((hue / 360.0) * 255)
        h = h.point(lambda i: (i + shift) % 256)
        
        hsv_image = Image.merge('HSV', (h, s, v))
        image = hsv_image.convert('RGB')
        
        if alpha:
            image.putalpha(alpha)

    # apply RGB balance and Temperature
    if r_factor != 1.0 or g_factor != 1.0 or b_factor != 1.0 or temperature != 0:
        if image.mode == 'L':
            image = image.convert('RGB')
        
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
        else:
            r, g, b = image.split()
            a = None
        
        # Calculate temperature factors
        temp_r = 0
        temp_b = 0
        if temperature != 0:
            temp_r = (temperature / 100.0) * 0.2
            temp_b = - (temperature / 100.0) * 0.2

        # Apply factors
        final_r_factor = r_factor + temp_r
        final_g_factor = g_factor
        final_b_factor = b_factor + temp_b

        if final_r_factor != 1.0:
            r = r.point(lambda i: i * final_r_factor)
        if final_g_factor != 1.0:
            g = g.point(lambda i: i * final_g_factor)
        if final_b_factor != 1.0:
            b = b.point(lambda i: i * final_b_factor)
            
        if a:
            image = Image.merge('RGBA', (r, g, b, a))
        else:
            image = Image.merge('RGB', (r, g, b))

    # apply enhancements
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    if color != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(color)
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
        
    # apply unsharp mask
    if unsharp > 0:
        image = image.filter(ImageFilter.UnsharpMask(radius=unsharp, percent=150, threshold=3))
        
    # apply gamma
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        # build a lookup table
        table = [int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)]
        if image.mode == 'L':
            image = image.point(table)
        elif image.mode == 'RGBA':
            r, g, b, a = image.split()
            r = r.point(table)
            g = g.point(table)
            b = b.point(table)
            image = Image.merge('RGBA', (r, g, b, a))
        else:
            # RGB
            image = image.point(table * 3)

    # apply scanlines
    if scanline > 0:
        w, h = image.size
        # Create a pattern
        overlay = Image.new('RGBA', (w, h), (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        # Draw lines every 4 pixels
        for y in range(0, h, 4):
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, scanline))
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        image = Image.alpha_composite(image, overlay)

    # apply vignette
    if vignette > 0:
        w, h = image.size
        # Create mask: White center, Black corners
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        # Draw white ellipse
        draw.ellipse((0, 0, w, h), fill=255)
        # Blur the mask to create gradient
        blur_radius = max(w, h) / 10
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Invert mask: Black center, White corners
        mask_inv = ImageOps.invert(mask)
        
        # Adjust intensity
        # We want the corners to be black with opacity = vignette %
        # So we scale the mask values
        opacity = vignette / 100.0
        mask_inv = mask_inv.point(lambda i: int(i * opacity))
        
        # Paste black layer using mask
        black_layer = Image.new('RGB', (w, h), (0, 0, 0))
        image.paste(black_layer, (0, 0), mask_inv)

    # apply border
    if border_size > 0:
        try:
            r, g, b = ImageColor.getrgb(border_color_hex)
        except:
            r, g, b = (255, 255, 255)
            
        if image.mode == 'RGBA':
            border_color = (r, g, b, 255)
        else:
            border_color = (r, g, b)
            
        image = ImageOps.expand(image, border=border_size, fill=border_color)

    # apply watermark
    if watermark_path and os.path.exists(watermark_path):
        try:
            wm = Image.open(watermark_path).convert("RGBA")
            
            # Scale watermark
            if watermark_scale != 100:
                wm_w, wm_h = wm.size
                new_wm_w = int(wm_w * (watermark_scale / 100))
                new_wm_h = int(wm_h * (watermark_scale / 100))
                wm = wm.resize((new_wm_w, new_wm_h), Image.Resampling.LANCZOS)
            
            # Opacity
            if watermark_opacity < 255:
                # Adjust alpha channel
                r, g, b, a = wm.split()
                a = a.point(lambda i: int(i * (watermark_opacity / 255)))
                wm = Image.merge('RGBA', (r, g, b, a))
            
            # Position
            w, h = image.size
            wm_w, wm_h = wm.size
            
            # Center based on percentage
            x = int((w * (watermark_x / 100)) - (wm_w / 2))
            y = int((h * (watermark_y / 100)) - (wm_h / 2))
            
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Paste (using alpha composite for transparency)
            # Create a layer for watermark
            wm_layer = Image.new('RGBA', image.size, (0,0,0,0))
            wm_layer.paste(wm, (x, y))
            image = Image.alpha_composite(image, wm_layer)
            
        except Exception as e:
            print(f"Error applying watermark: {e}")

    # apply text overlay
    if text_content:
        try:
            # Try to load arial, fallback to default
            font = ImageFont.truetype("arial.ttf", text_size)
        except IOError:
            try:
                font = ImageFont.truetype("Arial.ttf", text_size)
            except IOError:
                font = ImageFont.load_default()
        
        # Create a transparent layer for text
        txt_layer = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)
        
        # Parse color
        try:
            r, g, b = ImageColor.getrgb(text_color_hex)
            text_color = (r, g, b, text_opacity)
        except:
            text_color = (255, 255, 255, text_opacity)
            
        # Calculate position
        w, h = image.size
        
        # Get text size to center it if needed
        try:
            bbox = draw.textbbox((0, 0), text_content, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older Pillow versions
            text_w, text_h = draw.textsize(text_content, font=font)
        
        x = int((w * (text_x_percent / 100)) - (text_w / 2))
        y = int((h * (text_y_percent / 100)) - (text_h / 2))
        
        draw.text((x, y), text_content, font=font, fill=text_color)
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        image = Image.alpha_composite(image, txt_layer)
        
    return image

# Global state
gui_vars = {}
defaults = {}
original_image = None
current_image = None
history = []
history_index = -1
prev_values = {}
image_label = None

def get_values():
    return {k: v.get() for k, v in gui_vars.items()}

def update_image(original, values=None):
    global current_image
    if values is None:
        values = get_values()
    
    try:
        current_image = apply_effects(original, values)
        
        # Resize for display
        display_image = current_image.copy()
        
        # Get display area size? For now fixed max size
        max_w, max_h = 800, 800
        display_image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        
        tk_img = ImageTk.PhotoImage(display_image)
        image_label.configure(image=tk_img)
        image_label.image = tk_img
    except Exception as e:
        print(f"Error updating image: {e}")

def on_change(*args):
    global history, history_index, prev_values
    values = get_values()
    update_image(original_image, values)

def save_history(event=None):
    global history, history_index, prev_values
    values = get_values()
    if values != prev_values:
        if history_index < len(history) - 1:
            history = history[:history_index+1]
        history.append(values)
        history_index += 1
        if len(history) > 50:
            history.pop(0)
            history_index -= 1
        prev_values = values.copy()

def reset_controls():
    for k, v in defaults.items():
        if k in gui_vars:
            gui_vars[k].set(v)
    save_history()
    update_image(original_image)

def undo():
    global history_index, prev_values
    if history_index > 0:
        history_index -= 1
        restore_values = history[history_index]
        set_values(restore_values)
        prev_values = restore_values.copy()
        update_image(original_image, restore_values)

def redo():
    global history_index, prev_values
    if history_index < len(history) - 1:
        history_index += 1
        restore_values = history[history_index]
        set_values(restore_values)
        prev_values = restore_values.copy()
        update_image(original_image, restore_values)

def set_values(values):
    for k, v in values.items():
        if k in gui_vars:
            gui_vars[k].set(v)

def save_preset():
    preset_path = get_file_path('Save Preset', save_as=True, file_types=[("JSON", "*.json")], default_extension=".json")
    if preset_path:
        data = get_values()
        try:
            with open(preset_path, 'w') as f:
                json.dump(data, f, indent=4)
            show_message('Success', 'Preset saved successfully!')
        except Exception as e:
            show_message('Error', f'Error saving preset: {e}', is_error=True)

def load_preset():
    preset_path = get_file_path('Load Preset', file_types=[("JSON", "*.json")])
    if preset_path:
        try:
            with open(preset_path, 'r') as f:
                data = json.load(f)
            set_values(data)
            save_history()
            update_image(original_image)
        except Exception as e:
            show_message('Error', f'Error loading preset: {e}', is_error=True)

def batch_process():
    source_folder = get_folder_path('Select Source Folder')
    if source_folder:
        dest_folder = get_folder_path('Select Destination Folder')
        if dest_folder:
            values = get_values()
            try:
                files = os.listdir(source_folder)
                count = 0
                for filename in files:
                    lower_name = filename.lower()
                    if lower_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        try:
                            file_path = os.path.join(source_folder, filename)
                            img = Image.open(file_path)
                            if img.mode in ('P', 'CMYK', 'HSV'):
                                img = img.convert('RGBA')
                            
                            processed_img = apply_effects(img, values)
                            
                            save_path = os.path.join(dest_folder, filename)
                            if lower_name.endswith(('.jpg', '.jpeg')):
                                 if processed_img.mode in ('RGBA', 'LA'):
                                    background = Image.new(processed_img.mode[:-1], processed_img.size, (255, 255, 255))
                                    background.paste(processed_img, processed_img.split()[-1])
                                    processed_img = background
                                 processed_img.save(save_path, quality=95)
                            else:
                                processed_img.save(save_path)
                            
                            count += 1
                        except Exception as e:
                            print(f"Failed to process {filename}: {e}")
                
                show_message('Success', f'Batch processing complete! Processed {count} images.')
            except Exception as e:
                show_message('Error', f'Error reading folder: {e}', is_error=True)

def save_image():
    save_path = get_file_path('Save', save_as=True, file_types=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")], default_extension=".png")
    if save_path:
        filename, file_extension = os.path.splitext(save_path)
        if not file_extension:
            save_path += '.png'
            file_extension = '.png'
        
        if file_extension.lower() in ['.jpg', '.jpeg']:
            if current_image.mode in ('RGBA', 'LA'):
                background = Image.new(current_image.mode[:-1], current_image.size, (255, 255, 255))
                background.paste(current_image, current_image.split()[-1])
                image_to_save = background
            else:
                image_to_save = current_image
            image_to_save.save(save_path, quality=95)
        else:
            current_image.save(save_path)

def rotate_m90():
    curr = gui_vars['-ROTATION-'].get()
    new_val = (curr - 90) % 360
    gui_vars['-ROTATION-'].set(new_val)
    save_history()
    update_image(original_image)

def rotate_p90():
    curr = gui_vars['-ROTATION-'].get()
    new_val = (curr + 90) % 360
    gui_vars['-ROTATION-'].set(new_val)
    save_history()
    update_image(original_image)

def pick_color(key):
    try:
        curr = gui_vars[key].get()
        color = colorchooser.askcolor(color=curr)[1]
    except:
        color = colorchooser.askcolor()[1]
        
    if color:
        gui_vars[key].set(color)
        save_history()
        update_image(original_image)

def pick_watermark():
    path = get_file_path("Select Watermark", file_types=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
    if path:
        gui_vars['-WATERMARK_PATH-'].set(path)
        save_history()
        update_image(original_image)

# UI Construction Helpers
def add_slider(parent, label, key, from_, to, default, resolution=1):
    frame = ttk.LabelFrame(parent, text=label)
    frame.pack(fill='x', padx=5, pady=2)
    
    var = tk.DoubleVar(value=default)
    gui_vars[key] = var
    defaults[key] = default
    
    scale = tk.Scale(frame, from_=from_, to=to, orient='horizontal', variable=var, resolution=resolution, command=on_change)
    scale.pack(fill='x', padx=5, pady=2)
    
    # Bind release for history
    scale.bind("<ButtonRelease-1>", save_history)
    
    return scale

def add_checkbox(parent, label, key, default=False):
    var = tk.BooleanVar(value=default)
    gui_vars[key] = var
    defaults[key] = default
    
    chk = ttk.Checkbutton(parent, text=label, variable=var, command=lambda: [on_change(), save_history()])
    return chk

# Main execution
if __name__ == "__main__":
    # Hide root window for initial dialog
    root = tk.Tk()
    root.withdraw()
    
    image_path = get_file_path('Open')
    if not image_path:
        root.destroy()
        exit()
        
    original_image = Image.open(image_path)
    if original_image.mode in ('P', 'CMYK', 'HSV'):
        original_image = original_image.convert('RGBA')
    
    # Show root window
    root.deiconify()
    root.title("IMAGE EDITOR")
    
    # Layout
    main_paned = tk.PanedWindow(root, orient=tk.HORIZONTAL)
    main_paned.pack(fill=tk.BOTH, expand=True)
    
    control_frame = ttk.Frame(main_paned, width=350)
    image_frame = ttk.Frame(main_paned)
    
    main_paned.add(control_frame)
    main_paned.add(image_frame)
    
    # Tabs
    notebook = ttk.Notebook(control_frame)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    # Effects Tab
    tab_effects = ttk.Frame(notebook)
    notebook.add(tab_effects, text='Effects')
    add_slider(tab_effects, 'Blur', '-BLUR-', 0, 10, 0)
    add_slider(tab_effects, 'Noise Reduction', '-NOISE-', 0, 5, 0)
    add_slider(tab_effects, 'Pixelate', '-PIXELATE-', 1, 50, 1)
    add_slider(tab_effects, 'Posterize', '-POSTERIZE-', 1, 8, 8)
    add_slider(tab_effects, 'Solarize', '-SOLARIZE-', 0, 255, 255)
    add_slider(tab_effects, 'Threshold', '-THRESHOLD-', 0, 255, 255)
    add_slider(tab_effects, 'Vignette', '-VIGNETTE-', 0, 100, 0)
    add_slider(tab_effects, 'Sepia', '-SEPIA-', 0, 100, 0)
    
    # Artistic Tab (New)
    tab_artistic = ttk.Frame(notebook)
    notebook.add(tab_artistic, text='Artistic')
    add_slider(tab_artistic, 'Chromatic Aberration', '-CHROMATIC-', 0, 20, 0)
    add_slider(tab_artistic, 'Scanlines', '-SCANLINE-', 0, 255, 0)
    
    # Filters Tab
    tab_filters = ttk.Frame(notebook)
    notebook.add(tab_filters, text='Filters')
    f_grid = ttk.Frame(tab_filters)
    f_grid.pack(fill='x', padx=5, pady=5)
    add_checkbox(f_grid, 'Detail', '-DETAIL-').grid(row=0, column=0, sticky='w')
    add_checkbox(f_grid, 'Edge Enhance', '-EDGE-').grid(row=0, column=1, sticky='w')
    add_checkbox(f_grid, 'Emboss', '-EMBOSS-').grid(row=1, column=0, sticky='w')
    add_checkbox(f_grid, 'Contour', '-CONTOUR-').grid(row=1, column=1, sticky='w')
    add_checkbox(f_grid, 'Invert', '-INVERT-').grid(row=2, column=0, sticky='w')
    add_checkbox(f_grid, 'Grayscale', '-GRAYSCALE-').grid(row=2, column=1, sticky='w')
    add_checkbox(f_grid, 'Auto Contrast', '-AUTO_CONTRAST-').grid(row=3, column=0, sticky='w')
    add_checkbox(f_grid, 'Equalize', '-EQUALIZE-').grid(row=3, column=1, sticky='w')
    
    # Adjustments Tab
    tab_adjust = ttk.Frame(notebook)
    notebook.add(tab_adjust, text='Adjustments')
    add_slider(tab_adjust, 'Contrast', '-CONTRAST-', 0.0, 2.0, 1.0, 0.1)
    add_slider(tab_adjust, 'Brightness', '-BRIGHTNESS-', 0.0, 2.0, 1.0, 0.1)
    add_slider(tab_adjust, 'Gamma', '-GAMMA-', 0.1, 3.0, 1.0, 0.1)
    add_slider(tab_adjust, 'Sharpness', '-SHARPNESS-', 0.0, 2.0, 1.0, 0.1)
    add_slider(tab_adjust, 'Unsharp Mask', '-UNSHARP-', 0.0, 10.0, 0.0, 0.5)
    
    # Color Tab
    tab_color = ttk.Frame(notebook)
    notebook.add(tab_color, text='Color')
    add_slider(tab_color, 'Temperature', '-TEMPERATURE-', -100, 100, 0)
    add_slider(tab_color, 'Hue', '-HUE-', -180, 180, 0)
    add_slider(tab_color, 'Saturation', '-COLOR-', 0.0, 2.0, 1.0, 0.1)
    add_slider(tab_color, 'Red', '-R_FACTOR-', 0.0, 2.0, 1.0, 0.1)
    add_slider(tab_color, 'Green', '-G_FACTOR-', 0.0, 2.0, 1.0, 0.1)
    add_slider(tab_color, 'Blue', '-B_FACTOR-', 0.0, 2.0, 1.0, 0.1)
    
    # Transform Tab
    tab_transform = ttk.Frame(notebook)
    notebook.add(tab_transform, text='Transform')
    t_grid = ttk.Frame(tab_transform)
    t_grid.pack(fill='x', padx=5, pady=5)
    add_checkbox(t_grid, 'Flip X', '-FLIPX-').pack(side='left')
    add_checkbox(t_grid, 'Flip Y', '-FLIPY-').pack(side='left')
    
    add_slider(tab_transform, 'Rotation', '-ROTATION-', 0, 360, 0)
    rot_btn_frame = ttk.Frame(tab_transform)
    rot_btn_frame.pack(fill='x', padx=5)
    ttk.Button(rot_btn_frame, text='-90°', command=rotate_m90).pack(side='left', expand=True)
    ttk.Button(rot_btn_frame, text='+90°', command=rotate_p90).pack(side='left', expand=True)
    
    add_slider(tab_transform, 'Scale %', '-SCALE-', 10, 200, 100)
    
    crop_frame = ttk.LabelFrame(tab_transform, text='Crop %')
    crop_frame.pack(fill='x', padx=5, pady=2)
    # L, R, T, B sliders
    # To save space, maybe 2x2 grid
    c_grid = ttk.Frame(crop_frame)
    c_grid.pack(fill='x')
    
    # Helper for crop sliders
    def add_crop_slider(parent, label, key, row, col):
        ttk.Label(parent, text=label).grid(row=row, column=col*2, sticky='e')
        var = tk.DoubleVar(value=0)
        gui_vars[key] = var
        defaults[key] = 0
        s = tk.Scale(parent, from_=0, to=45, orient='horizontal', variable=var, command=on_change)
        s.grid(row=row, column=col*2+1, sticky='ew')
        s.bind("<ButtonRelease-1>", save_history)
        
    add_crop_slider(c_grid, 'L', '-CROP_L-', 0, 0)
    add_crop_slider(c_grid, 'R', '-CROP_R-', 0, 1)
    add_crop_slider(c_grid, 'T', '-CROP_T-', 1, 0)
    add_crop_slider(c_grid, 'B', '-CROP_B-', 1, 1)
    
    # Border Tab
    tab_border = ttk.Frame(notebook)
    notebook.add(tab_border, text='Border')
    add_slider(tab_border, 'Size (px)', '-BORDER_SIZE-', 0, 100, 0)
    
    b_col_frame = ttk.Frame(tab_border)
    b_col_frame.pack(fill='x', padx=5, pady=5)
    ttk.Label(b_col_frame, text='Color').pack(side='left')
    
    b_col_var = tk.StringVar(value='#FFFFFF')
    gui_vars['-BORDER_COLOR-'] = b_col_var
    defaults['-BORDER_COLOR-'] = '#FFFFFF'
    
    b_col_entry = ttk.Entry(b_col_frame, textvariable=b_col_var, width=10)
    b_col_entry.pack(side='left', padx=5)
    b_col_var.trace_add('write', lambda *args: [on_change(), save_history()])
    
    b_col_btn = ttk.Button(b_col_frame, text='Pick')
    b_col_btn.configure(command=lambda: pick_color('-BORDER_COLOR-'))
    b_col_btn.pack(side='left', padx=5)
    
    # Watermark Tab (New)
    tab_watermark = ttk.Frame(notebook)
    notebook.add(tab_watermark, text='Watermark')
    
    wm_frame = ttk.Frame(tab_watermark)
    wm_frame.pack(fill='x', padx=5, pady=5)
    ttk.Label(wm_frame, text='Image:').pack(side='left')
    
    wm_var = tk.StringVar(value='')
    gui_vars['-WATERMARK_PATH-'] = wm_var
    defaults['-WATERMARK_PATH-'] = ''
    
    wm_entry = ttk.Entry(wm_frame, textvariable=wm_var)
    wm_entry.pack(side='left', fill='x', expand=True, padx=5)
    wm_var.trace_add('write', lambda *args: [on_change(), save_history()])
    
    wm_btn = ttk.Button(wm_frame, text='Browse', command=pick_watermark)
    wm_btn.pack(side='left')
    
    add_slider(tab_watermark, 'Opacity', '-WATERMARK_OPACITY-', 0, 255, 255)
    add_slider(tab_watermark, 'Scale %', '-WATERMARK_SCALE-', 10, 200, 100)
    add_slider(tab_watermark, 'X %', '-WATERMARK_X-', 0, 100, 50)
    add_slider(tab_watermark, 'Y %', '-WATERMARK_Y-', 0, 100, 50)

    # Text Tab
    tab_text = ttk.Frame(notebook)
    notebook.add(tab_text, text='Text')
    
    txt_frame = ttk.Frame(tab_text)
    txt_frame.pack(fill='x', padx=5, pady=5)
    ttk.Label(txt_frame, text='Text:').pack(side='left')
    txt_var = tk.StringVar(value='')
    gui_vars['-TEXT_CONTENT-'] = txt_var
    defaults['-TEXT_CONTENT-'] = ''
    # Entry needs to trigger update on change? Maybe on return or focus out?
    # Or trace variable
    txt_entry = ttk.Entry(txt_frame, textvariable=txt_var)
    txt_entry.pack(side='left', fill='x', expand=True)
    txt_var.trace_add('write', lambda *args: [on_change(), save_history()])
    
    add_slider(tab_text, 'Size', '-TEXT_SIZE-', 10, 200, 20)
    add_slider(tab_text, 'Opacity', '-TEXT_OPACITY-', 0, 255, 255)
    
    t_col_frame = ttk.Frame(tab_text)
    t_col_frame.pack(fill='x', padx=5, pady=5)
    ttk.Label(t_col_frame, text='Color').pack(side='left')
    
    t_col_var = tk.StringVar(value='#FFFFFF')
    gui_vars['-TEXT_COLOR-'] = t_col_var
    defaults['-TEXT_COLOR-'] = '#FFFFFF'
    
    t_col_entry = ttk.Entry(t_col_frame, textvariable=t_col_var, width=10)
    t_col_entry.pack(side='left', padx=5)
    t_col_var.trace_add('write', lambda *args: [on_change(), save_history()])
    
    t_col_btn = ttk.Button(t_col_frame, text='Pick')
    t_col_btn.configure(command=lambda: pick_color('-TEXT_COLOR-'))
    t_col_btn.pack(side='left', padx=5)
    
    add_slider(tab_text, 'X %', '-TEXT_X-', 0, 100, 50)
    add_slider(tab_text, 'Y %', '-TEXT_Y-', 0, 100, 50)
    
    # Bottom Buttons
    btn_frame = ttk.Frame(control_frame)
    btn_frame.pack(fill='x', pady=10)
    ttk.Button(btn_frame, text='Save', command=save_image).pack(side='left', expand=True)
    ttk.Button(btn_frame, text='Reset', command=reset_controls).pack(side='left', expand=True)
    ttk.Button(btn_frame, text='Undo', command=undo).pack(side='left', expand=True)
    ttk.Button(btn_frame, text='Redo', command=redo).pack(side='left', expand=True)
    
    btn_frame2 = ttk.Frame(control_frame)
    btn_frame2.pack(fill='x', pady=5)
    ttk.Button(btn_frame2, text='Save Preset', command=save_preset).pack(side='left', expand=True)
    ttk.Button(btn_frame2, text='Load Preset', command=load_preset).pack(side='left', expand=True)
    ttk.Button(btn_frame2, text='Batch', command=batch_process).pack(side='left', expand=True)
    
    info = f"Image Editor v0.12\n\nPython: {platform.python_version()}\nPillow: {Image.__version__}\nTkinter: {tk.TkVersion}\nOS: {platform.system()} {platform.release()}"
    ttk.Button(control_frame, text='About', command=lambda: show_message('About', info)).pack(pady=5)
    
    # Image Area
    image_label = ttk.Label(image_frame)
    image_label.pack(fill=tk.BOTH, expand=True)
    
    # Initial update
    prev_values = get_values()
    history.append(prev_values)
    history_index = 0
    update_image(original_image)
    
    root.mainloop()
