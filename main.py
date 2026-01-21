# imports
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont, ImageColor, ImageTk, ImageChops
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser, Menu
from io import BytesIO
import os
import json
import ctypes
import platform
import copy

# Try importing rawpy for RAW support
try:
    import rawpy
except ImportError:
    rawpy = None

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
    
    # New features v0.13
    box_blur = values.get('-BOX_BLUR-', 0)
    min_filter = values.get('-MIN_FILTER-', 0)
    max_filter = values.get('-MAX_FILTER-', 0)
    tint = values.get('-TINT-', 0)
    
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
        
    # apply min/max filters
    if min_filter > 0:
        size = (min_filter * 2) + 1
        image = image.filter(ImageFilter.MinFilter(size))
    if max_filter > 0:
        size = (max_filter * 2) + 1
        image = image.filter(ImageFilter.MaxFilter(size))

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
    if box_blur > 0:
        image = image.filter(ImageFilter.BoxBlur(box_blur))
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

    # apply RGB balance and Temperature and Tint
    if r_factor != 1.0 or g_factor != 1.0 or b_factor != 1.0 or temperature != 0 or tint != 0:
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
            
        # Calculate tint factors (Green vs Magenta)
        tint_g = 0
        tint_rb = 0
        if tint != 0:
            if tint > 0:
                tint_g = (tint / 100.0) * 0.2
            else:
                tint_rb = (abs(tint) / 100.0) * 0.2

        # Apply factors
        final_r_factor = r_factor + temp_r + tint_rb
        final_g_factor = g_factor + tint_g
        final_b_factor = b_factor + temp_b + tint_rb

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
# Layer system: Each layer is {'name': str, 'original': Image, 'current': Image, 'settings': dict, 'visible': bool, 'opacity': 255}
layers = []
active_layer_idx = 0
history = []
history_index = -1
prev_values = {}
image_canvas = None
status_var = None
layer_listbox = None
last_draw_pos = None

def get_values():
    return {k: v.get() for k, v in gui_vars.items()}

def set_values(values):
    for k, v in values.items():
        if k in gui_vars:
            gui_vars[k].set(v)

def load_image_from_path(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.dng', '.cr2', '.nef', '.arw', '.orf', '.rw2'] and rawpy:
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess()
                return Image.fromarray(rgb)
        except Exception as e:
            print(f"RAW load error: {e}")
            return Image.open(path)
    else:
        return Image.open(path)

def init_layers(image):
    global layers, active_layer_idx
    if image.mode in ('P', 'CMYK', 'HSV'):
        image = image.convert('RGBA')
    elif image.mode != 'RGBA':
        image = image.convert('RGBA')
        
    layers = [{
        'name': 'Background',
        'original': image.copy(),
        'current': image.copy(),
        'settings': get_values(),
        'visible': True,
        'opacity': 255
    }]
    active_layer_idx = 0
    update_layer_list()

def add_layer():
    global layers, active_layer_idx
    if not layers: return
    w, h = layers[0]['original'].size
    new_img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    
    layers.append({
        'name': f'Layer {len(layers)}',
        'original': new_img,
        'current': new_img,
        'settings': defaults.copy(),
        'visible': True,
        'opacity': 255
    })
    active_layer_idx = len(layers) - 1
    update_layer_list()
    switch_layer(active_layer_idx)
    save_history()

def delete_layer():
    global layers, active_layer_idx
    if len(layers) > 1:
        layers.pop(active_layer_idx)
        if active_layer_idx >= len(layers):
            active_layer_idx = len(layers) - 1
        update_layer_list()
        switch_layer(active_layer_idx)
        save_history()
    else:
        show_message("Error", "Cannot delete the last layer.", True)

def switch_layer(index):
    global active_layer_idx
    if 0 <= index < len(layers):
        # Save current settings to old layer
        layers[active_layer_idx]['settings'] = get_values()
        
        active_layer_idx = index
        # Load new layer settings
        set_values(layers[active_layer_idx]['settings'])
        
        update_layer_list()
        update_image()

def update_layer_list():
    if layer_listbox:
        layer_listbox.delete(0, tk.END)
        for i, layer in enumerate(layers):
            status = "[v]" if layer['visible'] else "[ ]"
            name = layer['name']
            if i == active_layer_idx:
                name = f"> {name}"
            layer_listbox.insert(tk.END, f"{status} {name}")

def on_layer_select(event):
    selection = layer_listbox.curselection()
    if selection:
        index = selection[0]
        switch_layer(index)

def toggle_layer_visibility():
    if 0 <= active_layer_idx < len(layers):
        layers[active_layer_idx]['visible'] = not layers[active_layer_idx]['visible']
        update_layer_list()
        update_image()
        save_history()

def update_image(original=None, values=None):
    global layers
    if not layers: return

    # 1. Apply effects to active layer
    active_layer = layers[active_layer_idx]
    current_vals = get_values()
    
    try:
        processed = apply_effects(active_layer['original'], current_vals)
        active_layer['current'] = processed
        active_layer['settings'] = current_vals
    except Exception as e:
        print(f"Error processing layer: {e}")

    # 2. Composite all layers
    final_image = None
    for layer in layers:
        if not layer['visible']:
            continue
            
        img = layer['current']
        
        # Apply layer opacity
        if layer['opacity'] < 255:
            r, g, b, a = img.split()
            a = a.point(lambda i: int(i * (layer['opacity'] / 255)))
            img = Image.merge('RGBA', (r, g, b, a))
            
        if final_image is None:
            final_image = img
        else:
            if img.size != final_image.size:
                img = img.resize(final_image.size, Image.Resampling.LANCZOS)
            final_image = Image.alpha_composite(final_image, img)
            
    if final_image is None:
        final_image = Image.new('RGBA', (800, 600), (0,0,0,0))

    # 3. Display
    try:
        display_image = final_image.copy()
        max_w, max_h = 1000, 800
        display_image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        
        tk_img = ImageTk.PhotoImage(display_image)
        
        if image_canvas:
            image_canvas.delete("all")
            c_w = image_canvas.winfo_width()
            c_h = image_canvas.winfo_height()
            if c_w < 1: c_w = 1000
            if c_h < 1: c_h = 800
            
            x_pos = c_w // 2
            y_pos = c_h // 2
            
            image_canvas.create_image(x_pos, y_pos, image=tk_img, anchor='center')
            image_canvas.image = tk_img 
            
            image_canvas.scale_factor = display_image.width / final_image.width
            image_canvas.offset_x = x_pos - (display_image.width // 2)
            image_canvas.offset_y = y_pos - (display_image.height // 2)
        
        update_status_bar(final_image)
    except Exception as e:
        print(f"Error updating display: {e}")

def update_status_bar(img=None):
    if img and status_var:
        w, h = img.size
        mode = img.mode
        status_var.set(f"Size: {w}x{h} | Mode: {mode} | Layers: {len(layers)}")

def on_change(*args):
    global history, history_index, prev_values
    update_image()

def get_state():
    # Deep copy layers to preserve image state
    # We need to manually copy images because deepcopy might be slow or problematic with some objects
    layers_copy = []
    for layer in layers:
        new_layer = layer.copy()
        new_layer['original'] = layer['original'].copy()
        new_layer['current'] = layer['current'].copy()
        new_layer['settings'] = layer['settings'].copy()
        layers_copy.append(new_layer)
    return {
        'layers': layers_copy,
        'active_layer_idx': active_layer_idx,
        'gui_vars': get_values()
    }

def restore_state(state):
    global layers, active_layer_idx
    layers = state['layers']
    active_layer_idx = state['active_layer_idx']
    set_values(state['gui_vars'])
    update_layer_list()
    update_image()

def save_history(event=None):
    global history, history_index
    current_state = get_state()
    
    # Check if state actually changed to avoid duplicates
    if history_index >= 0:
        last_state = history[history_index]
        # Simple check on gui_vars and layer count/idx for performance
        # Full image comparison is too heavy
        if last_state['gui_vars'] == current_state['gui_vars'] and \
           len(last_state['layers']) == len(current_state['layers']) and \
           last_state['active_layer_idx'] == current_state['active_layer_idx']:
               # If painting, we must assume change. 
               # For now, we just save. Optimization can be added later.
               pass

    if history_index < len(history) - 1:
        history = history[:history_index+1]
    
    history.append(current_state)
    history_index += 1
    
    # Limit history to 20 steps to save memory (images are heavy)
    if len(history) > 20:
        history.pop(0)
        history_index -= 1

def reset_controls():
    for k, v in defaults.items():
        if k in gui_vars:
            gui_vars[k].set(v)
    save_history()
    update_image()

def undo():
    global history_index
    if history_index > 0:
        history_index -= 1
        restore_state(history[history_index])

def redo():
    global history_index
    if history_index < len(history) - 1:
        history_index += 1
        restore_state(history[history_index])

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
            update_image()
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
    # Save the composite image
    if not layers: return
    
    # Re-composite at full resolution
    final_image = None
    for layer in layers:
        if not layer['visible']: continue
        img = layer['current']
        if layer['opacity'] < 255:
            r, g, b, a = img.split()
            a = a.point(lambda i: int(i * (layer['opacity'] / 255)))
            img = Image.merge('RGBA', (r, g, b, a))
        if final_image is None:
            final_image = img
        else:
            if img.size != final_image.size:
                img = img.resize(final_image.size, Image.Resampling.LANCZOS)
            final_image = Image.alpha_composite(final_image, img)
            
    if not final_image: return

    save_path = get_file_path('Save', save_as=True, file_types=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")], default_extension=".png")
    if save_path:
        filename, file_extension = os.path.splitext(save_path)
        if not file_extension:
            save_path += '.png'
            file_extension = '.png'
        
        if file_extension.lower() in ['.jpg', '.jpeg']:
            if final_image.mode in ('RGBA', 'LA'):
                background = Image.new(final_image.mode[:-1], final_image.size, (255, 255, 255))
                background.paste(final_image, final_image.split()[-1])
                image_to_save = background
            else:
                image_to_save = final_image
            image_to_save.save(save_path, quality=95)
        else:
            final_image.save(save_path)

def rotate_m90():
    curr = gui_vars['-ROTATION-'].get()
    new_val = (curr - 90) % 360
    gui_vars['-ROTATION-'].set(new_val)
    save_history()
    update_image()

def rotate_p90():
    curr = gui_vars['-ROTATION-'].get()
    new_val = (curr + 90) % 360
    gui_vars['-ROTATION-'].set(new_val)
    save_history()
    update_image()

def pick_color(key):
    try:
        curr = gui_vars[key].get()
        color = colorchooser.askcolor(color=curr)[1]
    except:
        color = colorchooser.askcolor()[1]
        
    if color:
        gui_vars[key].set(color)
        save_history()
        update_image()

def pick_watermark():
    path = get_file_path("Select Watermark", file_types=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
    if path:
        gui_vars['-WATERMARK_PATH-'].set(path)
        save_history()
        update_image()

def show_histogram():
    if layers and layers[active_layer_idx]['current']:
        hist_win = tk.Toplevel()
        hist_win.title("Histogram")
        hist_win.geometry("300x200")
        
        canvas = tk.Canvas(hist_win, bg='white')
        canvas.pack(fill=tk.BOTH, expand=True)
        
        img = layers[active_layer_idx]['current']
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        hist = img.histogram()
        
        r_hist = hist[0:256]
        g_hist = hist[256:512]
        b_hist = hist[512:768]
        
        max_val = max(max(r_hist), max(g_hist), max(b_hist))
        if max_val == 0: max_val = 1
        
        w = 300
        h = 200
        
        for i in range(256):
            x = i * (w / 256)
            h_r = (r_hist[i] / max_val) * h
            canvas.create_line(x, h, x, h - h_r, fill='red', alpha=0.5)
            h_g = (g_hist[i] / max_val) * h
            canvas.create_line(x, h, x, h - h_g, fill='green', alpha=0.5)
            h_b = (b_hist[i] / max_val) * h
            canvas.create_line(x, h, x, h - h_b, fill='blue', alpha=0.5)

# Brush Logic
def start_paint(event):
    global last_draw_pos
    last_draw_pos = (event.x, event.y)

def paint(event):
    global last_draw_pos, layers, active_layer_idx
    if not gui_vars.get('-BRUSH_ACTIVE-', tk.BooleanVar(value=False)).get():
        return
        
    if last_draw_pos:
        x1, y1 = last_draw_pos
        x2, y2 = event.x, event.y
        
        try:
            scale = getattr(image_canvas, 'scale_factor', 1)
            off_x = getattr(image_canvas, 'offset_x', 0)
            off_y = getattr(image_canvas, 'offset_y', 0)
            
            img_x1 = (x1 - off_x) / scale
            img_y1 = (y1 - off_y) / scale
            img_x2 = (x2 - off_x) / scale
            img_y2 = (y2 - off_y) / scale
            
            layer = layers[active_layer_idx]
            draw = ImageDraw.Draw(layer['original'])
            
            brush_size = int(gui_vars['-BRUSH_SIZE-'].get())
            brush_color = gui_vars['-BRUSH_COLOR-'].get()
            brush_opacity = int(gui_vars['-BRUSH_OPACITY-'].get())
            
            r, g, b = ImageColor.getrgb(brush_color)
            color = (r, g, b, brush_opacity)
            
            draw.line([(img_x1, img_y1), (img_x2, img_y2)], fill=color, width=brush_size, joint='curve')
            r = brush_size / 2
            draw.ellipse((img_x1-r, img_y1-r, img_x1+r, img_y1+r), fill=color)
            draw.ellipse((img_x2-r, img_y2-r, img_x2+r, img_y2+r), fill=color)
            
            last_draw_pos = (x2, y2)
            update_image()
            
        except Exception as e:
            pass

def stop_paint(event):
    global last_draw_pos
    last_draw_pos = None
    save_history()

def on_canvas_resize(event):
    update_image()

# UI Construction Helpers
def add_control_slider(parent, label, key, from_, to, default, resolution=1, row=0):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
    var = tk.DoubleVar(value=default)
    gui_vars[key] = var
    defaults[key] = default
    scale = ttk.Scale(parent, from_=from_, to=to, orient='horizontal', variable=var, command=on_change)
    scale.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
    scale.bind("<ButtonRelease-1>", save_history)
    entry = ttk.Entry(parent, textvariable=var, width=5)
    entry.grid(row=row, column=2, sticky='e', padx=5, pady=2)
    entry.bind('<Return>', lambda e: [on_change(), save_history()])
    return scale

def add_checkbox(parent, label, key, default=False, row=0, col=0):
    var = tk.BooleanVar(value=default)
    gui_vars[key] = var
    defaults[key] = default
    chk = ttk.Checkbutton(parent, text=label, variable=var, command=lambda: [on_change(), save_history()])
    chk.grid(row=row, column=col, sticky='w', padx=5, pady=2)
    return chk

def create_menu(root):
    menubar = Menu(root)
    file_menu = Menu(menubar, tearoff=0)
    file_menu.add_command(label="Open", command=lambda: show_message("Info", "Restart app to open new image"))
    file_menu.add_command(label="Save", command=save_image)
    file_menu.add_separator()
    file_menu.add_command(label="Batch Process", command=batch_process)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=file_menu)
    
    edit_menu = Menu(menubar, tearoff=0)
    edit_menu.add_command(label="Undo", command=undo)
    edit_menu.add_command(label="Redo", command=redo)
    edit_menu.add_separator()
    edit_menu.add_command(label="Reset All", command=reset_controls)
    edit_menu.add_separator()
    edit_menu.add_command(label="Save Preset", command=save_preset)
    edit_menu.add_command(label="Load Preset", command=load_preset)
    menubar.add_cascade(label="Edit", menu=edit_menu)
    
    view_menu = Menu(menubar, tearoff=0)
    view_menu.add_command(label="Histogram", command=show_histogram)
    menubar.add_cascade(label="View", menu=view_menu)
    
    help_menu = Menu(menubar, tearoff=0)
    info = f"Image Editor v0.14\n\nPython: {platform.python_version()}\nPillow: {Image.__version__}\nTkinter: {tk.TkVersion}\nOS: {platform.system()} {platform.release()}"
    help_menu.add_command(label="About", command=lambda: show_message('About', info))
    menubar.add_cascade(label="Help", menu=help_menu)
    root.config(menu=menubar)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    image_path = get_file_path('Open', file_types=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.dng;*.cr2;*.nef;*.arw")])
    if not image_path:
        root.destroy()
        exit()
        
    original_image = load_image_from_path(image_path)
    
    root.deiconify()
    root.title("IMAGE EDITOR PRO")
    root.geometry("1200x800")
    
    create_menu(root)
    
    main_paned = tk.PanedWindow(root, orient=tk.HORIZONTAL)
    main_paned.pack(fill=tk.BOTH, expand=True)
    
    control_frame = ttk.Frame(main_paned, width=400)
    image_frame = ttk.Frame(main_paned)
    
    main_paned.add(control_frame)
    main_paned.add(image_frame)
    
    notebook = ttk.Notebook(control_frame)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # --- Layers Tab ---
    tab_layers = ttk.Frame(notebook)
    notebook.add(tab_layers, text='Layers')
    
    layer_listbox = tk.Listbox(tab_layers, height=10)
    layer_listbox.pack(fill='x', padx=5, pady=5)
    layer_listbox.bind('<<ListboxSelect>>', on_layer_select)
    
    l_btn_frame = ttk.Frame(tab_layers)
    l_btn_frame.pack(fill='x', padx=5)
    ttk.Button(l_btn_frame, text='Add Layer', command=add_layer).pack(side='left', expand=True)
    ttk.Button(l_btn_frame, text='Delete', command=delete_layer).pack(side='left', expand=True)
    ttk.Button(l_btn_frame, text='Toggle Vis', command=toggle_layer_visibility).pack(side='left', expand=True)
    
    # --- Brush Tab ---
    tab_brush = ttk.Frame(notebook)
    notebook.add(tab_brush, text='Brush')
    
    add_checkbox(tab_brush, 'Active', '-BRUSH_ACTIVE-', row=0, col=0)
    add_control_slider(tab_brush, 'Size', '-BRUSH_SIZE-', 1, 100, 10, row=1)
    add_control_slider(tab_brush, 'Opacity', '-BRUSH_OPACITY-', 0, 255, 255, row=2)
    
    br_col_frame = ttk.Frame(tab_brush)
    br_col_frame.grid(row=3, column=0, columnspan=3, sticky='w', padx=5, pady=5)
    ttk.Label(br_col_frame, text='Color').pack(side='left')
    br_col_var = tk.StringVar(value='#FF0000')
    gui_vars['-BRUSH_COLOR-'] = br_col_var
    defaults['-BRUSH_COLOR-'] = '#FF0000'
    ttk.Entry(br_col_frame, textvariable=br_col_var, width=10).pack(side='left', padx=5)
    ttk.Button(br_col_frame, text='Pick', command=lambda: pick_color('-BRUSH_COLOR-')).pack(side='left', padx=5)

    # --- Effects Tab ---
    tab_effects = ttk.Frame(notebook)
    notebook.add(tab_effects, text='Effects')
    tab_effects.columnconfigure(1, weight=1)
    add_control_slider(tab_effects, 'Blur', '-BLUR-', 0, 20, 0, row=0)
    add_control_slider(tab_effects, 'Box Blur', '-BOX_BLUR-', 0, 20, 0, row=1)
    add_control_slider(tab_effects, 'Noise Red.', '-NOISE-', 0, 10, 0, row=2)
    add_control_slider(tab_effects, 'Pixelate', '-PIXELATE-', 1, 50, 1, row=3)
    add_control_slider(tab_effects, 'Posterize', '-POSTERIZE-', 1, 8, 8, row=4)
    add_control_slider(tab_effects, 'Solarize', '-SOLARIZE-', 0, 255, 255, row=5)
    add_control_slider(tab_effects, 'Threshold', '-THRESHOLD-', 0, 255, 255, row=6)
    add_control_slider(tab_effects, 'Vignette', '-VIGNETTE-', 0, 100, 0, row=7)
    add_control_slider(tab_effects, 'Sepia', '-SEPIA-', 0, 100, 0, row=8)
    add_control_slider(tab_effects, 'Min Filter', '-MIN_FILTER-', 0, 10, 0, row=9)
    add_control_slider(tab_effects, 'Max Filter', '-MAX_FILTER-', 0, 10, 0, row=10)
    
    # --- Artistic Tab ---
    tab_artistic = ttk.Frame(notebook)
    notebook.add(tab_artistic, text='Artistic')
    tab_artistic.columnconfigure(1, weight=1)
    add_control_slider(tab_artistic, 'Chromatic', '-CHROMATIC-', 0, 50, 0, row=0)
    add_control_slider(tab_artistic, 'Scanlines', '-SCANLINE-', 0, 255, 0, row=1)
    
    # --- Filters Tab ---
    tab_filters = ttk.Frame(notebook)
    notebook.add(tab_filters, text='Filters')
    f_grid = ttk.LabelFrame(tab_filters, text="Toggles")
    f_grid.pack(fill='x', padx=5, pady=5)
    add_checkbox(f_grid, 'Detail', '-DETAIL-', row=0, col=0)
    add_checkbox(f_grid, 'Edge Enhance', '-EDGE-', row=0, col=1)
    add_checkbox(f_grid, 'Emboss', '-EMBOSS-', row=1, col=0)
    add_checkbox(f_grid, 'Contour', '-CONTOUR-', row=1, col=1)
    add_checkbox(f_grid, 'Invert', '-INVERT-', row=2, col=0)
    add_checkbox(f_grid, 'Grayscale', '-GRAYSCALE-', row=2, col=1)
    add_checkbox(f_grid, 'Auto Contrast', '-AUTO_CONTRAST-', row=3, col=0)
    add_checkbox(f_grid, 'Equalize', '-EQUALIZE-', row=3, col=1)
    
    # --- Adjustments Tab ---
    tab_adjust = ttk.Frame(notebook)
    notebook.add(tab_adjust, text='Adjust')
    tab_adjust.columnconfigure(1, weight=1)
    add_control_slider(tab_adjust, 'Contrast', '-CONTRAST-', 0.0, 3.0, 1.0, 0.05, row=0)
    add_control_slider(tab_adjust, 'Brightness', '-BRIGHTNESS-', 0.0, 3.0, 1.0, 0.05, row=1)
    add_control_slider(tab_adjust, 'Gamma', '-GAMMA-', 0.1, 5.0, 1.0, 0.05, row=2)
    add_control_slider(tab_adjust, 'Sharpness', '-SHARPNESS-', 0.0, 5.0, 1.0, 0.1, row=3)
    add_control_slider(tab_adjust, 'Unsharp', '-UNSHARP-', 0.0, 20.0, 0.0, 0.5, row=4)
    
    # --- Color Tab ---
    tab_color = ttk.Frame(notebook)
    notebook.add(tab_color, text='Color')
    tab_color.columnconfigure(1, weight=1)
    add_control_slider(tab_color, 'Temp.', '-TEMPERATURE-', -100, 100, 0, row=0)
    add_control_slider(tab_color, 'Tint', '-TINT-', -100, 100, 0, row=1)
    add_control_slider(tab_color, 'Hue', '-HUE-', -180, 180, 0, row=2)
    add_control_slider(tab_color, 'Saturation', '-COLOR-', 0.0, 3.0, 1.0, 0.1, row=3)
    ttk.Separator(tab_color, orient='horizontal').grid(row=4, column=0, columnspan=3, sticky='ew', pady=5)
    add_control_slider(tab_color, 'Red', '-R_FACTOR-', 0.0, 3.0, 1.0, 0.05, row=5)
    add_control_slider(tab_color, 'Green', '-G_FACTOR-', 0.0, 3.0, 1.0, 0.05, row=6)
    add_control_slider(tab_color, 'Blue', '-B_FACTOR-', 0.0, 3.0, 1.0, 0.05, row=7)
    
    # --- Transform Tab ---
    tab_transform = ttk.Frame(notebook)
    notebook.add(tab_transform, text='Transform')
    tab_transform.columnconfigure(1, weight=1)
    t_grid = ttk.Frame(tab_transform)
    t_grid.grid(row=0, column=0, columnspan=3, sticky='w', padx=5, pady=5)
    add_checkbox(t_grid, 'Flip X', '-FLIPX-', row=0, col=0)
    add_checkbox(t_grid, 'Flip Y', '-FLIPY-', row=0, col=1)
    add_control_slider(tab_transform, 'Rotation', '-ROTATION-', 0, 360, 0, row=1)
    rot_btn_frame = ttk.Frame(tab_transform)
    rot_btn_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5)
    ttk.Button(rot_btn_frame, text='-90°', command=rotate_m90).pack(side='left', expand=True)
    ttk.Button(rot_btn_frame, text='+90°', command=rotate_p90).pack(side='left', expand=True)
    add_control_slider(tab_transform, 'Scale %', '-SCALE-', 10, 400, 100, row=3)
    crop_frame = ttk.LabelFrame(tab_transform, text='Crop %')
    crop_frame.grid(row=4, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
    def add_crop_slider(parent, label, key, row, col):
        ttk.Label(parent, text=label).grid(row=row, column=col*2, sticky='e', padx=2)
        var = tk.DoubleVar(value=0)
        gui_vars[key] = var
        defaults[key] = 0
        s = ttk.Scale(parent, from_=0, to=45, orient='horizontal', variable=var, command=on_change)
        s.grid(row=row, column=col*2+1, sticky='ew', padx=2)
        s.bind("<ButtonRelease-1>", save_history)
    add_crop_slider(crop_frame, 'L', '-CROP_L-', 0, 0)
    add_crop_slider(crop_frame, 'R', '-CROP_R-', 0, 1)
    add_crop_slider(crop_frame, 'T', '-CROP_T-', 1, 0)
    add_crop_slider(crop_frame, 'B', '-CROP_B-', 1, 1)
    
    # --- Border Tab ---
    tab_border = ttk.Frame(notebook)
    notebook.add(tab_border, text='Border')
    tab_border.columnconfigure(1, weight=1)
    add_control_slider(tab_border, 'Size (px)', '-BORDER_SIZE-', 0, 200, 0, row=0)
    b_col_frame = ttk.Frame(tab_border)
    b_col_frame.grid(row=1, column=0, columnspan=3, sticky='w', padx=5, pady=5)
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
    
    # --- Watermark Tab ---
    tab_watermark = ttk.Frame(notebook)
    notebook.add(tab_watermark, text='Watermark')
    tab_watermark.columnconfigure(1, weight=1)
    wm_frame = ttk.Frame(tab_watermark)
    wm_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
    ttk.Label(wm_frame, text='Image:').pack(side='left')
    wm_var = tk.StringVar(value='')
    gui_vars['-WATERMARK_PATH-'] = wm_var
    defaults['-WATERMARK_PATH-'] = ''
    wm_entry = ttk.Entry(wm_frame, textvariable=wm_var)
    wm_entry.pack(side='left', fill='x', expand=True, padx=5)
    wm_var.trace_add('write', lambda *args: [on_change(), save_history()])
    wm_btn = ttk.Button(wm_frame, text='Browse', command=pick_watermark)
    wm_btn.pack(side='left')
    add_control_slider(tab_watermark, 'Opacity', '-WATERMARK_OPACITY-', 0, 255, 255, row=1)
    add_control_slider(tab_watermark, 'Scale %', '-WATERMARK_SCALE-', 10, 200, 100, row=2)
    add_control_slider(tab_watermark, 'X %', '-WATERMARK_X-', 0, 100, 50, row=3)
    add_control_slider(tab_watermark, 'Y %', '-WATERMARK_Y-', 0, 100, 50, row=4)

    # --- Text Tab ---
    tab_text = ttk.Frame(notebook)
    notebook.add(tab_text, text='Text')
    tab_text.columnconfigure(1, weight=1)
    txt_frame = ttk.Frame(tab_text)
    txt_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
    ttk.Label(txt_frame, text='Text:').pack(side='left')
    txt_var = tk.StringVar(value='')
    gui_vars['-TEXT_CONTENT-'] = txt_var
    defaults['-TEXT_CONTENT-'] = ''
    txt_entry = ttk.Entry(txt_frame, textvariable=txt_var)
    txt_entry.pack(side='left', fill='x', expand=True)
    txt_var.trace_add('write', lambda *args: [on_change(), save_history()])
    add_control_slider(tab_text, 'Size', '-TEXT_SIZE-', 10, 300, 20, row=1)
    add_control_slider(tab_text, 'Opacity', '-TEXT_OPACITY-', 0, 255, 255, row=2)
    t_col_frame = ttk.Frame(tab_text)
    t_col_frame.grid(row=3, column=0, columnspan=3, sticky='w', padx=5, pady=5)
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
    add_control_slider(tab_text, 'X %', '-TEXT_X-', 0, 100, 50, row=4)
    add_control_slider(tab_text, 'Y %', '-TEXT_Y-', 0, 100, 50, row=5)
    
    # Bottom Buttons
    btn_frame = ttk.Frame(control_frame)
    btn_frame.pack(fill='x', pady=10)
    ttk.Button(btn_frame, text='Save', command=save_image).pack(side='left', expand=True, padx=2)
    ttk.Button(btn_frame, text='Reset', command=reset_controls).pack(side='left', expand=True, padx=2)
    ttk.Button(btn_frame, text='Undo', command=undo).pack(side='left', expand=True, padx=2)
    ttk.Button(btn_frame, text='Redo', command=redo).pack(side='left', expand=True, padx=2)
    
    # Image Area (Canvas)
    image_canvas = tk.Canvas(image_frame, bg='#333333')
    image_canvas.pack(fill=tk.BOTH, expand=True)
    
    # Bindings
    image_canvas.bind("<B1-Motion>", paint)
    image_canvas.bind("<ButtonPress-1>", start_paint)
    image_canvas.bind("<ButtonRelease-1>", stop_paint)
    image_canvas.bind("<Configure>", on_canvas_resize)
    
    # Status Bar
    status_var = tk.StringVar()
    status_var.set("Ready")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor='w')
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Initial Setup
    init_layers(original_image)
    prev_values = get_values()
    history.append(get_state())
    history_index = 0
    update_image()
    
    root.mainloop()
