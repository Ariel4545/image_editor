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
import time

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

# Constants
PREVIEW_MAX_SIZE = 1024 # Will be updated dynamically based on screen size

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

def create_proxy(image):
    """Creates a low-resolution proxy of the image and returns (proxy, scale_ratio)"""
    w, h = image.size
    if w <= 0 or h <= 0:
        # Handle invalid image size gracefully
        return image.copy(), 1.0
        
    if max(w, h) > PREVIEW_MAX_SIZE:
        ratio = PREVIEW_MAX_SIZE / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        # Ensure dimensions are at least 1x1
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        # Use BILINEAR for better quality proxy generation
        proxy = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return proxy, ratio
    else:
        return image.copy(), 1.0

# function that applies effects to an image based on values
def apply_effects(original, values, scale_factor=1.0):
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

    # Scale spatial parameters
    if scale_factor != 1.0:
        blur = blur * scale_factor
        box_blur = box_blur * scale_factor
        noise = int(noise * scale_factor)
        unsharp = unsharp * scale_factor
        min_filter = int(min_filter * scale_factor)
        max_filter = int(max_filter * scale_factor)
        chromatic = int(chromatic * scale_factor)
        border_size = int(border_size * scale_factor)
        text_size = int(text_size * scale_factor)
        # Scanline stride scaling
        scanline_stride = max(1, int(4 * scale_factor))
    else:
        scanline_stride = 4

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
        # Draw lines every stride pixels
        for y in range(0, h, scanline_stride):
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
# Layer system: Each layer is {'name': str, 'original': Image, 'preview_original': Image, 'preview_current': Image, 'settings': dict, 'visible': bool, 'opacity': 255, 'scale_ratio': float}
layers = []
active_layer_idx = 0
history = []
history_index = -1
prev_values = {}
image_canvas = None
status_var = None
layer_listbox = None
last_draw_pos = None
layer_opacity_var = None
update_job = None
composite_cache = None # Stores the composite image of layers BELOW the active layer
canvas_image_id = None # Persistent ID for the canvas image item

# Zoom/Pan state
zoom_level = 1.0
pan_offset_x = 0
pan_offset_y = 0

# UI Colors
COLOR_BG = '#2b2b2b'
COLOR_FG = '#ffffff'
COLOR_ACCENT = '#4a90e2'
COLOR_PANEL = '#3c3f41'

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
    global layers, active_layer_idx, zoom_level, pan_offset_x, pan_offset_y, composite_cache
    if image.mode in ('P', 'CMYK', 'HSV'):
        image = image.convert('RGBA')
    elif image.mode != 'RGBA':
        image = image.convert('RGBA')
        
    proxy, ratio = create_proxy(image)
    
    layers = [{
        'name': 'Background',
        'original': image.copy(),
        'preview_original': proxy,
        'preview_current': proxy.copy(),
        'settings': get_values(),
        'visible': True,
        'opacity': 255,
        'scale_ratio': ratio
    }]
    active_layer_idx = 0
    composite_cache = None
    
    # Reset zoom/pan
    zoom_level = 1.0
    pan_offset_x = 0
    pan_offset_y = 0
    
    update_layer_list()

def add_layer():
    global layers, active_layer_idx, composite_cache
    if not layers: return
    w, h = layers[0]['original'].size
    new_img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    
    proxy, ratio = create_proxy(new_img)
    
    layers.append({
        'name': f'Layer {len(layers)}',
        'original': new_img,
        'preview_original': proxy,
        'preview_current': proxy.copy(),
        'settings': defaults.copy(),
        'visible': True,
        'opacity': 255,
        'scale_ratio': ratio
    })
    active_layer_idx = len(layers) - 1
    composite_cache = None # Invalidate cache
    update_layer_list()
    switch_layer(active_layer_idx)
    save_history()

def delete_layer():
    global layers, active_layer_idx, composite_cache
    if len(layers) > 1:
        layers.pop(active_layer_idx)
        if active_layer_idx >= len(layers):
            active_layer_idx = len(layers) - 1
        composite_cache = None # Invalidate cache
        update_layer_list()
        switch_layer(active_layer_idx)
        save_history()
    else:
        show_message("Error", "Cannot delete the last layer.", True)

def duplicate_layer():
    global layers, active_layer_idx, composite_cache
    if not layers: return
    
    current = layers[active_layer_idx]
    
    new_layer = {
        'name': f"{current['name']} Copy",
        'original': current['original'].copy(),
        'preview_original': current['preview_original'].copy(),
        'preview_current': current['preview_current'].copy(),
        'settings': current['settings'].copy(),
        'visible': current['visible'],
        'opacity': current['opacity'],
        'scale_ratio': current['scale_ratio']
    }
    
    layers.insert(active_layer_idx + 1, new_layer)
    active_layer_idx += 1
    composite_cache = None # Invalidate cache
    update_layer_list()
    switch_layer(active_layer_idx)
    save_history()

def import_layer():
    global layers, active_layer_idx, composite_cache
    if not layers: return
    
    path = get_file_path('Import Layer', file_types=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.dng;*.cr2;*.nef;*.arw")])
    if not path: return
    
    try:
        imported_img = load_image_from_path(path)
        if imported_img.mode != 'RGBA':
            imported_img = imported_img.convert('RGBA')
            
        canvas_w, canvas_h = layers[0]['original'].size
        
        iw, ih = imported_img.size
        if iw > canvas_w or ih > canvas_h:
            ratio = min(canvas_w/iw, canvas_h/ih)
            new_w = int(iw * ratio)
            new_h = int(ih * ratio)
            imported_img = imported_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        new_layer_img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
        
        lw, lh = imported_img.size
        x = (canvas_w - lw) // 2
        y = (canvas_h - lh) // 2
        
        new_layer_img.paste(imported_img, (x, y))
        
        proxy, ratio = create_proxy(new_layer_img)
        
        layers.append({
            'name': f'Layer {len(layers)} (Import)',
            'original': new_layer_img,
            'preview_original': proxy,
            'preview_current': proxy.copy(),
            'settings': defaults.copy(),
            'visible': True,
            'opacity': 255,
            'scale_ratio': ratio
        })
        active_layer_idx = len(layers) - 1
        composite_cache = None # Invalidate cache
        update_layer_list()
        switch_layer(active_layer_idx)
        save_history()
        
    except Exception as e:
        show_message("Error", f"Failed to import layer: {e}", True)

def switch_layer(index):
    global active_layer_idx, composite_cache
    if 0 <= index < len(layers):
        # Save current settings to old layer
        layers[active_layer_idx]['settings'] = get_values()
        
        # If switching layers, we need to invalidate cache because the "active" layer changes,
        # and our cache strategy is "everything below active".
        if active_layer_idx != index:
            composite_cache = None
            
        active_layer_idx = index
        # Load new layer settings
        set_values(layers[active_layer_idx]['settings'])
        
        # Update opacity slider
        if layer_opacity_var is not None:
            layer_opacity_var.set(layers[active_layer_idx]['opacity'])
        
        update_layer_list()
        update_image()

def update_layer_list():
    if layer_listbox:
        layer_listbox.delete(0, tk.END)
        for i, layer in enumerate(layers):
            status = "👁" if layer['visible'] else " "
            name = layer['name']
            if i == active_layer_idx:
                name = f"➤ {name}"
            layer_listbox.insert(tk.END, f"{status} {name}")

def on_layer_select(event):
    selection = layer_listbox.curselection()
    if selection:
        index = selection[0]
        switch_layer(index)

def toggle_layer_visibility():
    global composite_cache
    if 0 <= active_layer_idx < len(layers):
        layers[active_layer_idx]['visible'] = not layers[active_layer_idx]['visible']
        # Visibility change affects composition
        composite_cache = None
        update_layer_list()
        update_image()
        save_history()

def update_image(original=None, values=None):
    global layers, zoom_level, pan_offset_x, pan_offset_y, composite_cache, canvas_image_id
    if not layers: return

    # 1. Apply effects to active layer (PREVIEW)
    active_layer = layers[active_layer_idx]
    current_vals = get_values()
    
    try:
        # Use preview_original and scale_ratio
        processed = apply_effects(active_layer['preview_original'], current_vals, scale_factor=active_layer['scale_ratio'])
        active_layer['preview_current'] = processed
        active_layer['settings'] = current_vals
    except Exception as e:
        print(f"Error processing layer: {e}")

    # 2. Composite all layers (PREVIEW) - Optimized with Cache
    final_image = None
    
    # If we have a valid cache for layers below active_layer, use it
    # Cache is valid if it exists and we are compositing up to active_layer_idx
    # Note: We only cache layers strictly BELOW the active one.
    
    start_index = 0
    if composite_cache is not None and active_layer_idx > 0:
        final_image = composite_cache
        start_index = active_layer_idx # Start compositing from active layer
    
    # Composite loop
    for i in range(start_index, len(layers)):
        layer = layers[i]
        if not layer['visible']:
            continue
            
        img = layer['preview_current']
        
        # Apply layer opacity
        if layer['opacity'] < 255:
            r, g, b, a = img.split()
            a = a.point(lambda i: int(i * (layer['opacity'] / 255)))
            img = Image.merge('RGBA', (r, g, b, a))
            
        if final_image is None:
            final_image = img
        else:
            if img.size != final_image.size:
                img = img.resize(final_image.size, Image.Resampling.NEAREST)
            final_image = Image.alpha_composite(final_image, img)
        
        # Update cache if we just finished compositing the layer BEFORE the active one
        if i == active_layer_idx - 1:
            composite_cache = final_image
            
    if final_image is None:
        final_image = Image.new('RGBA', (800, 600), (0,0,0,0))

    # 3. Display
    try:
        display_image = final_image.copy()
        
        # Calculate display size based on zoom
        c_w = image_canvas.winfo_width()
        c_h = image_canvas.winfo_height()
        if c_w < 1: c_w = 1000
        if c_h < 1: c_h = 800
        
        # Calculate fit ratio based on PREVIEW size
        img_w, img_h = final_image.size
        ratio = min(c_w/img_w, c_h/img_h)
        
        # Apply zoom
        final_scale = ratio * zoom_level
        new_w = int(img_w * final_scale)
        new_h = int(img_h * final_scale)
        
        # Ensure dimensions are at least 1x1 to prevent "height and width must be > 0" error
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # Resize for display - Use BILINEAR for better quality than NEAREST
        display_image = display_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        tk_img = ImageTk.PhotoImage(display_image)
        
        if image_canvas:
            # Center position + Pan offset
            x_pos = (c_w // 2) + pan_offset_x
            y_pos = (c_h // 2) + pan_offset_y
            
            # Draw checkerboard background (only if needed, but for now simple redraw is fast enough for rects)
            draw_checkerboard(image_canvas, c_w, c_h)
            
            # Update or Create Image Item
            if canvas_image_id is None:
                canvas_image_id = image_canvas.create_image(x_pos, y_pos, image=tk_img, anchor='center')
            else:
                image_canvas.itemconfig(canvas_image_id, image=tk_img)
                image_canvas.coords(canvas_image_id, x_pos, y_pos)
                # Ensure it's above checkerboard but below brush preview
                try:
                    image_canvas.tag_raise(canvas_image_id, "checkerboard")
                except tk.TclError:
                    # If checkerboard tag doesn't exist yet, just raise to top
                    image_canvas.tag_raise(canvas_image_id)
            
            image_canvas.image = tk_img 
            
            # Store scale factor for brush coordinate mapping
            # Note: final_scale is relative to PREVIEW. 
            # We need scale relative to ORIGINAL for brush mapping.
            # original_scale = final_scale * scale_ratio
            image_canvas.scale_factor = final_scale * active_layer['scale_ratio']
            
            # Offset calculation
            img_tl_x = x_pos - (new_w // 2)
            img_tl_y = y_pos - (new_h // 2)
            
            image_canvas.offset_x = img_tl_x
            image_canvas.offset_y = img_tl_y
        
        # Update status bar with original size
        orig_w, orig_h = layers[0]['original'].size
        if status_var:
             status_var.set(f"Size: {orig_w}x{orig_h} | Mode: {final_image.mode} | Layers: {len(layers)} | Zoom: {int(zoom_level*100)}%")
             
    except Exception as e:
        print(f"Error updating display: {e}")

def draw_checkerboard(canvas, w, h, size=20):
    # Check if we already have checkerboard
    if canvas.find_withtag("checkerboard"):
        # Just lower it to bottom
        canvas.tag_lower("checkerboard")
        return

    for y in range(0, h, size):
        for x in range(0, w, size):
            if (x // size + y // size) % 2 == 0:
                color = "#444444"
            else:
                color = "#333333"
            canvas.create_rectangle(x, y, x+size, y+size, fill=color, outline="", tags="checkerboard")
    canvas.tag_lower("checkerboard")

def update_status_bar(img=None):
    pass # Handled in update_image

def on_change(*args):
    global update_job
    if update_job:
        root.after_cancel(update_job)
    update_job = root.after(50, update_image) # Faster debounce for proxy

def get_state():
    # Structural Sharing: Only copy the layer structure, not the images themselves
    layers_snapshot = []
    for layer in layers:
        # Shallow copy the dictionary
        new_layer = layer.copy()
        # Deep copy settings to avoid reference issues
        new_layer['settings'] = layer['settings'].copy()
        # Images are NOT copied. They are references.
        layers_snapshot.append(new_layer)
        
    return {
        'layers': layers_snapshot,
        'active_layer_idx': active_layer_idx,
        'gui_vars': get_values()
    }

def restore_state(state):
    global layers, active_layer_idx, composite_cache
    
    # Reconstruct layers list to detach from history
    layers = []
    for layer_data in state['layers']:
        new_layer = layer_data.copy()
        new_layer['settings'] = layer_data['settings'].copy()
        layers.append(new_layer)

    active_layer_idx = state['active_layer_idx']
    set_values(state['gui_vars'])
    
    if layer_opacity_var is not None and 0 <= active_layer_idx < len(layers):
        layer_opacity_var.set(layers[active_layer_idx]['opacity'])
        
    composite_cache = None # Invalidate cache on restore
    update_layer_list()
    update_image()

def save_history(event=None):
    global history, history_index
    current_state = get_state()
    
    if history_index >= 0:
        last_state = history[history_index]
        if last_state['gui_vars'] == current_state['gui_vars'] and \
           len(last_state['layers']) == len(current_state['layers']) and \
           last_state['active_layer_idx'] == current_state['active_layer_idx']:
               pass

    if history_index < len(history) - 1:
        history = history[:history_index+1]
    
    history.append(current_state)
    history_index += 1
    
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
    # Save the composite image (FULL RESOLUTION)
    if not layers: return
    
    # Re-composite at full resolution
    final_image = None
    for layer in layers:
        if not layer['visible']: continue
        
        # Apply effects to ORIGINAL image with scale_factor=1.0
        img = apply_effects(layer['original'], layer['settings'], scale_factor=1.0)
        
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
    if layers and layers[active_layer_idx]['preview_current']:
        hist_win = tk.Toplevel()
        hist_win.title("Histogram")
        hist_win.geometry("300x200")
        
        canvas = tk.Canvas(hist_win, bg='white')
        canvas.pack(fill=tk.BOTH, expand=True)
        
        img = layers[active_layer_idx]['preview_current']
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
    global last_draw_pos, layers, active_layer_idx, composite_cache
    last_draw_pos = (event.x, event.y)
    
    # Copy-On-Write for Painting
    layer = layers[active_layer_idx]
    layer['original'] = layer['original'].copy()
    
    # Painting on active layer does NOT affect layers below, so cache remains valid.
    # No need to invalidate composite_cache here.

def paint(event):
    global last_draw_pos, layers, active_layer_idx
    if not gui_vars.get('-BRUSH_ACTIVE-', tk.BooleanVar(value=False)).get():
        return
        
    if last_draw_pos:
        x1, y1 = last_draw_pos
        x2, y2 = event.x, event.y
        
        try:
            # Draw visual feedback on canvas
            brush_size = int(gui_vars['-BRUSH_SIZE-'].get())
            brush_color = gui_vars['-BRUSH_COLOR-'].get()
            # Canvas line
            # We need to scale brush size for canvas view?
            # Canvas coordinates are already screen coordinates.
            # Just draw line.
            image_canvas.create_line(x1, y1, x2, y2, fill=brush_color, width=brush_size * zoom_level, capstyle=tk.ROUND, smooth=True, tags="brush_preview")
            
            # Record stroke on actual image (Full Res)
            scale = getattr(image_canvas, 'scale_factor', 1) # This is preview_scale * zoom
            off_x = getattr(image_canvas, 'offset_x', 0)
            off_y = getattr(image_canvas, 'offset_y', 0)
            
            # Map to ORIGINAL image coordinates
            # scale_factor = final_scale * active_layer['scale_ratio']
            # So to get original coords: (screen - offset) / scale_factor
            
            img_x1 = (x1 - off_x) / scale
            img_y1 = (y1 - off_y) / scale
            img_x2 = (x2 - off_x) / scale
            img_y2 = (y2 - off_y) / scale
            
            layer = layers[active_layer_idx]
            draw = ImageDraw.Draw(layer['original'])
            
            brush_opacity = int(gui_vars['-BRUSH_OPACITY-'].get())
            r, g, b = ImageColor.getrgb(brush_color)
            color = (r, g, b, brush_opacity)
            
            draw.line([(img_x1, img_y1), (img_x2, img_y2)], fill=color, width=brush_size, joint='curve')
            r = brush_size / 2
            draw.ellipse((img_x1-r, img_y1-r, img_x1+r, img_y1+r), fill=color)
            draw.ellipse((img_x2-r, img_y2-r, img_x2+r, img_y2+r), fill=color)
            
            last_draw_pos = (x2, y2)
            # Do NOT call update_image() here to avoid lag
            
        except Exception as e:
            pass

def stop_paint(event):
    global last_draw_pos, layers, active_layer_idx, composite_cache
    last_draw_pos = None
    
    # Clear canvas preview lines
    image_canvas.delete("brush_preview")
    
    # Update proxy from modified original
    layer = layers[active_layer_idx]
    proxy, ratio = create_proxy(layer['original'])
    layer['preview_original'] = proxy
    layer['scale_ratio'] = ratio
    
    # Painting only affects active layer, cache below remains valid.
    
    save_history()
    update_image()

def on_canvas_resize(event):
    # Force redraw of checkerboard on resize
    if image_canvas:
        image_canvas.delete("checkerboard")
    update_image()

def move_layer_up():
    global active_layer_idx, composite_cache
    if active_layer_idx > 0:
        layers[active_layer_idx], layers[active_layer_idx-1] = layers[active_layer_idx-1], layers[active_layer_idx]
        active_layer_idx -= 1
        composite_cache = None # Invalidate cache
        update_layer_list()
        if layer_listbox:
            layer_listbox.selection_clear(0, tk.END)
            layer_listbox.selection_set(active_layer_idx)
        update_image()
        save_history()

def move_layer_down():
    global active_layer_idx, composite_cache
    if active_layer_idx < len(layers) - 1:
        layers[active_layer_idx], layers[active_layer_idx+1] = layers[active_layer_idx+1], layers[active_layer_idx]
        active_layer_idx += 1
        composite_cache = None # Invalidate cache
        update_layer_list()
        if layer_listbox:
            layer_listbox.selection_clear(0, tk.END)
            layer_listbox.selection_set(active_layer_idx)
        update_image()
        save_history()

def on_layer_opacity_change(val):
    if layers:
        layers[active_layer_idx]['opacity'] = int(float(val))
        # Opacity change on active layer does NOT affect layers below.
        # Cache remains valid.
        update_image()

def on_layer_opacity_release(event):
    save_history()

# Zoom/Pan Logic
def on_mouse_wheel(event):
    global zoom_level
    if event.delta > 0:
        zoom_level *= 1.1
    else:
        zoom_level /= 1.1
    update_image()

def start_pan(event):
    global last_draw_pos
    last_draw_pos = (event.x, event.y)

def pan(event):
    global last_draw_pos, pan_offset_x, pan_offset_y
    if last_draw_pos:
        dx = event.x - last_draw_pos[0]
        dy = event.y - last_draw_pos[1]
        pan_offset_x += dx
        pan_offset_y += dy
        last_draw_pos = (event.x, event.y)
        update_image()

def stop_pan(event):
    global last_draw_pos
    last_draw_pos = None

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

def apply_theme(root):
    style = ttk.Style(root)
    style.theme_use('clam')
    
    style.configure('.', background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_PANEL)
    style.configure('TFrame', background=COLOR_BG)
    style.configure('TLabel', background=COLOR_BG, foreground=COLOR_FG)
    style.configure('TButton', background=COLOR_PANEL, foreground=COLOR_FG, borderwidth=1, relief='flat')
    style.map('TButton', background=[('active', COLOR_ACCENT)])
    
    style.configure('TNotebook', background=COLOR_BG, tabmargins=[2, 5, 2, 0])
    style.configure('TNotebook.Tab', background=COLOR_PANEL, foreground=COLOR_FG, padding=[10, 2], borderwidth=0)
    style.map('TNotebook.Tab', background=[('selected', COLOR_ACCENT)], foreground=[('selected', '#ffffff')])
    
    style.configure('TScale', background=COLOR_BG, troughcolor=COLOR_PANEL, sliderlength=15)
    style.configure('TEntry', fieldbackground=COLOR_PANEL, foreground=COLOR_FG, insertcolor=COLOR_FG)
    style.configure('TCheckbutton', background=COLOR_BG, foreground=COLOR_FG)
    style.configure('TLabelframe', background=COLOR_BG, foreground=COLOR_FG, bordercolor=COLOR_PANEL)
    style.configure('TLabelframe.Label', background=COLOR_BG, foreground=COLOR_ACCENT)
    
    root.configure(bg=COLOR_BG)

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
    
    # Dynamic Window Sizing
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Target 75% of screen size for better visibility
    win_w = int(screen_width * 0.75)
    win_h = int(screen_height * 0.75)
    
    # Center the window
    x_pos = (screen_width - win_w) // 2
    y_pos = (screen_height - win_h) // 2
    
    root.geometry(f"{win_w}x{win_h}+{x_pos}+{y_pos}")
    
    # Dynamic Preview Size based on screen height
    # Ensure it's reasonable (e.g., between 800 and 2048)
    PREVIEW_MAX_SIZE = max(800, min(2048, int(screen_height * 0.85)))
    
    apply_theme(root)
    create_menu(root)
    
    # Top Toolbar
    toolbar = ttk.Frame(root, padding=5)
    toolbar.pack(side=tk.TOP, fill=tk.X)
    
    ttk.Button(toolbar, text='💾 Save', command=save_image).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar, text='⟲ Undo', command=undo).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar, text='⟳ Redo', command=redo).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar, text='⚠ Reset', command=reset_controls).pack(side=tk.LEFT, padx=2)
    
    main_paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg=COLOR_BG, sashwidth=4)
    main_paned.pack(fill=tk.BOTH, expand=True)
    
    # Dynamic Control Frame Width (e.g., 25% of window width, min 300, max 450)
    ctrl_w = max(300, min(450, int(win_w * 0.25)))
    
    control_frame = ttk.Frame(main_paned, width=ctrl_w)
    image_frame = ttk.Frame(main_paned)
    
    main_paned.add(control_frame)
    main_paned.add(image_frame)
    
    # Scrollable Control Panel Setup
    ctrl_canvas = tk.Canvas(control_frame, bg=COLOR_BG, highlightthickness=0)
    scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=ctrl_canvas.yview)
    scrollable_frame = ttk.Frame(ctrl_canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: ctrl_canvas.configure(
            scrollregion=ctrl_canvas.bbox("all")
        )
    )
    
    canvas_window = ctrl_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    
    def configure_scroll_window(event):
        ctrl_canvas.itemconfig(canvas_window, width=event.width)
    
    ctrl_canvas.bind("<Configure>", configure_scroll_window)
    ctrl_canvas.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side="right", fill="y")
    ctrl_canvas.pack(side="left", fill="both", expand=True)
    
    # Layer Panel (Inside scrollable frame)
    layer_frame = ttk.LabelFrame(scrollable_frame, text="Layers", padding=5)
    layer_frame.pack(fill=tk.X, padx=5, pady=5)
    
    layer_listbox = tk.Listbox(layer_frame, height=6, bg=COLOR_PANEL, fg=COLOR_FG, selectbackground=COLOR_ACCENT, relief='flat')
    layer_listbox.pack(fill='x', padx=2, pady=2)
    layer_listbox.bind('<<ListboxSelect>>', on_layer_select)
    
    l_btn_frame = ttk.Frame(layer_frame)
    l_btn_frame.pack(fill='x', pady=2)
    
    ttk.Button(l_btn_frame, text='➕', width=3, command=add_layer).pack(side='left', padx=1)
    ttk.Button(l_btn_frame, text='🗑', width=3, command=delete_layer).pack(side='left', padx=1)
    ttk.Button(l_btn_frame, text='📄', width=3, command=duplicate_layer).pack(side='left', padx=1)
    ttk.Button(l_btn_frame, text='📥', width=3, command=import_layer).pack(side='left', padx=1)
    ttk.Button(l_btn_frame, text='👁', width=3, command=toggle_layer_visibility).pack(side='left', padx=1)
    ttk.Button(l_btn_frame, text='⬆', width=3, command=move_layer_up).pack(side='left', padx=1)
    ttk.Button(l_btn_frame, text='⬇', width=3, command=move_layer_down).pack(side='left', padx=1)
    
    op_frame = ttk.Frame(layer_frame)
    op_frame.pack(fill='x', pady=2)
    ttk.Label(op_frame, text='Opacity').pack(side='left')
    layer_opacity_var = tk.DoubleVar(value=255)
    op_scale = ttk.Scale(op_frame, from_=0, to=255, orient='horizontal', variable=layer_opacity_var, command=on_layer_opacity_change)
    op_scale.pack(side='left', fill='x', expand=True, padx=5)
    op_scale.bind("<ButtonRelease-1>", on_layer_opacity_release)

    # Notebook for other tools (Inside scrollable frame)
    notebook = ttk.Notebook(scrollable_frame)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # --- Adjustments Tab (Combined) ---
    tab_adjust = ttk.Frame(notebook)
    notebook.add(tab_adjust, text='Adjust')
    
    adj_basic = ttk.LabelFrame(tab_adjust, text="Basic", padding=5)
    adj_basic.pack(fill='x', padx=5, pady=5)
    add_control_slider(adj_basic, 'Contrast', '-CONTRAST-', 0.0, 3.0, 1.0, 0.05, row=0)
    add_control_slider(adj_basic, 'Brightness', '-BRIGHTNESS-', 0.0, 3.0, 1.0, 0.05, row=1)
    add_control_slider(adj_basic, 'Saturation', '-COLOR-', 0.0, 3.0, 1.0, 0.1, row=2)
    add_control_slider(adj_basic, 'Gamma', '-GAMMA-', 0.1, 5.0, 1.0, 0.05, row=3)
    
    adj_color = ttk.LabelFrame(tab_adjust, text="Color Balance", padding=5)
    adj_color.pack(fill='x', padx=5, pady=5)
    add_control_slider(adj_color, 'Temp.', '-TEMPERATURE-', -100, 100, 0, row=0)
    add_control_slider(adj_color, 'Tint', '-TINT-', -100, 100, 0, row=1)
    add_control_slider(adj_color, 'Hue', '-HUE-', -180, 180, 0, row=2)
    
    adj_channels = ttk.LabelFrame(tab_adjust, text="RGB Channels", padding=5)
    adj_channels.pack(fill='x', padx=5, pady=5)
    add_control_slider(adj_channels, 'Red', '-R_FACTOR-', 0.0, 3.0, 1.0, 0.05, row=0)
    add_control_slider(adj_channels, 'Green', '-G_FACTOR-', 0.0, 3.0, 1.0, 0.05, row=1)
    add_control_slider(adj_channels, 'Blue', '-B_FACTOR-', 0.0, 3.0, 1.0, 0.05, row=2)
    
    # --- Filters Tab (Combined) ---
    tab_filters = ttk.Frame(notebook)
    notebook.add(tab_filters, text='Filters')
    
    # Filter Category Selector
    filter_cat_var = tk.StringVar(value="Effects")
    
    def show_filter_frame(val):
        f_effects.pack_forget()
        f_artistic.pack_forget()
        f_toggles.pack_forget()
        if val == "Effects":
            f_effects.pack(fill='x', padx=5, pady=5)
        elif val == "Artistic":
            f_artistic.pack(fill='x', padx=5, pady=5)
        elif val == "Toggles":
            f_toggles.pack(fill='x', padx=5, pady=5)
            
    ttk.OptionMenu(tab_filters, filter_cat_var, "Effects", "Effects", "Artistic", "Toggles", command=show_filter_frame).pack(fill='x', padx=5, pady=5)
    
    f_effects = ttk.Frame(tab_filters)
    add_control_slider(f_effects, 'Blur', '-BLUR-', 0, 20, 0, row=0)
    add_control_slider(f_effects, 'Box Blur', '-BOX_BLUR-', 0, 20, 0, row=1)
    add_control_slider(f_effects, 'Sharpen', '-SHARPNESS-', 0.0, 5.0, 1.0, 0.1, row=2)
    add_control_slider(f_effects, 'Unsharp', '-UNSHARP-', 0.0, 20.0, 0.0, 0.5, row=3)
    add_control_slider(f_effects, 'Noise', '-NOISE-', 0, 10, 0, row=4)
    add_control_slider(f_effects, 'Vignette', '-VIGNETTE-', 0, 100, 0, row=5)
    add_control_slider(f_effects, 'Min Filter', '-MIN_FILTER-', 0, 10, 0, row=6)
    add_control_slider(f_effects, 'Max Filter', '-MAX_FILTER-', 0, 10, 0, row=7)
    f_effects.pack(fill='x', padx=5, pady=5)
    
    f_artistic = ttk.Frame(tab_filters)
    add_control_slider(f_artistic, 'Pixelate', '-PIXELATE-', 1, 50, 1, row=0)
    add_control_slider(f_artistic, 'Posterize', '-POSTERIZE-', 1, 8, 8, row=1)
    add_control_slider(f_artistic, 'Solarize', '-SOLARIZE-', 0, 255, 255, row=2)
    add_control_slider(f_artistic, 'Threshold', '-THRESHOLD-', 0, 255, 255, row=3)
    add_control_slider(f_artistic, 'Sepia', '-SEPIA-', 0, 100, 0, row=4)
    add_control_slider(f_artistic, 'Chromatic', '-CHROMATIC-', 0, 50, 0, row=5)
    add_control_slider(f_artistic, 'Scanlines', '-SCANLINE-', 0, 255, 0, row=6)
    
    f_toggles = ttk.LabelFrame(tab_filters, text="Quick Filters")
    add_checkbox(f_toggles, 'Invert', '-INVERT-', row=0, col=0)
    add_checkbox(f_toggles, 'Grayscale', '-GRAYSCALE-', row=0, col=1)
    add_checkbox(f_toggles, 'Auto Contrast', '-AUTO_CONTRAST-', row=1, col=0)
    add_checkbox(f_toggles, 'Equalize', '-EQUALIZE-', row=1, col=1)
    add_checkbox(f_toggles, 'Emboss', '-EMBOSS-', row=2, col=0)
    add_checkbox(f_toggles, 'Contour', '-CONTOUR-', row=2, col=1)
    add_checkbox(f_toggles, 'Detail', '-DETAIL-', row=3, col=0)
    add_checkbox(f_toggles, 'Edge Enhance', '-EDGE-', row=3, col=1)
    
    # --- Tools Tab ---
    tab_tools = ttk.Frame(notebook)
    notebook.add(tab_tools, text='Tools')
    
    tool_cat_var = tk.StringVar(value="Brush")
    def show_tool_frame(val):
        t_brush.pack_forget()
        t_transform.pack_forget()
        t_text.pack_forget()
        t_watermark.pack_forget()
        t_border.pack_forget()
        if val == "Brush": t_brush.pack(fill='x', padx=5, pady=5)
        elif val == "Transform": t_transform.pack(fill='x', padx=5, pady=5)
        elif val == "Text": t_text.pack(fill='x', padx=5, pady=5)
        elif val == "Watermark": t_watermark.pack(fill='x', padx=5, pady=5)
        elif val == "Border": t_border.pack(fill='x', padx=5, pady=5)
        
    ttk.OptionMenu(tab_tools, tool_cat_var, "Brush", "Brush", "Transform", "Text", "Watermark", "Border", command=show_tool_frame).pack(fill='x', padx=5, pady=5)
    
    # Brush Tool
    t_brush = ttk.Frame(tab_tools)
    add_checkbox(t_brush, 'Active', '-BRUSH_ACTIVE-', row=0, col=0)
    add_control_slider(t_brush, 'Size', '-BRUSH_SIZE-', 1, 100, 10, row=1)
    add_control_slider(t_brush, 'Opacity', '-BRUSH_OPACITY-', 0, 255, 255, row=2)
    br_col_frame = ttk.Frame(t_brush)
    br_col_frame.grid(row=3, column=0, columnspan=3, sticky='w', padx=5, pady=5)
    ttk.Label(br_col_frame, text='Color').pack(side='left')
    br_col_var = tk.StringVar(value='#FF0000')
    gui_vars['-BRUSH_COLOR-'] = br_col_var
    defaults['-BRUSH_COLOR-'] = '#FF0000'
    ttk.Entry(br_col_frame, textvariable=br_col_var, width=10).pack(side='left', padx=5)
    ttk.Button(br_col_frame, text='Pick', command=lambda: pick_color('-BRUSH_COLOR-')).pack(side='left', padx=5)
    t_brush.pack(fill='x', padx=5, pady=5)
    
    # Transform Tool
    t_transform = ttk.Frame(tab_tools)
    add_control_slider(t_transform, 'Rotation', '-ROTATION-', 0, 360, 0, row=0)
    rot_btn_frame = ttk.Frame(t_transform)
    rot_btn_frame.grid(row=1, column=0, columnspan=3, sticky='ew', padx=5)
    ttk.Button(rot_btn_frame, text='-90°', command=rotate_m90).pack(side='left', expand=True)
    ttk.Button(rot_btn_frame, text='+90°', command=rotate_p90).pack(side='left', expand=True)
    add_control_slider(t_transform, 'Scale %', '-SCALE-', 10, 400, 100, row=2)
    tf_checks = ttk.Frame(t_transform)
    tf_checks.grid(row=3, column=0, columnspan=3, sticky='w')
    add_checkbox(tf_checks, 'Flip X', '-FLIPX-', row=0, col=0)
    add_checkbox(tf_checks, 'Flip Y', '-FLIPY-', row=0, col=1)
    
    crop_frame = ttk.LabelFrame(t_transform, text='Crop %')
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
    
    # Text Tool
    t_text = ttk.Frame(tab_tools)
    txt_frame = ttk.Frame(t_text)
    txt_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
    ttk.Label(txt_frame, text='Text:').pack(side='left')
    txt_var = tk.StringVar(value='')
    gui_vars['-TEXT_CONTENT-'] = txt_var
    defaults['-TEXT_CONTENT-'] = ''
    txt_entry = ttk.Entry(txt_frame, textvariable=txt_var)
    txt_entry.pack(side='left', fill='x', expand=True)
    txt_var.trace_add('write', lambda *args: [on_change(), save_history()])
    add_control_slider(t_text, 'Size', '-TEXT_SIZE-', 10, 300, 20, row=1)
    add_control_slider(t_text, 'Opacity', '-TEXT_OPACITY-', 0, 255, 255, row=2)
    add_control_slider(t_text, 'X %', '-TEXT_X-', 0, 100, 50, row=3)
    add_control_slider(t_text, 'Y %', '-TEXT_Y-', 0, 100, 50, row=4)
    
    # Watermark Tool
    t_watermark = ttk.Frame(tab_tools)
    wm_frame = ttk.Frame(t_watermark)
    wm_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
    ttk.Button(wm_frame, text='Select Image...', command=pick_watermark).pack(fill='x')
    add_control_slider(t_watermark, 'Opacity', '-WATERMARK_OPACITY-', 0, 255, 255, row=1)
    add_control_slider(t_watermark, 'Scale %', '-WATERMARK_SCALE-', 10, 200, 100, row=2)
    add_control_slider(t_watermark, 'X %', '-WATERMARK_X-', 0, 100, 50, row=3)
    add_control_slider(t_watermark, 'Y %', '-WATERMARK_Y-', 0, 100, 50, row=4)
    
    # Border Tool
    t_border = ttk.Frame(tab_tools)
    add_control_slider(t_border, 'Size (px)', '-BORDER_SIZE-', 0, 200, 0, row=0)
    b_col_frame = ttk.Frame(t_border)
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
    
    # Image Area (Canvas)
    image_canvas = tk.Canvas(image_frame, bg='#333333', highlightthickness=0)
    image_canvas.pack(fill=tk.BOTH, expand=True)
    
    # Bindings
    image_canvas.bind("<B1-Motion>", paint)
    image_canvas.bind("<ButtonPress-1>", start_paint)
    image_canvas.bind("<ButtonRelease-1>", stop_paint)
    image_canvas.bind("<Configure>", on_canvas_resize)
    
    # Zoom/Pan Bindings
    image_canvas.bind("<MouseWheel>", on_mouse_wheel)
    image_canvas.bind("<ButtonPress-2>", start_pan)
    image_canvas.bind("<B2-Motion>", pan)
    image_canvas.bind("<ButtonRelease-2>", stop_pan)
    
    # Status Bar
    status_var = tk.StringVar()
    status_var.set("Ready")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.FLAT, anchor='w', background=COLOR_PANEL, foreground=COLOR_FG, padding=2)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Initial Setup
    init_layers(original_image)
    prev_values = get_values()
    history.append(get_state())
    history_index = 0
    
    # Force update to get correct canvas size for initial fit
    root.update_idletasks()
    update_image()
    
    # Set initial sash position
    try:
        main_paned.sashpos(0, ctrl_w)
    except AttributeError:
        # Fallback for older Tkinter versions or if sashpos is not available
        pass
    
    root.mainloop()
