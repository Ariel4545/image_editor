# imports
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont, ImageColor
import PySimpleGUI as sg
from io import BytesIO
import os
import json

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

# function that take the original image and the values from the buttons to update the image
def update_image(original, values):
    global image
    image = apply_effects(original, values)

    bio = BytesIO()
    image.save(bio, format='PNG')

    win['-IMAGE-'].update(data=bio.getvalue())


# open file selection
image_path = sg.popup_get_file('Open', no_window=True)

if not image_path:
    exit()

# control layout
effects_tab = [
    [sg.Frame('Blur', layout=[[sg.Slider(range=(0, 10), orientation='h', key='-BLUR-')]])],
    [sg.Frame('Pixelate', layout=[[sg.Slider(range=(1, 50), default_value=1, orientation='h', key='-PIXELATE-')]])],
    [sg.Frame('Posterize', layout=[[sg.Slider(range=(1, 8), default_value=8, orientation='h', key='-POSTERIZE-')]])],
    [sg.Frame('Solarize', layout=[[sg.Slider(range=(0, 255), default_value=255, orientation='h', key='-SOLARIZE-')]])],
    [sg.Frame('Threshold', layout=[[sg.Slider(range=(0, 255), default_value=255, orientation='h', key='-THRESHOLD-')]])],
    [sg.Frame('Vignette', layout=[[sg.Slider(range=(0, 100), default_value=0, orientation='h', key='-VIGNETTE-')]])],
    [sg.Frame('Sepia', layout=[[sg.Slider(range=(0, 100), default_value=0, orientation='h', key='-SEPIA-')]])],
]

filters_tab = [
    [sg.Checkbox('Detail', key='-DETAIL-'), sg.Checkbox('Edge Enhance', key='-EDGE-')],
    [sg.Checkbox('Emboss', key='-EMBOSS-'), sg.Checkbox('Contour', key='-CONTOUR-')],
    [sg.Checkbox('Invert', key='-INVERT-')],
    [sg.Checkbox('Grayscale', key='-GRAYSCALE-')],
]

adjustments_tab = [
    [sg.Frame('Contrast', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-CONTRAST-')]])],
    [sg.Frame('Brightness', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-BRIGHTNESS-')]])],
    [sg.Frame('Gamma', layout=[[sg.Slider(range=(0.1, 3.0), default_value=1.0, resolution=0.1, orientation='h', key='-GAMMA-')]])],
    [sg.Frame('Sharpness', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-SHARPNESS-')]])],
]

color_tab = [
    [sg.Frame('Temperature', layout=[[sg.Slider(range=(-100, 100), default_value=0, orientation='h', key='-TEMPERATURE-')]])],
    [sg.Frame('Hue', layout=[[sg.Slider(range=(-180, 180), default_value=0, orientation='h', key='-HUE-')]])],
    [sg.Frame('Saturation', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-COLOR-')]])],
    [sg.Frame('Red', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-R_FACTOR-')]])],
    [sg.Frame('Green', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-G_FACTOR-')]])],
    [sg.Frame('Blue', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-B_FACTOR-')]])],
]

transform_tab = [
    [sg.Checkbox('Flip X', key='-FLIPX-'), sg.Checkbox('Flip Y', key='-FLIPY-')],
    [sg.Frame('Rotation', layout=[[sg.Slider(range=(0, 360), default_value=0, orientation='h', key='-ROTATION-')]])],
    [sg.Frame('Scale %', layout=[[sg.Slider(range=(10, 200), default_value=100, orientation='h', key='-SCALE-')]])],
    [sg.Frame('Crop %', layout=[
        [sg.Text('L'), sg.Slider(range=(0, 45), default_value=0, orientation='h', size=(10, 15), key='-CROP_L-'),
         sg.Text('R'), sg.Slider(range=(0, 45), default_value=0, orientation='h', size=(10, 15), key='-CROP_R-')],
        [sg.Text('T'), sg.Slider(range=(0, 45), default_value=0, orientation='h', size=(10, 15), key='-CROP_T-'),
         sg.Text('B'), sg.Slider(range=(0, 45), default_value=0, orientation='h', size=(10, 15), key='-CROP_B-')]
    ])],
]

text_tab = [
    [sg.Text('Text:'), sg.Input(key='-TEXT_CONTENT-', enable_events=True)],
    [sg.Frame('Font Settings', layout=[
        [sg.Text('Size'), sg.Slider(range=(10, 200), default_value=20, orientation='h', key='-TEXT_SIZE-')],
        [sg.Text('Opacity'), sg.Slider(range=(0, 255), default_value=255, orientation='h', key='-TEXT_OPACITY-')],
        [sg.Text('Color'), sg.Input(default_text='#FFFFFF', size=(10, 1), key='-TEXT_COLOR-', enable_events=True), sg.ColorChooserButton('Pick', target='-TEXT_COLOR-')]
    ])],
    [sg.Frame('Position %', layout=[
        [sg.Text('X'), sg.Slider(range=(0, 100), default_value=50, orientation='h', key='-TEXT_X-')],
        [sg.Text('Y'), sg.Slider(range=(0, 100), default_value=50, orientation='h', key='-TEXT_Y-')]
    ])]
]

control_column = sg.Column([
    [sg.TabGroup([
        [sg.Tab('Effects', effects_tab), sg.Tab('Filters', filters_tab), sg.Tab('Adjustments', adjustments_tab), sg.Tab('Color', color_tab), sg.Tab('Transform', transform_tab), sg.Tab('Text', text_tab)]
    ])],
    [sg.Button('Save', key='-SAVE-'), sg.Button('Reset', key='-RESET-'), sg.Button('Undo', key='-UNDO-'), sg.Button('Redo', key='-REDO-')],
    [sg.Button('Save Preset', key='-SAVE_PRESET-'), sg.Button('Load Preset', key='-LOAD_PRESET-'), sg.Button('Batch Process', key='-BATCH-')]
])

# image layout
image_column = sg.Column([[sg.Image(image_path, key='-IMAGE-')]])

# combination of the columns to make the full layout
layout = [
    [control_column, image_column]
]

# open the original selected image
original = Image.open(image_path)

# create the window
win = sg.Window('IMAGE EDITOR', layout)

# app loop
prev_values = None
history = []
history_index = -1

SETTINGS_KEYS = [
    '-BLUR-', '-PIXELATE-', '-POSTERIZE-', '-SOLARIZE-', '-THRESHOLD-', '-VIGNETTE-', '-SEPIA-',
    '-DETAIL-', '-EDGE-', '-EMBOSS-', '-CONTOUR-', '-INVERT-', '-GRAYSCALE-',
    '-CONTRAST-', '-BRIGHTNESS-', '-GAMMA-', '-SHARPNESS-',
    '-TEMPERATURE-', '-HUE-', '-COLOR-', '-R_FACTOR-', '-G_FACTOR-', '-B_FACTOR-',
    '-FLIPX-', '-FLIPY-', '-ROTATION-', '-SCALE-',
    '-CROP_L-', '-CROP_R-', '-CROP_T-', '-CROP_B-',
    '-TEXT_CONTENT-', '-TEXT_SIZE-', '-TEXT_OPACITY-', '-TEXT_COLOR-', '-TEXT_X-', '-TEXT_Y-'
]

while True:
    event, values = win.read(timeout=100)
    if event == sg.WIN_CLOSED:
        break
    
    if event == '-RESET-':
        # Reset all values to defaults
        win['-BLUR-'].update(0)
        win['-PIXELATE-'].update(1)
        win['-POSTERIZE-'].update(8)
        win['-SOLARIZE-'].update(255)
        win['-THRESHOLD-'].update(255)
        win['-VIGNETTE-'].update(0)
        win['-SEPIA-'].update(0)
        win['-DETAIL-'].update(False)
        win['-EDGE-'].update(False)
        win['-EMBOSS-'].update(False)
        win['-CONTOUR-'].update(False)
        win['-INVERT-'].update(False)
        win['-GRAYSCALE-'].update(False)
        win['-CONTRAST-'].update(1.0)
        win['-BRIGHTNESS-'].update(1.0)
        win['-GAMMA-'].update(1.0)
        win['-SHARPNESS-'].update(1.0)
        win['-TEMPERATURE-'].update(0)
        win['-HUE-'].update(0)
        win['-COLOR-'].update(1.0)
        win['-R_FACTOR-'].update(1.0)
        win['-G_FACTOR-'].update(1.0)
        win['-B_FACTOR-'].update(1.0)
        win['-FLIPX-'].update(False)
        win['-FLIPY-'].update(False)
        win['-ROTATION-'].update(0)
        win['-SCALE-'].update(100)
        win['-CROP_L-'].update(0)
        win['-CROP_R-'].update(0)
        win['-CROP_T-'].update(0)
        win['-CROP_B-'].update(0)
        
        # Reset Text Tab
        win['-TEXT_CONTENT-'].update('')
        win['-TEXT_SIZE-'].update(20)
        win['-TEXT_OPACITY-'].update(255)
        win['-TEXT_COLOR-'].update('#FFFFFF')
        win['-TEXT_X-'].update(50)
        win['-TEXT_Y-'].update(50)
        
        # Force update loop to pick up changes
        values = win.read(timeout=0)[1]

    # Handle Undo
    if event == '-UNDO-':
        if history_index > 0:
            history_index -= 1
            restore_values = history[history_index]
            # Update UI elements
            for key in restore_values:
                try:
                    win[key].update(restore_values[key])
                except:
                    pass
            values = restore_values
            prev_values = values
            update_image(original, values)

    # Handle Redo
    elif event == '-REDO-':
        if history_index < len(history) - 1:
            history_index += 1
            restore_values = history[history_index]
            # Update UI elements
            for key in restore_values:
                try:
                    win[key].update(restore_values[key])
                except:
                    pass
            values = restore_values
            prev_values = values
            update_image(original, values)
            
    # Handle Save Preset
    if event == '-SAVE_PRESET-':
        preset_path = sg.popup_get_file('Save Preset', save_as=True, no_window=True, file_types=(("JSON", "*.json"),))
        if preset_path:
            if not preset_path.lower().endswith('.json'):
                preset_path += '.json'
            
            data_to_save = {k: values[k] for k in SETTINGS_KEYS if k in values}
            try:
                with open(preset_path, 'w') as f:
                    json.dump(data_to_save, f, indent=4)
                sg.popup('Preset saved successfully!')
            except Exception as e:
                sg.popup_error(f'Error saving preset: {e}')

    # Handle Load Preset
    if event == '-LOAD_PRESET-':
        preset_path = sg.popup_get_file('Load Preset', no_window=True, file_types=(("JSON", "*.json"),))
        if preset_path:
            try:
                with open(preset_path, 'r') as f:
                    loaded_data = json.load(f)
                
                # Update UI and values
                for key, val in loaded_data.items():
                    if key in win.element_list(): # Check if element exists
                        win[key].update(val)
                        values[key] = val # Update current values dict
                
                # Trigger update
                prev_values = values
                update_image(original, values)
                
                # Add to history
                if history_index < len(history) - 1:
                    history = history[:history_index+1]
                history.append(values)
                history_index += 1
                
            except Exception as e:
                sg.popup_error(f'Error loading preset: {e}')

    # Handle Batch Process
    if event == '-BATCH-':
        source_folder = sg.popup_get_folder('Select Source Folder', no_window=True)
        if source_folder:
            dest_folder = sg.popup_get_folder('Select Destination Folder', no_window=True)
            if dest_folder:
                # Get list of files
                try:
                    files = os.listdir(source_folder)
                    count = 0
                    for filename in files:
                        lower_name = filename.lower()
                        if lower_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            try:
                                file_path = os.path.join(source_folder, filename)
                                img = Image.open(file_path)
                                
                                # Apply effects
                                processed_img = apply_effects(img, values)
                                
                                # Save
                                save_path = os.path.join(dest_folder, filename)
                                # Handle format
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
                    
                    sg.popup(f'Batch processing complete! Processed {count} images.')
                except Exception as e:
                    sg.popup_error(f'Error reading folder: {e}')

    if values != prev_values:
        # New change detected
        
        # If we are not at the end of history, truncate it (new branch)
        if history_index < len(history) - 1:
            history = history[:history_index+1]
        
        history.append(values)
        history_index += 1
        
        # Limit history size
        if len(history) > 50:
            history.pop(0)
            history_index -= 1
            
        update_image(original, values)
        prev_values = values

    # if user pressed save button
    if event == '-SAVE-':
        save_path = sg.popup_get_file('Save', save_as=True, no_window=True, file_types=(("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")))
        if save_path:
            # check extension
            filename, file_extension = os.path.splitext(save_path)
            if not file_extension:
                # default to png
                save_path += '.png'
                file_extension = '.png'
            
            # save based on extension
            if file_extension.lower() in ['.jpg', '.jpeg']:
                # JPEG does not support alpha
                if image.mode in ('RGBA', 'LA'):
                    background = Image.new(image.mode[:-1], image.size, (255, 255, 255))
                    background.paste(image, image.split()[-1])
                    image_to_save = background
                else:
                    image_to_save = image
                image_to_save.save(save_path, quality=95)
            else:
                image.save(save_path)

win.close()
