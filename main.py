# imports
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import PySimpleGUI as sg
from io import BytesIO
import os

# function that take the original image and the values from the buttons to update the image
def update_image(original, blur, contrast, brightness, color, sharpness, edge_enhance, detail, emboss, contour, flipx, flipy, rotation, scale, invert, sepia, posterize, solarize, r_factor, g_factor, b_factor):
    global image
    image = original.copy()

    # apply rotation
    if rotation != 0:
        image = image.rotate(rotation, expand=True)

    # apply flip
    if flipx:
        image = ImageOps.mirror(image)
    if flipy:
        image = ImageOps.flip(image)

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

    if sepia:
        if image.mode == 'RGBA':
            alpha = image.split()[3]
            gray = image.convert('L')
            image = ImageOps.colorize(gray, '#704214', '#C0C080')
            image.putalpha(alpha)
        else:
            gray = image.convert('L')
            image = ImageOps.colorize(gray, '#704214', '#C0C080')

    # apply RGB balance
    if r_factor != 1.0 or g_factor != 1.0 or b_factor != 1.0:
        if image.mode == 'L':
            image = image.convert('RGB')
        
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
        else:
            r, g, b = image.split()
            a = None
        
        if r_factor != 1.0:
            r = r.point(lambda i: i * r_factor)
        if g_factor != 1.0:
            g = g.point(lambda i: i * g_factor)
        if b_factor != 1.0:
            b = b.point(lambda i: i * b_factor)
            
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
    [sg.Frame('Posterize', layout=[[sg.Slider(range=(1, 8), default_value=8, orientation='h', key='-POSTERIZE-')]])],
    [sg.Frame('Solarize', layout=[[sg.Slider(range=(0, 255), default_value=255, orientation='h', key='-SOLARIZE-')]])],
]

filters_tab = [
    [sg.Checkbox('Detail', key='-DETAIL-'), sg.Checkbox('Edge Enhance', key='-EDGE-')],
    [sg.Checkbox('Emboss', key='-EMBOSS-'), sg.Checkbox('Contour', key='-CONTOUR-')],
    [sg.Checkbox('Invert', key='-INVERT-'), sg.Checkbox('Sepia', key='-SEPIA-')],
]

adjustments_tab = [
    [sg.Frame('Contrast', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-CONTRAST-')]])],
    [sg.Frame('Brightness', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-BRIGHTNESS-')]])],
    [sg.Frame('Sharpness', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-SHARPNESS-')]])],
]

color_tab = [
    [sg.Frame('Saturation', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-COLOR-')]])],
    [sg.Frame('Red', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-R_FACTOR-')]])],
    [sg.Frame('Green', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-G_FACTOR-')]])],
    [sg.Frame('Blue', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-B_FACTOR-')]])],
]

transform_tab = [
    [sg.Checkbox('Flip X', key='-FLIPX-'), sg.Checkbox('Flip Y', key='-FLIPY-')],
    [sg.Frame('Rotation', layout=[[sg.Slider(range=(0, 360), default_value=0, orientation='h', key='-ROTATION-')]])],
    [sg.Frame('Scale %', layout=[[sg.Slider(range=(10, 200), default_value=100, orientation='h', key='-SCALE-')]])],
]

control_column = sg.Column([
    [sg.TabGroup([
        [sg.Tab('Effects', effects_tab), sg.Tab('Filters', filters_tab), sg.Tab('Adjustments', adjustments_tab), sg.Tab('Color', color_tab), sg.Tab('Transform', transform_tab)]
    ])],
    [sg.Button('Save', key='-SAVE-')],
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
while True:
    event, values = win.read(timeout=100)
    if event == sg.WIN_CLOSED:
        break
    
    if values != prev_values:
        # passing the buttons arguments in real time
        update_image(original,
                     values['-BLUR-'],
                     values['-CONTRAST-'],
                     values['-BRIGHTNESS-'],
                     values['-COLOR-'],
                     values['-SHARPNESS-'],
                     values['-EDGE-'],
                     values['-DETAIL-'],
                     values['-EMBOSS-'],
                     values['-CONTOUR-'],
                     values['-FLIPX-'],
                     values['-FLIPY-'],
                     values['-ROTATION-'],
                     values['-SCALE-'],
                     values['-INVERT-'],
                     values['-SEPIA-'],
                     values['-POSTERIZE-'],
                     values['-SOLARIZE-'],
                     values['-R_FACTOR-'],
                     values['-G_FACTOR-'],
                     values['-B_FACTOR-'])
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
