# imports
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import PySimpleGUI as sg
from io import BytesIO

# function that take the original image and the values from the buttons to update the image
def update_image(original, blur, contrast, brightness, color, sharpness, edge_enhance, detail, emboss, contour, flipx, flipy, rotation, scale, invert, sepia):
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
    [sg.Checkbox('Detail', key='-DETAIL-'), sg.Checkbox('Edge Enhance', key='-EDGE-')],
    [sg.Checkbox('Emboss', key='-EMBOSS-'), sg.Checkbox('Contour', key='-CONTOUR-')],
    [sg.Checkbox('Invert', key='-INVERT-'), sg.Checkbox('Sepia', key='-SEPIA-')],
]

adjustments_tab = [
    [sg.Frame('Contrast', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-CONTRAST-')]])],
    [sg.Frame('Brightness', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-BRIGHTNESS-')]])],
    [sg.Frame('Color', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-COLOR-')]])],
    [sg.Frame('Sharpness', layout=[[sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', key='-SHARPNESS-')]])],
]

transform_tab = [
    [sg.Checkbox('Flip X', key='-FLIPX-'), sg.Checkbox('Flip Y', key='-FLIPY-')],
    [sg.Frame('Rotation', layout=[[sg.Slider(range=(0, 360), default_value=0, orientation='h', key='-ROTATION-')]])],
    [sg.Frame('Scale %', layout=[[sg.Slider(range=(10, 200), default_value=100, orientation='h', key='-SCALE-')]])],
]

control_column = sg.Column([
    [sg.TabGroup([
        [sg.Tab('Effects', effects_tab), sg.Tab('Adjustments', adjustments_tab), sg.Tab('Transform', transform_tab)]
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
                     values['-SEPIA-'])
        prev_values = values

    # if user pressed save button
    if event == '-SAVE-':
        save_path = sg.popup_get_file('Save', save_as=True, no_window=True)
        if save_path:
            image.save(save_path + '.png', 'PNG')

win.close()
