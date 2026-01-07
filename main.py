# imports
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import PySimpleGUI as sg
from io import BytesIO

# function that take the original image and the values from the buttons to update the image
def update_image(original, blur, contrast, brightness, color, sharpness, edge_enhance, detail, emboss, contour, flipx, flipy, rotation):
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
                     values['-ROTATION-'])
        prev_values = values

    # if user pressed save button
    if event == '-SAVE-':
        save_path = sg.popup_get_file('Save', save_as=True, no_window=True)
        if save_path:
            image.save(save_path + '.png', 'PNG')

win.close()
