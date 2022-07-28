# imports
from PIL import Image, ImageFilter, ImageOps
import PySimpleGUI as sg
from io import BytesIO


# function that take the original image and the values from the buttons to update the image
def update_image(original, edge_enhance, detail, blur, contrast, emboss, contour, flipx, flipy):
    global image
    image = original.filter(ImageFilter.GaussianBlur(blur))
    image = image.filter(ImageFilter.UnsharpMask(contrast))
    if edge_enhance:
        image = image.filter(ImageFilter.EDGE_ENHANCE())
    if detail:
        image = image.filter(ImageFilter.DETAIL())
    if emboss:
        image = image.filter(ImageFilter.EMBOSS())
    if contour:
        image = image.filter(ImageFilter.CONTOUR())
    if flipx:
        image = ImageOps.mirror(image)
    if flipy:
        image = ImageOps.flip(image)
    bio = BytesIO()
    image.save(bio, format='PNG')

    win['-IMAGE-'].update(data=bio.getvalue())


# open file selection
image_path = sg.popup_get_file('Open', no_window=True)
# control layout
control_column = sg.Column([
    [sg.Frame('blur', layout=[[sg.Slider(range=(0, 10), orientation='h', key='-BLUR-')]])],
    [sg.Frame('contrast', layout=[[sg.Slider(range=(0, 10), orientation='h', key='-CONTRAST-')]])],
    [sg.Checkbox('detail', key='-DETAIL-'),sg.Checkbox('edge enhance',  key='-EDGE-')],
    [sg.Checkbox('emboss', key='-EMBOSS-'), sg.Checkbox('contour', key='-CONTOUR-')],
    [sg.Checkbox('flip x', key='-FLIPX-'), sg.Checkbox('flip y', key='-FLIPY-')],
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
while True:
    event, values = win.read(timeout=100)
    if event == sg.WIN_CLOSED:
        break
    # passing the buttons arguments in real time
    update_image(original,
                 values['-DETAIL-'],
                 values['-EDGE-'],
                 values['-BLUR-'],
                 values['-CONTRAST-'],
                 values['-EMBOSS-'],
                 values['-CONTOUR-'],
                 values['-FLIPX-'],
                 values['-FLIPY-'])
    # if user pressed save button
    if event == '-SAVE-':
        file_path = sg.popup_get_file('Save', save_as=True, no_window=True) + '.png'
        image.save(file_path, 'PNG')

win.close()
