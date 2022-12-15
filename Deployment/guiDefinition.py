import PySimpleGUI as sg

pageSatu = [
    [
        sg.Button(image_filename= 'Assets/kepala.png', enable_events=True, key="-1-", button_color = ("white", "#C0EEE4")),
        sg.VSeperator(), 
        sg.Button(image_filename= 'Assets/mata.png', enable_events=True, key="-2-", button_color = ("white", "#C0EEE4"))
    ]
]
pageDua = [
    [
        sg.Button(image_filename= 'Assets/suara.png', enable_events=True, key="-3-", button_color = ("white", "#C0EEE4")),
        sg.VSeperator(), 
        sg.Button(image_filename= 'Assets/touch.png', enable_events=True, key="-4-", button_color = ("white", "#C0EEE4"))
    ],
]



layout = [
    [
        sg.Column(pageSatu, key='-pageSatu-', visible=True), sg.Column(pageDua, key='-pageDua-', visible= False)
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Button(image_filename="Assets/kiri.png", enable_events=True, key="-KIRI BAWAH-", button_color = ("white", "#C0EEE4")),
        sg.VSeperator(), 
        sg.Button(image_filename="Assets/kanan.png", enable_events=True, key="-KANAN BAWAH-",  button_color = ("white", "#C0EEE4"))
    ],
]

window = sg.Window(title="TEST", layout=layout, finalize=True, no_titlebar=True, keep_on_top=False)
window.maximize()
buttonClicked = [False, False, False, False]
buttons = ["-1-", '-2-', '-3-', '-4-']
page = 1
