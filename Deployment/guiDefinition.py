import PySimpleGUIQt as sg

pageSatu = [
    [
        sg.Button(image_filename= 'kepala.png', enable_events=True, key="-1-", size_px=(540,360), button_color = ("white", "blue")),
        sg.VSeperator(), 
        sg.Button(image_filename= 'mata.png', enable_events=True, key="-2-", size_px=(540,360), button_color = ("white", "blue"))
    ]
]
pageDua = [
    [
        sg.Button(image_filename= '', enable_events=True, key="-3-", size_px=(540,360), button_color = ("white", "blue")),
        sg.VSeperator(), 
        sg.Button(image_filename= 'touch.png', enable_events=True, key="-4-", size_px=(540,360), button_color = ("white", "blue"))
    ],
]



layout = [
    [
        sg.Column(pageSatu, key='-pageSatu-', visible=True), sg.Column(pageDua, key='-pageDua-', visible= False)
    ],
    [
        sg.HSeperator()
    ],
    [
        sg.Button(button_text="KIRI", enable_events=True, key="-KIRI BAWAH-", size_px=(540,360), button_color = ("white", "blue")),
        sg.VSeperator(), 
        sg.Button(button_text="KANAN", enable_events=True, key="-KANAN BAWAH-", size_px=(540,360), button_color = ("white", "blue"))
    ],
]

window = sg.Window(title="TEST", layout=layout, size=(1080, 720), finalize=True)
buttonClicked = [False, False, False, False]
buttons = ["-1-", '-2-', '-3-', '-4-']
page = 1
