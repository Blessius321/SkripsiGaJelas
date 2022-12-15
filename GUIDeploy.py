import PySimpleGUI as sg
from  Deployment.TrackingDualGUI import *
import Deployment.guiDefinition as gui

def main():
    while True:
        event, values = gui.window.Read(timeout=5)
        #reset gaze
        for i in range(0,4):
            gui.window[f"-{i+1}-"].update(button_color = ("white", "#C0EEE4"))
        gui.window["-KANAN BAWAH-"].update(button_color = ("white", "#C0EEE4"))
        gui.window["-KIRI BAWAH-"].update(button_color = ("white", "#C0EEE4"))
        

        if event == None:
            pass
        if event == sg.WIN_CLOSED: 
            break
        # Ganti Page 
        if event == "-KANAN BAWAH-":
            gui.page = 2

        if event == "-KIRI BAWAH-":
            gui.page = 1
        # Tombol
        if event in gui.buttons:
            print(event)
            gui.buttonClicked[int(event[1]) -1] = not gui.buttonClicked[int(event[1])- 1]

        #ubah tampilan
        if gui.page == 1:
            gui.window['-pageDua-'].update(visible = False)
            gui.window['-pageSatu-'].update(visible = True)
        elif gui.page == 2:
            gui.window['-pageSatu-'].update(visible = False)
            gui.window['-pageDua-'].update(visible = True)

        for (i, clicked) in enumerate(gui.buttonClicked):
            gui.window[f'-{i+1}-'].update(button_color = ("white", "#C0EEE4") if not clicked else ("black", "#FF9E9E"))

        gaze = findGaze()
        
        if gaze == 0:
            gui.window[f"-{1 if gui.page == 1 else 3 }-"].update(button_color = ("black", "#F8F988"))
        elif gaze == 1:
            gui.window[f"-{2 if gui.page == 1 else 4 }-"].update(button_color = ("black", "#F8F988"))
        elif gaze == 2:
            gui.window["-KIRI BAWAH-"].update(button_color = ("black", "#F8F988"))
        elif gaze == 3:
            gui.window["-KANAN BAWAH-"].update(button_color = ("black", "#F8F988"))
            

    gui.window.close()
    




if __name__ == "__main__":
    main()