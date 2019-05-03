import pickle
import src

def test_init_checkpoint():
    src.main(FREQ=1)
    checkpoint = "checkpoint_name.pkl"
    with open(checkpoint, "r") as cp_file:
        cp = pickle.load(cp_file)

    start_gen = cp["generation"]
    assert(start_gen) == 2

def test_ini_checkpoint():
    population, logbook = src.main(checkpoint="checkpoint_name.pkl", FREQ=1)
    assert(logbook[-1]['gen']) == 2


import win32com.client as com
import Tkinter
import tkFileDialog

Tkinter.Tk().withdraw()
in_path = tkFileDialog.askopenfilename()
path = in_path.replace("/", "\\")

Vissim = com.Dispatch('Vissim.Vissim')
Vissim.LoadNet(path)