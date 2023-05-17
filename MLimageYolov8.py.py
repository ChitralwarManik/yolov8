#pip install ultralytics
import os
import time

from ultralytics import YOLO
import sys
# Load a model
import sys, os
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    configString = sys.argv[1]

    #InstalledLocation##im$$imageLocation

    configs = configString.split("##im$$")

    modelPath = os.path.join(configs[0],"All21PII.pt")

    imagePath = configs[1]

    #"D:\ML MODEL\ImageToPickelEXE##im$$D:\TEST_UAT\PII PCI combined_Agent\MLImageScan\Kavita pan.jpg"
    # names = ['AadharCard', 'AmericanExpress', 'DinersClub', 'Discover', 'JCB', 'Maestro', 'MasterCard', 'PANCard', 'Rupay', 'UnionPay', 'VISA', 'VoterId']
    names = ['AadharCard', 'AmericanExpress', 'CanaraBankLOGO', 'DinersClub', 'Discover', 'DrivingLicense', 'IndianPassport', 'JCB', 'Maestro', 'MasterCard', 'NINO -UK-', 'NRIC -Singapore-', 'National ID -Saudi Arabia-', 'National ID -UAE-', 'PANCard', 'Rupay', 'SSN', 'UnionPay', 'VISA', 'VoterId']
    # for i in range(100):
    model = YOLO(modelPath)
        # time.sleep(2)

    result = model(imagePath)
    boxes = result[0].boxes
    Clsses = []
    for ele in boxes.cls:
        Clsses.append(names[int(ele)])

    confidences = []
    for con in boxes.conf:
        confidences.append(float(con))

    res = dict(zip(confidences, Clsses))
    if len(res) == 0:
        sys.stdout.write("$*$" + "Class Not Found"+ "$*$")
    else:
        for key, value in res.items():
            sys.stdout.write("$*$" + str(value)+ ':'+ str(key) + "$*$")