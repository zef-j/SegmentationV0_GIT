timeAvailable = 10

lastNumEpoch = 20
lastTime = 2
timePerEpoch = (lastTime*60)/lastNumEpoch

numberEpochEstimate = (timeAvailable*60) / timePerEpoch
print(f"Estimated {timePerEpoch} minutes per epoch")

print(f"For {timeAvailable} Hours available :\naim for {numberEpochEstimate} epochs")