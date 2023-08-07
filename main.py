import whisper
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = whisper.load_model("base", device=device)

result = model.transcribe("02 - Installation.mp4", verbose=True)

segments = result["segments"]

text = "".join(["->" + str(s["start"]) + "--" + str(s["end"]) + ":>" + s["text"] + "\n" for s in segments])

with open("02 - Installation.txt","w") as f:
    f.write(text)
