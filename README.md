# Banana Spot Project

YOLO-based banana black spot detection for estimating ripeness and sweetness from uploaded or camera-captured images.

## Features

- Detects whole bananas (`all`) and black spots (`s`) with YOLO.
- Calculates black spot percentage from detected spot area and banana area.
- Classifies black spot index:
  - `< 5%`: low
  - `5% - 15%`: medium
  - `>= 15%`: high
- Shows ripeness and sweetness guidance in the Streamlit app.

## Run Streamlit

```powershell
C:\Users\user\miniconda3\envs\banana\python.exe -m streamlit run frontend_streamlit.py --server.port 8501
```

Then open:

```text
http://localhost:8501
```
