# Table Detection and Extraction

This project automatically detects and extracts tables from scanned images or documents.  
It uses **Microsoft Table Transformer** for table detection and **PaddleOCR (PPStructureV3)** for table recognition.  
The results are saved as cropped images and Excel files for easy use.  

---

## ✨ Features
- Detects tables in input images with `microsoft/table-transformer-detection`.
- Crops detected tables and saves them as `.png` images.
- Expands bounding boxes slightly for better cropping accuracy.
- Parses tables using `PaddleOCR (PPStructureV3)`.
- Exports parsed results as Excel files (`.xlsx`).
- Supports **GPU acceleration** if available, otherwise runs on CPU.

---

## ⚙️ Requirements

- Python **3.8+**
- Install dependencies:

```bash
pip install -r requirements.txt
```
## 🚀 Usage
1. Place your input image (e.g., `sample.png`) inside the `input/` folder.  
2. Update `main.py` with the correct paths:  
   ```python
   INPUT_IMG = r"input/sample.png"
   OUT_DIR   = r"output"
## Run the script 
    python main.py

## 📝 Example Output
- Cropped tables:
  - `output/table_00.png`
  - `output/table_01.png`
- Extracted Excel files:
  - `output/table_00.xlsx`
  - `output/table_01.xlsx`

## ⚡ Configuration Options
- `DET_THRESH` → Confidence threshold (default `0.7`)
- `EXPAND_CM` → Expand box size in cm (default `0.30`)
- `DEFAULT_DPI` → Default DPI if missing (default `300`)
