# 📊 Promotional analysis - Step by step for Dashboard

This repository contains a Streamlit app for analyzing promotional performance (discounts, sales uplift, SKU & group analysis, compliments, etc.) using a Kaggle-hosted dataset.

---

## 🔧 Prerequisites

- Python 3.7+  
- A Kaggle account with an API token  
- Basic familiarity with the command line

---

## ✅ Step 1: Install Required Libraries

```bash
pip install kaggle pandas streamlit plotly
```

---

## ✅ Step 2: Get Your Kaggle API Token

1. Sign in to your Kaggle account and go to:  
   https://www.kaggle.com/account  
2. Scroll down to the **API** section.  
3. Click **Create New API Token**.  
4. A file named `kaggle.json` will download. Keep this safe—it contains your credentials.

---

## ✅ Step 3: Place `kaggle.json` in the Right Location

### macOS / Linux

```bash
mkdir -p ~/.kaggle
mv /path/to/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Windows

1. Open File Explorer and navigate to your user folder:  
   `C:\Users\<YourUsername>\`  
2. Create a subfolder named `.kaggle`.  
3. Move `kaggle.json` into:  
   `C:\Users\<YourUsername>\.kaggle\kaggle.json`

> **Alternative:**  
> You can also point Streamlit to a custom location by setting an environment variable at the top of your script:
> ```python
> import os
> os.environ['KAGGLE_CONFIG_DIR'] = '/path/to/your/.kaggle'
> ```

---

## ✅ Step 4: Download the Dataset

Use the Kaggle CLI to download and unzip your dataset into a local `data/` folder.

```bash
# Replace <username>/<dataset-name> with your dataset slug
kaggle datasets download -d <username>/<dataset-name> -p data --unzip
```

**Example:**

```bash
kaggle datasets download -d shivamb/netflix-shows -p data --unzip
```

After this step, you should have your CSV file (e.g. `filtered_data1.csv`) in the `data/` directory.

---

## ✅ Step 5: Configure the Script

1. Open `app.py` in your code editor.  
2. At the top of the file, update these constants to match your dataset:

```python
DATASET_SLUG = "your-username/your-dataset"   # for reference
CSV_FILENAME = "filtered_data1.csv"           # your CSV filename
LOCAL_DIR    = "data"                         # folder where your CSV lives
```

---

## ✅ Step 6: Run the Dashboard

Launch the Streamlit app with:

```bash
streamlit run app.py
```

Point your browser to:

```
http://localhost:8501
```

You can now interactively explore:

- **SKU Analysis**  
- **Group Analysis**  
- **Compliment Analysis**  
- **Discount, Sales and Uplift**  
- **(Plus any additional views you add!)**

---

## 🛠 Troubleshooting

- If you see `ModuleNotFoundError: No module named 'streamlit'`, ensure you installed the library into your active environment.
- Check that `kaggle.json` permissions are correct (`chmod 600` on Unix).
- Verify the path to your CSV in `app.py` matches where you unzipped the data.

---
