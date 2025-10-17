# Label Studio Setup and Custom Vision Model Training

This guide provides step-by-step instructions for setting up **Label Studio**, annotating datasets, and training a **custom Vision model** using Oracle Cloud Infrastructure (OCI Vision).

---

## Step 1: Setup Label Studio

### 1. Create a Python Virtual Environment
```bash
python3 -m venv env
source env/bin/activate
```

### 2. Install Label Studio
```bash
python -m pip install label-studio
```

---

## Step 2: Run Label Studio

After installation, launch Label Studio by running:
```bash
label-studio
```

Once it starts, open the provided local URL (typically `http://localhost:8080`) in your browser.

---

## Step 3: Annotate Dataset

1. Create a new **project** in Label Studio.  
2. Import your **images** or dataset.  
3. Define the labeling interface (bounding boxes, polygons, etc.).  
4. Annotate the images as required.  
5. **Export** the labeled data as a **JSON file** (`exported.json`).

---

## Step 4: Upload Data to OCI Bucket

Upload both:
- Your **images**  
- The **exported JSON annotation file**  

to your designated OCI Object Storage **bucket**.

---

## Step 5: Generate JSONL for Vision Training

Run the provided script to process the annotated JSON and generate the required **`od.jsonl`** file in the same bucket.

```bash
python generate_jsonl.py
```

Ensure that the resulting `od.jsonl` file and images reside in the same location (bucket path) for OCI Vision training.

---

## Step 6: Train Custom Model in OCI Vision

1. Open the **OCI Vision Console**.  
2. Create a new **Custom Model Training Job**.  
3. Select the generated `od.jsonl` file as your **training dataset**.  
4. Configure model type (Object Detection / Image Classification).  
5. Start training and monitor progress.

---

## Detailed Documentation

For an in-depth walkthrough and OCI Vision configuration details, refer to the official document:  
🔗 **[Vision Service Documentation](<Insert_Link_Here>)**
