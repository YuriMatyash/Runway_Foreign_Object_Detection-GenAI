# GenAI - Runway Foreign Object Detection


**DataSet Link:**  
https://drive.google.com/drive/folders/1yT7Xh2XZuWhQT9_ujwvSrzlyq6i68UFk?usp=sharing  
To setup the dataset for training put the "dataset" folder with it's contents inside the cloned repo.  

The README is divided into sections where in each section we go into depth about each model:
* [**📷 Dataset Generation**](#dataset)
* [**📊 Exploratory Data Analysis (EDA)**](#eda)
* [**🛠️ Data Augmentations**](#augmentations)
* [**🤖 Model Selection and Training**](#training)
* [**📈 Evaluation**](#evaluation)
* [**💯 End Summary**](#summary)

## 📖 Project Overview
This project focuses on developing an automated Foreign Object Debris (FOD) detection system for airport runways using state-of-the-art Computer Vision. Detecting hazards like oil spills, ice chunks, and potholes is critical for aviation safety, but acquiring high-quality, labeled satellite or aerial imagery of these specific defects is often difficult and expensive.

To solve this, we leverage Generative AI (Stable Diffusion XL) to create a massive, high-fidelity synthetic dataset. By using a controlled "Base Layout" approach, we mathematically define the exact location of defects ensuring 100% accurate YOLO labels while the AI handles the complex task of rendering realistic textures like weathered asphalt, granular soil, and translucent ice.

**Core Objectives**
* **Synthetic Data Pipeline:** A custom-built pipeline that generates randomized runway environments, including grass/dirt shoulders, tire skid marks, and varied markings, and injects realistic anomalies.  

* **High-Resolution Detection:** Training a YOLOv26 model to identify small-scale hazards in high-resolution aerial views.

* **Safety-Critical Accuracy:** Achieving high precision and recall to ensure that even the most subtle oil puddles or ice fragments are flagged before they pose a risk to aircraft.

**Target Classes**
* **Oil Puddles:** Viscous black puddles with realistic liquid reflections and varied spill shapes.

* **Ice Chunks:** Solid, jagged blocks of frozen debris with translucent white and blue textures.

* **Potholes/Sinkholes:** Deep structural failures in the asphalt showing jagged edges and exposed granular soil or voids.

## <a id="dataset"></a>📷 Dataset Generation
The dataset is built using a proprietary synthetic pipeline that bridges procedural computer graphics with Stable Diffusion XL (SDXL). This hybrid approach solves the "Data Scarcity" problem in aviation safety by generating high-fidelity, diverse imagery that would be impossible or cost-prohibitive to capture in the real world. By mathematically defining the environment before the generative step, we achieve pixel-perfect label accuracy without the need for manual human annotation.


**Step 1:** Basic Layout (Procedural Environment)  
The pipeline first constructs a clean runway environment using OpenCV. It randomizes asphalt gray-tones to simulate varying levels of weathering. To prevent model over-fitting, we procedurally generate grass or dirt shoulders (5-15% image width) and inject "burnt rubber" tire skid marks that serve as texture anchors for the AI.  
<img src="README_materials/oil_example_step1_gray.png" width="300" height="300" alt="Alt Text">  

**Step 2:** Object Injection (Geometric Proxies)  
We calculate the spatial coordinates for the anomaly and inject a high-contrast geometric proxy. For Oil Spills, we use randomized ellipses; for Ice Chunks and Potholes, we generate jagged, irregular polygons. These proxies act as a "roadmap" for the diffusion model, telling it exactly where to focus the generative detail.  
<img src="README_materials/oil_example_step2_layout.png" width="300" height="300" alt="Alt Text">  

**Step 3:** Image Generation (AI Refinement) 
The low-fidelity layout is processed through the SDXL Img2Img pipeline. Using a controlled AI_STRENGTH (0.55 - 0.65), the AI "hallucinates" realistic textures over our proxies. It transforms the shapes into viscous liquid pools with reflections, translucent frozen blocks, or jagged asphalt collapses with granular soil textures.  
<img src="README_materials/oil_example_step3_final.png" width="300" height="300" alt="Alt Text">  

**Step 4:** Labeling (YOLO Mapping)  
Because the object's position was mathematically defined in Step 2, the bounding box is already known. The system automatically maps these coordinates to the final image, generating a normalized YOLO .txt file. This ensures the training labels are perfectly aligned with the visual anomalies, eliminating "label noise" entirely.  
<img src="README_materials/oil_example_step4_debug.png" width="300" height="300" alt="Alt Text">  

## <a id="eda"></a>📊 Exploratory Data Analysis (EDA)

## <a id="augmentations"></a>🛠️ Data Augmentations
The following augmentations are applied during training to ensure the model generalizes across diverse runway conditions, lighting environments, and flight altitudes.

| Augmentation | Explanation | Rationale | Metric |  
| :--- | :--- | :--- | :--- |
| Mosaic | Combines four training images into one at different scales.| Essential for FOD: Forces the model to recognize small debris (oil, ice) within complex, varied backgrounds. | 1.0 (Enabled) |  
| Resize | Scales input images to a fixed square dimension. | Standardizes input for the YOLO architecture while maintaining normalized coordinate accuracy. | 640 × 640 pixels |  
| Random Rotation | Rotates the image randomly within a specified range. | Simulates varying flight paths or satellite sweep angles where the runway is not perfectly vertical. | ± 15.0 Degrees |  
| Vertical/Horizontal Flip | Mirrors the image along the X or Y axis. | Satellite/Aerial views are orientation-independent; a pothole is valid from any direction. | p = 0.5 |  
| HSV Jitter | Randomly adjusts Hue, Saturation, and Value (Brightness). | Simulates different times of day (harsh midday sun vs. evening) and weathered asphalt color shifts. | H=0.015, S=0.7, V=0.4 |  
| MixUp | Overlays two images with varying transparency. | Regularizes the model by preventing over-reliance on specific pixel-perfect edges, improving generalization. | p = 0.1 |  
| Scale & Translate | Zooms and shifts the image randomly. | Simulates different camera altitudes and framing offsets from the aircraft/UAV center. | Scale=0.5, Trans=0.1 |  

## <a id="training"></a>🤖 Model Selection and Training

## <a id="evaluation"></a>📈 Evaluation

## <a id="summary"></a>💯 End Summary