# Binary Classification with Flask and PyTorch

This project implements a binary image classification system using a Convolutional Neural Network (CNN) in PyTorch. The trained model is integrated into a Flask application for inference with a simple HTML frontend.

---

## Project Structure

```
├── app.py                 # Flask backend (serving model + UI)
├── requirements.txt       # Dependencies
├── model_train_code/  
│   └── train.py           # Training script for CNN model
├── templates/             # HTML templates for Flask frontend
│   └── index.html         # Example UI page
└── saved_model.pth        # Trained model (generated after training)
```

---

## Tech Stack

- Python 3.x  
- Flask – Web framework  
- PyTorch – Deep learning framework  
- Torchvision – Dataset utilities and transforms  
- Kaggle API – Dataset download  
- HTML/CSS – Frontend templates  

---

## Model Details

- Architecture: Convolutional Neural Network (CNN)  
- Optimizer: Adam  
- Loss Function: BCEWithLogitsLoss  
- Training Epochs: 15  
- Task: Binary image classification  

---

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd <your-project>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up Kaggle API credentials (optional if dataset is included):  
Place `kaggle.json` in `~/.kaggle/`.

---

## Training the Model

The training script will:  
- Download the dataset from Kaggle  
- Extract the `.zip` file  
- Create train/test DataLoaders  
- Train the CNN model for 15 epochs  
- Save the trained model as `saved_model.pth`  

Run training:

```bash
cd model_train_code
python train.py
```

After training, `saved_model.pth` will be created in the project root.

---

## Running the Flask App

Start the Flask server:

```bash
python app.py
```

Access the web application at:

```
http://127.0.0.1:5000/
```

---

## Example Workflow

1. Train the model with `train.py` (saves `saved_model.pth`).  
2. Run Flask app with `app.py` (loads the trained model).  
3. Upload an image via the HTML form (`index.html`).  
4. The model performs inference and displays the result in the browser.  

---

## Requirements

Example `requirements.txt`:

```
flask
torch
torchvision
pandas
numpy
Pillow
kaggle
```

---

## Future Improvements

- Add GPU support for faster training  
- Deploy the Flask app on Render/Heroku  
- Extend to multi-class classification  
- Add Docker support  
