# Pneumo/TB/Normal Classifier

Local Flask app to upload a chest X-ray and return probabilities for Normal, Pneumonia, and Tuberculosis.

## Quick start

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Ensure the model file is available:

- Default path: `models/model_inception.h5`
- Or set `MODEL_PATH` to point to your model
- The app also auto-detects the legacy model in `Computer Vision/Transfer Learning/model_inception.h5`

3. Run the app:

```
python app.py
```

Open `http://localhost:8080` in your browser.

## API

`POST /predict` with `multipart/form-data` and field `file`.

Response:

```
{
  "predicted_class": "Pneumonia",
  "confidence": 0.83,
  "probabilities": {
    "normal": 0.05,
    "pneumonia": 0.83,
    "tuberculosis": 0.12
  },
  "model_version": "model_inception.h5"
}
```

## Notes

This project is for educational use only and is not a medical diagnosis tool.
