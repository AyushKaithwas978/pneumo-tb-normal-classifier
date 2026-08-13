# Chest X-ray RAG Assistant

Local Flask app to upload a chest X-ray, return probabilities for Normal, Pneumonia, and Tuberculosis, and generate a cited educational explanation from a local Markdown knowledge base.

This is an upgraded copy of the original classifier project. It runs offline and does not require an LLM API key.

## Features

- Chest X-ray upload UI
- TensorFlow/Keras image classification
- Probabilities for Normal, Pneumonia, and Tuberculosis
- Local retrieval over `knowledge_base/*.md`
- Structured assistant explanation with clinical context, next steps, suggested questions, citations, and safety disclaimer
- JSON prediction API

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
  "model_version": "model_inception.h5",
  "explanation": {
    "summary": "The model's top prediction is Pneumonia with 83.00% confidence.",
    "clinical_context": ["..."],
    "next_steps": ["..."],
    "suggested_questions": ["..."],
    "citations": [
      {
        "source": "pneumonia.md",
        "title": "Pneumonia - Overview"
      }
    ],
    "disclaimer": "This assistant is for educational use only..."
  }
}
```

## Tests

Run the offline assistant tests:

```
python -m unittest tests.test_rag_assistant
```

## Notes

This project is for educational use only and is not a medical diagnosis tool. A qualified clinician should review symptoms, history, examination findings, and imaging before medical decisions are made.
