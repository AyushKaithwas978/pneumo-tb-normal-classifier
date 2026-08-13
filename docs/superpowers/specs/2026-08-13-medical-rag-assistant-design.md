# Medical Image RAG Assistant Design

## Scope

Upgrade the copied Pneumo/TB/Normal classifier into an offline medical image assistant for educational use. This phase covers:

- richer classifier result presentation
- a small local knowledge base for Normal, Pneumonia, and Tuberculosis
- deterministic retrieval and explanation generation without paid API keys

The original `pneumo-tb-normal-classifier` directory remains unchanged.

## Architecture

The Flask app keeps the existing upload and prediction flow. After `predict_probs()` returns the top label, confidence, and probabilities, a new local assistant module retrieves relevant knowledge snippets and builds a structured explanation.

The new module owns:

- loading Markdown knowledge files
- ranking sections using lightweight keyword scoring
- building an explanation payload with summary, clinical context, suggested next steps, follow-up questions, and citations

The Flask layer owns:

- saving uploaded images
- calling the TensorFlow model
- passing explanation data into the HTML template
- adding the same explanation payload to the JSON API

## Data Flow

1. User uploads a JPG or PNG.
2. Flask saves the file under `static/uploads`.
3. TensorFlow model predicts `Normal`, `Pneumonia`, or `Tuberculosis`.
4. The assistant retrieves knowledge snippets for the predicted class plus safety guidance.
5. The UI displays probabilities and a cited educational explanation.
6. `/predict` returns probabilities, model version, explanation, suggested questions, and citations.

## Safety

The assistant must never present the output as a diagnosis. Every explanation includes a disclaimer and a recommendation to consult a qualified clinician, especially for urgent symptoms or high-confidence abnormal predictions.

## Testing

Use standard-library `unittest` tests for the retrieval and explanation module so the project does not need extra test dependencies. The tests cover:

- retrieval returns relevant snippets for pneumonia queries
- explanation includes prediction context, citations, next steps, and the disclaimer
- unknown labels still produce a safe fallback explanation
