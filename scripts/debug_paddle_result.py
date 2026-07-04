"""Quick diagnostic: inspect PaddleOCR detection result structure."""
import os
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddlex.inference.models import create_predictor
import json

print("Loading model...")
det_predictor = create_predictor(
    model_name="PP-OCRv6_medium_det",
    model_dir="paddle_ocr_weights/PP-OCRv6_medium_det_infer",
)

print("Running inference on img.png...")
results = list(det_predictor.predict("img.png"))
if not results:
    print("No results returned")
    exit()

result = results[0]
print("\n--- result type ---")
print(type(result))
print("\n--- result attributes ---")
print([a for a in dir(result) if not a.startswith('_')])

if hasattr(result, 'json'):
    print("\n--- result.json ---")
    print(result.json)
    print("\n--- result.json keys ---")
    print(list(result.json.keys()))
    for k, v in result.json.items():
        print(f"\nkey '{k}' type: {type(v)}")
        if isinstance(v, list) and len(v) > 0:
            print(f"  first element type: {type(v[0])}")
            print(f"  first element: {v[0]}")

if hasattr(result, 'res'):
    print("\n--- result.res ---")
    print(type(result.res))
    print(result.res)
    if isinstance(result.res, list) and len(result.res) > 0:
        print(f"\n  first element type: {type(result.res[0])}")
        print(f"  first element: {result.res[0]}")
