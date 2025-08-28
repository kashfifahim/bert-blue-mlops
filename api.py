import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import glob

app = FastAPI(title="BERT GLUE Model API")

# Load all models
models = {}
model_files = glob.glob("models/*")
for model_path in model_files:
    if os.path.isdir(model_path) and not model_path.endswith("_metrics.txt"):
        task_name = os.path.basename(model_path).split("_")[0] + "/" + os.path.basename(model_path).split("_")[1]
        models[task_name] = tf.saved_model.load(model_path)
        print(f"Loaded model for task: {task_name}")

class TextInput(BaseModel):
    text1: str
    text2: Optional[str] = None
    task: str

@app.post("/predict")
async def predict(input_data: TextInput):
    if input_data.task not in models:
        raise HTTPException(status_code=400, detail=f"Model for task {input_data.task} not found")
    
    model = models[input_data.task]
    
    # Prepare inputs based on task
    if input_data.task in ["glue/cola", "glue/sst2"]:
        inputs = tf.constant([input_data.text1])
        result = model(inputs)
    else:
        if not input_data.text2:
            raise HTTPException(status_code=400, detail=f"Task {input_data.task} requires text2 input")
        inputs = [tf.constant([input_data.text1]), tf.constant([input_data.text2])]
        result = model(inputs)
    
    # Process results
    probabilities = tf.nn.softmax(result).numpy().tolist()
    predicted_class = tf.argmax(result, axis=1).numpy().tolist()[0]
    
    return {
        "task": input_data.task,
        "predicted_class": predicted_class,
        "probabilities": probabilities[0]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)