from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import os
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model and label encoders
model_path = "random_forest_modell.pkl"
label_encoders_path = "label_encoderss.pkl"

# Ensure model and encoders are loaded properly
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

if os.path.exists(label_encoders_path):
    with open(label_encoders_path, "rb") as f:
        label_encoders = pickle.load(f)
else:
    label_encoders = None

# Initialize Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")


# Function to safely encode unseen categories
def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Handle unseen categories by returning the most frequent class or defaulting to the first class
        print(f"Warning: Unseen value '{value}' detected for encoder. Defaulting to '{encoder.classes_[0]}'.")
        return encoder.transform([encoder.classes_[0]])[0]


# Route to render the index.html form
@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route to handle form submission and prediction
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    last_percentage: float = Form(...),
    attendance: float = Form(...),
    parents_qualification: str = Form(...),
    area: str = Form(...),
    num_teachers: int = Form(...),
    income: float = Form(...),
):
    if model is not None and label_encoders is not None:
        try:
            # Safely encode categorical features
            parents_qualification_encoded = safe_encode(label_encoders["parents_qualification"], parents_qualification)
            area_encoded = safe_encode(label_encoders["area"], area)

            # Prepare the input for the model
            input_features = [[
                last_percentage,
                attendance,
                parents_qualification_encoded,
                area_encoded,
                num_teachers,
                income,
            ]]

            # Predict dropout status
            probabilities = model.predict_proba(input_features)

            dropout_probability = probabilities[0][1]

            # Render result.html with the prediction result
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "dropout_probability": round(dropout_probability, 3),
                    "parents_qualification": parents_qualification,
                    "area": area,
                    "attendance": attendance,
                    "last_percentage": last_percentage,
                    "income": income,
                    "num_teachers":num_teachers
                },
            )

        except Exception as e:
            print(f"Error during prediction: {e}")
            return HTMLResponse(content=f"Error during prediction: {e}", status_code=500)
    else:
        return HTMLResponse(content="Model or encoders not loaded!", status_code=500)
