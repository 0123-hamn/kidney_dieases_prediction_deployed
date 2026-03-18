from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import os
from src.mlproject.pipelines.prediction_pipeline import CustomData, PredictPipeline
import uvicorn

app = FastAPI()

# Make sure the static directory exists
os.makedirs("static/css", exist_ok=True)
# Mount static files correctly
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_datapoint", response_class=HTMLResponse)
async def predict_datapoint(
    request: Request,
    age: float = Form(...),
    bp: float = Form(...),
    sg: str = Form(...),
    al: str = Form(...),
    su: str = Form(...),
    rbc: str = Form(...),
    pc: str = Form(...),
    pcc: str = Form(...),
    ba: str = Form(...),
    bgr: float = Form(...),
    bu: float = Form(...),
    sc: float = Form(...),
    sod: float = Form(...),
    pot: float = Form(...),
    hemo: float = Form(...),
    pcv: float = Form(...),
    wc: float = Form(...),
    rc: float = Form(...),
    htn: str = Form(...),
    dm: str = Form(...),
    cad: str = Form(...),
    appet: str = Form(...),
    pe: str = Form(...),
    ane: str = Form(...),
):
    try:
        # Convert values to match CustomData
        sg_val = float(sg)
        al_val = float(al)
        su_val = float(su)

        data = CustomData(
            age=age,
            bp=bp,
            sg=sg_val,
            al=al_val,
            su=su_val,
            rbc=rbc,
            pc=pc,
            pcc=pcc,
            ba=ba,
            bgr=bgr,
            bu=bu,
            sc=sc,
            sod=sod,
            pot=pot,
            hemo=hemo,
            pcv=pcv,
            wc=wc,
            rc=rc,
            htn=htn,
            dm=dm,
            cad=cad,
            appet=appet,
            pe=pe,
            ane=ane
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Format the result based on whatever the model returns
        # Assuming binary classification where 1 is CKD and 0 is not CKD
        # Map this to string text for the UI 
        if results[0] == 1 or results[0] == 'ckd':
            result_text = "Chronic Kidney Disease"
        else:
            result_text = "No Chronic Kidney Disease"

        return templates.TemplateResponse("index.html", {"request": request, "results": result_text})
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # You could also display an error string on the frontend
        return templates.TemplateResponse("index.html", {"request": request, "results": "Error processing prediction"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
