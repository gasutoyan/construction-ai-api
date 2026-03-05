from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import math

# 1. 保存したAIモデルの読み込み
data = joblib.load('construction_ai_model.joblib')
model = data['model']
features = data['features']
config = data['config']
staff_master = data['staff_master']

app = FastAPI(title="現場利益予測AI API")

# 2. リクエストのデータ形式定義
class DiagnosisRequest(BaseModel):
    site_name: str
    budget: int
    man_days: int
    distance_km: float
    mat_cost: int = 0
    staff_names: list[str]
    has_trouble: int = 0

# 3. 予測エンドポイント
@app.post("/predict")
def predict(req: DiagnosisRequest):
    try:
        # 初心者比率の算出
        beginner_scores = []
        for name in req.staff_names:
            if name not in staff_master:
                continue
            rank = staff_master[name]['ランク']
            score = 1.0 if rank == '初心者' else (0.5 if rank == '普通' else 0.0)
            beginner_scores.append(score)
        
        avg_beginner_ratio = sum(beginner_scores) / len(beginner_scores) if beginner_scores else 0.5

        # AI予測用のデータフレーム作成
        input_df = pd.DataFrame([[
            req.budget, req.man_days, req.distance_km, 
            req.mat_cost, avg_beginner_ratio, req.has_trouble
        ]], columns=features)

        # 予測実行
        pred_margin = model.predict(input_df)[0]
        pred_profit = int(req.budget * (pred_margin / 100))

        # 判定
        status = "✅ 優良" if pred_margin >= 25 else ("⚠️ 注意" if pred_margin >= 10 else "🚨 危険")

        return {
            "site_name": req.site_name,
            "predicted_profit": f"{pred_profit:,}円",
            "predicted_margin": f"{pred_margin:.1f}%",
            "judgment": status,
            "advice": "熟練工の配置により利益が安定します" if status == "✅ 優良" else "コストの見直しが必要です"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# サーバー起動確認用
@app.get("/")
def index():
    return {"message": "Construction AI Engine is Online"}