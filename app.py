from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import random

# 1. 保存したAIモデルの読み込み
data = joblib.load('construction_ai_model.joblib')
model = data['model']
features = data['features']
staff_master = data['staff_master']

app = FastAPI(title="現場利益予測AI")

# 仮の従業員データ（10名）
DUMMY_STAFF = [
    "佐藤", "鈴木", "高橋", "田中", "伊藤", 
    "渡辺", "山本", "中村", "小林", "加藤"
]

class DiagnosisRequest(BaseModel):
    site_name: str
    budget: int
    man_days: int
    distance_km: float = 30.0  # デフォルト値
    mat_cost: int = 0         # デフォルト値
    staff_names: list[str] = []
    has_trouble: int = 0

@app.post("/predict")
def predict(req: DiagnosisRequest):
    try:
        # 従業員が指定されていない場合はランダムに5名選抜（シミュレーション用）
        selected_staff = req.staff_names if req.staff_names else random.sample(DUMMY_STAFF, 5)
        
        beginner_scores = []
        for name in selected_staff:
            rank = staff_master.get(name, {'ランク': '普通'})['ランク']
            score = 1.0 if rank == '初心者' else (0.5 if rank == '普通' else 0.0)
            beginner_scores.append(score)
        
        avg_beginner_ratio = sum(beginner_scores) / len(beginner_scores)

        input_df = pd.DataFrame([[
            req.budget, req.man_days, req.distance_km, 
            req.mat_cost, avg_beginner_ratio, req.has_trouble
        ]], columns=features)

        pred_margin = model.predict(input_df)[0]
        pred_profit = int(req.budget * (pred_margin / 100))
        status = "✅ 優良" if pred_margin >= 25 else ("⚠️ 注意" if pred_margin >= 10 else "🚨 危険")

        return {
            "site_name": req.site_name,
            "predicted_profit": f"{pred_profit:,}円",
            "predicted_margin": f"{pred_margin:.1f}%",
            "judgment": status,
            "advice": "この条件なら利益がしっかり出ます！" if status == "✅ 優良" else "人件費か材料費を抑える工夫が必要です。"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>現場利益予測AI</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { background-color: #f0f2f5; font-family: sans-serif; }
            .card { background: white; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }
            .input-label { font-size: 0.875rem; font-weight: 700; color: #4b5563; margin-bottom: 4px; }
            .input-field { width: 100%; border: 2px solid #e5e7eb; padding: 12px; border-radius: 10px; transition: border-color 0.2s; }
            .input-field:focus { border-color: #2563eb; outline: none; }
            .btn-submit { background: #2563eb; color: white; padding: 16px; border-radius: 12px; width: 100%; font-weight: 800; font-size: 1.1rem; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2); }
            .btn-submit:active { transform: scale(0.98); }
        </style>
    </head>
    <body class="p-5 flex items-center justify-center min-h-screen">
        <div class="w-full max-w-sm">
            <header class="text-center mb-8">
                <div class="inline-block p-3 bg-blue-100 rounded-2xl mb-3">
                    <span class="text-3xl">🏗️</span>
                </div>
                <h1 class="text-2xl font-black text-gray-800 tracking-tight">利益予測くん</h1>
                <p class="text-gray-500 text-sm font-medium mt-1">現場の数字をパッと診断</p>
            </header>

            <div class="card p-7 space-y-6">
                <div>
                    <p class="input-label">現場名</p>
                    <input type="text" id="site_name" class="input-field" placeholder="例：新宿マンション工事">
                </div>
                <div>
                    <p class="input-label">受注金額 (円)</p>
                    <input type="number" id="budget" class="input-field" value="500000" step="10000">
                </div>
                <div>
                    <p class="input-label">予定工期 (人工)</p>
                    <input type="number" id="man_days" class="input-field" value="10">
                </div>
                <button onclick="diagnose()" class="btn-submit">AI診断を開始</button>
            </div>

            <div id="result" class="mt-8 hidden card p-7 text-center animate-bounce-in">
                <div id="judgment" class="text-4xl font-black mb-3"></div>
                <div id="profit" class="text-2xl font-bold text-gray-800"></div>
                <div id="margin" class="text-blue-600 font-bold text-sm mb-4"></div>
                <div id="advice" class="p-3 bg-gray-50 rounded-lg text-sm text-gray-600 leading-relaxed"></div>
            </div>
        </div>

        <script>
            async function diagnose() {
                const btn = document.querySelector('.btn-submit');
                btn.innerText = "診断中...";
                btn.disabled = true;

                const data = {
                    site_name: document.getElementById('site_name').value || "未設定の現場",
                    budget: parseInt(document.getElementById('budget').value),
                    man_days: parseInt(document.getElementById('man_days').value)
                };

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    const result = await res.json();

                    document.getElementById('result').classList.remove('hidden');
                    document.getElementById('judgment').innerText = result.judgment;
                    document.getElementById('profit').innerText = result.predicted_profit;
                    document.getElementById('margin').innerText = "利益率: " + result.predicted_margin;
                    document.getElementById('advice').innerText = result.advice;
                    
                    const j = document.getElementById('judgment');
                    if(result.judgment.includes('✅')) j.className = "text-4xl font-black mb-3 text-green-600";
                    if(result.judgment.includes('⚠️')) j.className = "text-4xl font-black mb-3 text-yellow-500";
                    if(result.judgment.includes('🚨')) j.className = "text-4xl font-black mb-3 text-red-600";

                    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                } catch (e) {
                    alert("エラーが起きました。もう一度試してください。");
                } finally {
                    btn.innerText = "AI診断を開始";
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
