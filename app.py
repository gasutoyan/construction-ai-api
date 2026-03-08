import os
import sys
import subprocess

# --- 🚀 起動時にライブラリを強制インストールする魔法 ---
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"📦 {package} が見つからないのでインストールします...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# geopyを強制インストール
install_and_import('geopy')

# ここから通常のインポート
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import json
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

app = FastAPI()

# --- 💾 データ保存設定 ---
DATA_DIR = "user_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_user_path(user_id: str):
    return os.path.join(DATA_DIR, f"{user_id}.json")

# --- 📝 データモデル ---
class Employee(BaseModel):
    name: str
    wage: int
    rank: str

class Expense(BaseModel):
    type: str
    amount: int
    date: str = ""

class Job(BaseModel):
    name: str
    budget: int
    duration: int
    current_day: int = 0
    expenses: List[Expense] = []

class UserData(BaseModel):
    office_address: str = ""
    employees: List[Employee] = []
    jobs: List[Job] = []

class PredictRequest(BaseModel):
    site_address: str
    budget: int
    duration: int
    user_id: str

class AnalyzeJobRequest(BaseModel):
    user_id: str
    job_index: int

# --- 🧠 計算ロジック ---
def get_estimated_distance(addr1, addr2):
    try:
        if not addr1 or not addr2: return 0.0
        # Render環境でも動きやすいようにUser-Agentを工夫
        geolocator = Nominatim(user_agent="my_const_app_final_deploy")
        loc1 = geolocator.geocode(addr1)
        loc2 = geolocator.geocode(addr2)
        if loc1 and loc2:
            p1 = (loc1.latitude, loc1.longitude)
            p2 = (loc2.latitude, loc2.longitude)
            dist = geodesic(p1, p2).km
            return round(dist * 1.3, 1)
    except Exception as e:
        print(f"Distance calculation error: {e}")
    return 0.0

# --- 📡 APIエンドポイント ---
@app.get("/api/data/{user_id}")
def load_data(user_id: str):
    path = get_user_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {"office_address": "", "employees": [], "jobs": []}

@app.post("/api/data/{user_id}")
def save_data(user_id: str, data: UserData):
    with open(get_user_path(user_id), "w", encoding="utf-8") as f:
        json.dump(data.dict(), f, ensure_ascii=False, indent=4)
    return {"status": "ok"}

@app.post("/api/analyze")
def analyze_job(req: AnalyzeJobRequest):
    user_data = load_data(req.user_id)
    if not (0 <= req.job_index < len(user_data['jobs'])):
        return {"status": "error"}
    job = user_data['jobs'][req.job_index]
    total_exp = sum([e['amount'] for e in job.get('expenses', [])])
    current_day = job.get('current_day', 0)
    duration = job.get('duration', 1)
    budget = job.get('budget', 0)
    if current_day == 0: return {"status": "nodata"}
    avg_daily_cost = total_exp / current_day
    remaining_days = duration - current_day
    predicted_total = total_exp + (avg_daily_cost * remaining_days)
    predicted_profit = budget - predicted_total
    current_profit = budget - total_exp
    allowed_daily_cost = current_profit / remaining_days if remaining_days > 0 else 0
    cut_needed = avg_daily_cost - allowed_daily_cost
    cut_people = round(cut_needed / 18000, 1)
    return {
        "status": "ok",
        "current": {"day": current_day, "avg_cost": int(avg_daily_cost), "spent": total_exp},
        "prediction": {"final_cost": int(predicted_total), "final_profit": int(predicted_profit), "is_danger": predicted_profit < 0},
        "advice": {"cut_daily": int(cut_needed), "cut_people": cut_people}
    }

@app.post("/api/predict")
def predict(req: PredictRequest):
    user_data_dict = load_data(req.user_id)
    office_addr = user_data_dict.get("office_address", "")
    employees = user_data_dict.get("employees", [])
    dist = get_estimated_distance(office_addr, req.site_address)
    one_day_fuel = (dist * 2 / 8) * 170
    total_fuel = int(one_day_fuel * req.duration)
    total_toll = int((dist * 25) * 2 * req.duration) if dist > 30 else 0
    transport_total = total_fuel + total_toll
    daily_team_cost = sum([e['wage'] for e in employees]) if employees else 18000
    labor_total = daily_team_cost * req.duration
    total_cost = transport_total + labor_total
    profit = req.budget - total_cost
    margin = (profit / req.budget * 100) if req.budget > 0 else 0
    status = "✅ 超優良" if margin >= 30 else ("⭕ 良好" if margin >= 20 else ("⚠️ 注意" if margin >= 10 else "🚨 赤字危険"))
    return {
        "meta": {"dist": dist, "days": req.duration, "head_count": len(employees)},
        "breakdown": {"transport": transport_total, "labor": labor_total, "total": total_cost},
        "result": {"profit": profit, "margin": margin, "status": status}
    }

@app.get("/", response_class=HTMLResponse)
def read_root():
    # ※ここは以前のHTMLコードと全く同じなので、そのまま保持されます
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>現場マネージャー</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700;900&display=swap');
        body { font-family: 'Noto Sans JP', sans-serif; background-color: #f1f5f9; color: #334155; }
        .view { display: none; }
        .active { display: block; }
        .card { background: white; border-radius: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        .progress-bar { transition: width 0.5s ease-in-out; }
    </style>
</head>
<body class="max-w-md mx-auto min-h-screen pb-20">
    <div id="v-login" class="view active px-6 pt-20">
        <h1 class="text-4xl font-black text-center mb-2 tracking-tighter">現場マネージャー</h1>
        <p class="text-center text-slate-400 mb-10 font-bold">Construction AI v6.1 (Stable)</p>
        <div class="card p-8 space-y-4 shadow-xl">
            <input type="text" id="user_id" class="w-full border-2 p-4 rounded-xl text-lg font-bold" placeholder="ユーザーID">
            <button onclick="login()" class="w-full bg-slate-900 text-white p-4 rounded-xl font-bold text-lg shadow-lg active:scale-95 transition">ログイン</button>
        </div>
    </div>
    <div id="v-dash" class="view px-6 pt-8">
        <header class="flex justify-between items-center mb-8">
            <h2 class="text-xl font-black">MENU</h2>
            <button onclick="location.reload()" class="text-xs font-bold text-red-500 bg-red-50 px-3 py-1 rounded-full border border-red-100">ログアウト</button>
        </header>
        <div class="space-y-4">
            <button onclick="show('v-set')" class="w-full card p-6 flex items-center hover:bg-slate-50 text-left transition shadow-md">
                <span class="text-3xl mr-5">⚙️</span>
                <div><h3 class="font-bold text-lg">会社設定</h3><p class="text-xs text-slate-400">拠点・社員マスタ</p></div>
            </button>
            <button onclick="show('v-mgr')" class="w-full card p-6 flex items-center hover:bg-slate-50 text-left transition shadow-md">
                <span class="text-3xl mr-5">📈</span>
                <div><h3 class="font-bold text-lg">現場管理</h3><p class="text-xs text-slate-400">実行予算 & 未来予測</p></div>
            </button>
            <button onclick="show('v-pre')" class="w-full card p-6 flex items-center hover:bg-slate-50 text-left transition shadow-md">
                <span class="text-3xl mr-5">🏗️</span>
                <div><h3 class="font-bold text-lg">利益予測</h3><p class="text-xs text-slate-400">受注前 AI診断</p></div>
            </button>
        </div>
    </div>
    <div id="v-set" class="view px-6 pt-8">
        <button onclick="show('v-dash')" class="mb-4 text-sm font-bold text-slate-400">← MENU</button>
        <h2 class="text-xl font-black mb-4">⚙️ 会社設定</h2>
        <div class="card p-6 space-y-6 shadow-lg">
            <div>
                <label class="text-xs font-black text-slate-400 uppercase">拠点住所</label>
                <input type="text" id="set-addr" class="w-full border-b-2 py-2 outline-none font-bold text-slate-700" placeholder="群馬県伊勢崎市...">
            </div>
            <div>
                <label class="text-xs font-black text-slate-400 uppercase">社員登録</label>
                <div id="emp-list" class="mt-3 space-y-2 mb-4"></div>
                <div class="bg-slate-50 p-4 rounded-xl space-y-3">
                    <input type="text" id="emp-n" class="w-full text-sm p-3 rounded-lg border font-bold" placeholder="氏名">
                    <div class="flex gap-2">
                        <input type="number" id="emp-w" class="w-1/2 text-sm p-3 rounded-lg border font-bold" placeholder="日当">
                        <select id="emp-r" class="w-1/2 text-sm p-3 rounded-lg border font-bold">
                            <option>標準</option><option>職長</option><option>見習</option>
                        </select>
                    </div>
                    <button onclick="addEmp()" class="w-full bg-slate-200 text-slate-600 p-2 rounded-lg font-bold text-xs hover:bg-slate-300">＋ 追加</button>
                </div>
            </div>
            <button onclick="saveAll()" class="w-full bg-blue-600 text-white p-4 rounded-xl font-bold shadow-lg">設定を保存</button>
        </div>
    </div>
    <div id="v-mgr" class="view px-6 pt-8 pb-32">
        <button onclick="show('v-dash')" class="mb-4 text-sm font-bold text-slate-400">← MENU</button>
        <div class="flex justify-between items-center mb-4">
            <h2 class="text-xl font-black">📈 現場管理</h2>
            <button onclick="toggleAddJob()" class="bg-slate-900 text-white text-xs px-4 py-2 rounded-full font-bold shadow-lg">＋ 新規現場</button>
        </div>
        <div id="add-job-box" class="hidden card p-5 mb-6 space-y-3 border-2 border-slate-200">
            <input type="text" id="j-n" class="w-full border-2 p-2 rounded-lg text-sm font-bold" placeholder="現場名">
            <input type="number" id="j-b" class="w-full border-2 p-2 rounded-lg text-sm font-bold" placeholder="受注金額">
            <div class="flex items-center gap-2">
                <input type="number" id="j-d" class="w-full border-2 p-2 rounded-lg text-sm font-bold" placeholder="計画工期">
                <span class="text-xs font-bold text-slate-400 whitespace-nowrap">日間</span>
            </div>
            <button onclick="addJob()" class="w-full bg-emerald-600 text-white p-2 rounded-lg font-bold text-sm shadow">登録開始</button>
        </div>
        <div id="job-list" class="space-y-6"></div>
        <div id="modal-exp" class="hidden fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center p-6 z-50 backdrop-blur-sm">
            <div class="bg-white w-full max-w-sm rounded-3xl p-6 space-y-4 shadow-2xl">
                <h3 class="font-black text-xl text-center" id="modal-title">経費入力</h3>
                <div class="grid grid-cols-2 gap-3">
                    <button onclick="setExpType('gas')" class="exp-btn border-2 p-3 rounded-xl text-sm font-bold hover:bg-slate-50">⛽ ガソリン</button>
                    <button onclick="setExpType('toll')" class="exp-btn border-2 p-3 rounded-xl text-sm font-bold hover:bg-slate-50">🛣️ 高速代</button>
                    <button onclick="setExpType('labor')" class="exp-btn border-2 p-3 rounded-xl text-sm font-bold hover:bg-slate-50">👷 人件費</button>
                    <button onclick="setExpType('other')" class="exp-btn border-2 p-3 rounded-xl text-sm font-bold hover:bg-slate-50">🍱 その他</button>
                </div>
                <input type="hidden" id="exp-type">
                <input type="number" id="exp-amount" class="w-full border-4 border-slate-100 p-4 rounded-2xl text-2xl font-black text-right outline-none focus:border-blue-500" placeholder="0">
                <div class="flex gap-3 pt-2">
                    <button onclick="closeModal()" class="w-1/2 bg-slate-100 text-slate-500 p-4 rounded-xl font-bold">戻る</button>
                    <button onclick="submitExpense()" class="w-1/2 bg-blue-600 text-white p-4 rounded-xl font-bold shadow-lg">追加する</button>
                </div>
            </div>
        </div>
        <div id="modal-ai" class="hidden fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center p-4 z-50 backdrop-blur-sm">
            <div class="bg-white w-full max-w-md rounded-3xl p-6 shadow-2xl overflow-y-auto max-h-[80vh]">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="font-black text-2xl">🔮 AI未来予測</h3>
                    <button onclick="closeAiModal()" class="text-slate-400 font-bold text-2xl">×</button>
                </div>
                <div id="ai-loading" class="text-center py-10 font-bold text-slate-400">分析中...</div>
                <div id="ai-content" class="hidden space-y-6">
                    <div class="text-center">
                        <p class="text-xs font-bold text-slate-400 uppercase mb-1">このまま進むと...</p>
                        <div id="ai-pred-profit" class="text-4xl font-black tracking-tight mb-2"></div>
                        <div id="ai-pred-msg" class="text-sm font-bold px-3 py-1 rounded-full inline-block"></div>
                    </div>
                    <div class="border-t border-slate-100 my-2"></div>
                    <div class="bg-slate-50 p-5 rounded-2xl border border-slate-200">
                        <h4 class="font-bold text-slate-700 mb-3 flex items-center">🛡️ 生き残り対策</h4>
                        <div id="ai-advice-box">
                            <p class="text-sm text-slate-600 mb-2">残り日程を...</p>
                            <div class="flex items-center gap-3 mb-2">
                                <span class="text-2xl">📉</span>
                                <div><p class="text-xs text-slate-400 font-bold">1日あたりのコスト削減</p><p class="font-black text-lg text-slate-800" id="ai-cut-money"></p></div>
                            </div>
                            <div class="flex items-center gap-3">
                                <span class="text-2xl">👷</span>
                                <div><p class="text-xs text-slate-400 font-bold">人員削減目安</p><p class="font-black text-lg text-slate-800" id="ai-cut-people"></p></div>
                            </div>
                        </div>
                        <div id="ai-advice-ok" class="hidden text-center text-emerald-600 font-bold py-2">今のペースで問題ありません！<br>この調子で進めましょう👍</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div id="v-pre" class="view px-6 pt-8 pb-20">
        <button onclick="show('v-dash')" class="mb-4 text-sm font-bold text-slate-400">← MENU</button>
        <h2 class="text-xl font-black mb-4">🏗️ 利益予測</h2>
        <div class="card p-5 space-y-3 shadow-md mb-6">
            <input type="text" id="p-addr" class="w-full border p-3 rounded-lg outline-none" placeholder="現場住所">
            <input type="number" id="p-bud" class="w-full border p-3 rounded-lg outline-none" placeholder="受注金額">
            <div class="flex items-center gap-2">
                <input type="number" id="p-dur" class="w-full border p-3 rounded-lg outline-none" placeholder="予定日数">
                <span class="text-sm font-bold text-slate-500 whitespace-nowrap">日間</span>
            </div>
            <p class="text-xs text-slate-400 text-right">※登録社員全員で稼働計算</p>
            <button onclick="runPre()" class="w-full bg-orange-500 text-white p-3 rounded-lg font-bold shadow text-lg">AI診断実行</button>
        </div>
        <div id="p-res" class="hidden animate-fade-in">
            <div class="card p-6 text-center border-4 border-orange-100 mb-4">
                <div id="r-stat" class="text-4xl font-black mb-1"></div>
                <div id="r-marg" class="text-orange-600 font-bold mb-4"></div>
                <div class="text-xs text-slate-400 font-bold mb-1">予測粗利益</div>
                <div id="r-prof" class="text-3xl font-black text-slate-800 tracking-tight"></div>
            </div>
            <div class="card p-5">
                <h3 class="font-bold text-sm mb-3 border-b pb-2">📊 試算内訳</h3>
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between"><span class="text-slate-500">条件</span><span class="font-bold" id="d-cond"></span></div>
                    <div class="flex justify-between"><span class="text-slate-500">交通費</span><span class="font-bold" id="d-trans"></span></div>
                    <div class="flex justify-between"><span class="text-slate-500">人件費</span><span class="font-bold" id="d-labor"></span></div>
                    <div class="border-t pt-2 flex justify-between text-base"><span class="font-bold text-slate-700">経費合計</span><span class="font-black text-red-500" id="d-total"></span></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        let user = null;
        let data = { office_address: "", employees: [], jobs: [] };
        let currentJobIndex = -1;
        function show(id) {
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            if(id === 'v-set') renderEmps();
            if(id === 'v-mgr') renderJobs();
        }
        async function login() {
            user = document.getElementById('user_id').value;
            if(!user) return;
            const res = await fetch('/api/data/' + user);
            data = await res.json();
            document.getElementById('set-addr').value = data.office_address || "";
            show('v-dash');
        }
        async function saveAll() {
            data.office_address = document.getElementById('set-addr').value;
            await fetch('/api/data/'+user, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
        }
        function renderEmps() {
            document.getElementById('emp-list').innerHTML = data.employees.map((e, i) => `
                <div class="flex justify-between bg-white border p-3 rounded-xl text-xs font-bold items-center shadow-sm">
                    <span>${e.name} <span class="bg-slate-100 px-2 py-1 rounded ml-1 text-[10px] text-slate-500">${e.rank}</span></span>
                    <div><span class="mr-3 text-slate-500">¥${e.wage.toLocaleString()}</span>
                    <button onclick="data.employees.splice(${i},1);saveAll();renderEmps()" class="text-red-400 font-bold px-2">×</button></div>
                </div>
            `).join('');
        }
        function addEmp() {
            const n = document.getElementById('emp-n').value;
            const w = parseInt(document.getElementById('emp-w').value);
            const r = document.getElementById('emp-r').value;
            if(n && w) { data.employees.push({name:n, wage:w, rank:r}); saveAll(); renderEmps(); document.getElementById('emp-n').value=''; }
        }
        function renderJobs() {
            document.getElementById('job-list').innerHTML = data.jobs.map((j, i) => {
                const totalExp = (j.expenses || []).reduce((sum, e) => sum + e.amount, 0);
                const currentProfit = j.budget - totalExp;
                const progress = Math.min((totalExp / j.budget) * 100, 100);
                const statusColor = progress > 80 ? "bg-red-500" : (progress > 50 ? "bg-orange-400" : "bg-emerald-500");
                const curDay = j.current_day || 0;
                const duration = j.duration || 1;
                return `
                <div class="card p-5 shadow-md border border-slate-100 relative">
                    <button onclick="deleteJob(${i})" class="absolute top-4 right-4 text-slate-300 hover:text-red-500 font-bold text-xs p-2">🗑️ 削除</button>
                    <div class="mb-4 pr-10">
                        <h3 class="font-black text-xl text-slate-800">${j.name}</h3>
                        <p class="text-xs text-slate-400 font-bold mt-1">予算: ¥${j.budget.toLocaleString()}</p>
                    </div>
                    <div class="flex gap-2 mb-4">
                        <button onclick="openAiModal(${i})" class="flex-1 bg-purple-600 text-white text-[10px] px-3 py-2 rounded-lg font-bold shadow-md active:scale-95 transition flex items-center justify-center gap-1">
                            🔮 AI診断
                        </button>
                        <button onclick="openModal(${i})" class="flex-1 bg-blue-600 text-white text-[10px] px-3 py-2 rounded-lg font-bold shadow-md active:scale-95 transition">
                            ＋ 経費入力
                        </button>
                    </div>
                    <div class="bg-slate-50 p-3 rounded-xl mb-4 flex justify-between items-center border border-slate-200">
                        <span class="text-xs font-bold text-slate-500">経過日数</span>
                        <div class="flex items-center gap-3">
                            <button onclick="updateDay(${i}, -1)" class="w-8 h-8 bg-white rounded-full font-bold shadow text-slate-500 hover:bg-slate-100">-</button>
                            <span class="font-black text-lg w-16 text-center">${curDay} <span class="text-xs text-slate-400">/ ${duration}日</span></span>
                            <button onclick="updateDay(${i}, 1)" class="w-8 h-8 bg-white rounded-full font-bold shadow text-blue-500 hover:bg-blue-50">+</button>
                        </div>
                    </div>
                    <div class="w-full bg-slate-100 h-3 rounded-full overflow-hidden mb-2">
                        <div class="${statusColor} h-full progress-bar" style="width: ${progress}%"></div>
                    </div>
                    <div class="flex justify-between items-end">
                        <div class="text-xs text-slate-500 font-bold">支出: ¥${totalExp.toLocaleString()}</div>
                        <div class="text-right">
                            <p class="text-[10px] text-slate-400 font-bold">現在の残金</p>
                            <p class="text-xl font-black ${currentProfit < 0 ? 'text-red-500' : 'text-slate-800'}">¥${currentProfit.toLocaleString()}</p>
                        </div>
                    </div>
                </div>`;
            }).join('');
        }
        function toggleAddJob() { document.getElementById('add-job-box').classList.toggle('hidden'); }
        function addJob() {
            const n = document.getElementById('j-n').value;
            const b = parseInt(document.getElementById('j-b').value);
            const d = parseInt(document.getElementById('j-d').value);
            if(n && b && d) { data.jobs.push({name:n, budget:b, duration:d, current_day:0, expenses:[]}); saveAll(); renderJobs(); toggleAddJob(); }
        }
        function deleteJob(index) {
            if(confirm("本当にこの現場を削除しますか？")) { data.jobs.splice(index, 1); saveAll(); renderJobs(); }
        }
        function updateDay(index, change) {
            let job = data.jobs[index];
            let newDay = (job.current_day || 0) + change;
            if(newDay < 0) newDay = 0;
            if(newDay > job.duration) newDay = job.duration;
            job.current_day = newDay;
            saveAll(); renderJobs();
        }
        function openModal(index) {
            currentJobIndex = index;
            document.getElementById('modal-title').innerText = `${data.jobs[index].name}`;
            document.getElementById('modal-exp').classList.remove('hidden');
        }
        function closeModal() { document.getElementById('modal-exp').classList.add('hidden'); document.getElementById('exp-amount').value = ''; }
        function setExpType(type) {
            document.getElementById('exp-type').value = type;
            document.querySelectorAll('.exp-btn').forEach(b => b.classList.remove('bg-slate-200'));
            event.target.classList.add('bg-slate-200');
        }
        async function submitExpense() {
            const amount = parseInt(document.getElementById('exp-amount').value);
            const type = document.getElementById('exp-type').value || 'other';
            if(amount && currentJobIndex >= 0) {
                if(!data.jobs[currentJobIndex].expenses) data.jobs[currentJobIndex].expenses = [];
                data.jobs[currentJobIndex].expenses.push({type:type, amount:amount, date:""});
                await saveAll(); renderJobs(); closeModal();
            }
        }
        async function openAiModal(index) {
            document.getElementById('modal-ai').classList.remove('hidden');
            document.getElementById('ai-loading').classList.remove('hidden');
            document.getElementById('ai-content').classList.add('hidden');
            const req = { user_id: user, job_index: index };
            const res = await fetch('/api/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(req)});
            const r = await res.json();
            document.getElementById('ai-loading').classList.add('hidden');
            document.getElementById('ai-content').classList.remove('hidden');
            if(r.status === 'nodata') {
                document.getElementById('ai-pred-profit').innerText = "データ不足";
                document.getElementById('ai-pred-msg').innerText = "まずは経過日数を入力してください";
                document.getElementById('ai-pred-msg').className = "text-sm font-bold bg-slate-100 text-slate-500 px-3 py-1 rounded-full inline-block";
                document.getElementById('ai-advice-box').classList.add('hidden');
                document.getElementById('ai-advice-ok').classList.add('hidden');
                return;
            }
            const profit = r.prediction.final_profit;
            const pEl = document.getElementById('ai-pred-profit');
            pEl.innerText = (profit > 0 ? "+" : "") + profit.toLocaleString() + "円";
            pEl.className = `text-4xl font-black tracking-tight mb-2 ${profit < 0 ? 'text-red-500' : 'text-emerald-500'}`;
            const msgEl = document.getElementById('ai-pred-msg');
            if(profit < 0) {
                msgEl.innerText = "🚨 このままだと赤字確定です";
                msgEl.className = "text-sm font-bold bg-red-100 text-red-500 px-3 py-1 rounded-full inline-block";
                document.getElementById('ai-advice-box').classList.remove('hidden');
                document.getElementById('ai-advice-ok').classList.add('hidden');
                document.getElementById('ai-cut-money').innerText = `-¥${r.advice.cut_daily.toLocaleString()}`;
                document.getElementById('ai-cut-people').innerText = `-${r.advice.cut_people}人`;
            } else {
                msgEl.innerText = "✨ 黒字見込みです";
                msgEl.className = "text-sm font-bold bg-emerald-100 text-emerald-600 px-3 py-1 rounded-full inline-block";
                document.getElementById('ai-advice-box').classList.add('hidden');
                document.getElementById('ai-advice-ok').classList.remove('hidden');
            }
        }
        function closeAiModal() { document.getElementById('modal-ai').classList.add('hidden'); }
        async function runPre() {
            const req = { site_address: document.getElementById('p-addr').value, budget: parseInt(document.getElementById('p-bud').value), duration: parseInt(document.getElementById('p-dur').value), user_id: user };
            const res = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(req)});
            const r = await res.json();
            document.getElementById('p-res').classList.remove('hidden');
            document.getElementById('r-stat').innerText = r.result.status;
            document.getElementById('r-marg').innerText = `利益率 ${r.result.margin.toFixed(1)}%`;
            document.getElementById('r-prof').innerText = `¥${r.result.profit.toLocaleString()}`;
            document.getElementById('d-cond').innerText = `${r.meta.head_count}名 × ${r.meta.days}日間 / ${r.meta.dist}km`;
            document.getElementById('d-trans').innerText = `¥${r.breakdown.transport.toLocaleString()}`;
            document.getElementById('d-labor').innerText = `¥${r.breakdown.labor.toLocaleString()}`;
            document.getElementById('d-total').innerText = `¥${r.breakdown.total.toLocaleString()}`;
        }
    </script>
</body>
</html>
    """