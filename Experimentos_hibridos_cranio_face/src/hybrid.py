from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from .common import LAB_ROOT, save_csv
from .matching import _calibration, _shared_distance


def build_component_scores(paired: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    by_id = paired.set_index(paired["id_pessoa"].astype(str), drop=False); stats = _calibration(paired); rows=[]
    for _, p in pairs.iterrows():
        face = by_id.loc[str(p["face_id"])]; skull = by_id.loc[str(p["cranio_id"])]
        d, n = _shared_distance(face, skull, stats); anatomy = 0.0 if n == 0 else 1/(1+d)
        fg, cg = str(face.get("genero_final", "INDEFINIDO")), str(skull.get("classe_cranio", "INDEFINIDO"))
        gender = 0.5 if "INDEFINIDO" in (fg+cg).upper() or not fg else float((fg.startswith("MASC") and cg.startswith("MASC")) or (fg.startswith("FEM") and cg.startswith("FEM")))
        quality = float(pd.to_numeric(skull.get("score_qualidade_cranio", 50), errors="coerce") or 50)/100
        rows.append({**p.to_dict(), "score_anatomico": anatomy, "score_genero": gender, "score_qualidade": quality, "regioes_comparadas": n})
    out=pd.DataFrame(rows); save_csv(out, LAB_ROOT/"resultados/matching/componentes_hibridos.csv"); return out


def evaluate_hybrid(components: pd.DataFrame, anatomy_enabled: bool) -> dict:
    global_path=LAB_ROOT/"resultados/matching/predicoes_two_tower.csv"
    if global_path.exists():
        g=pd.read_csv(global_path); components=components.merge(g[["face_id","cranio_id","score_global_two_tower"]],on=["face_id","cranio_id"],how="left")
    features=[c for c in ["score_global_two_tower","score_anatomico","score_genero","score_qualidade"] if c in components]
    if not anatomy_enabled:
        status={"status":"BLOQUEADO_GATE_ANATOMICO","oficial":False,"motivo":"validacao anatomica pendente ou cobertura insuficiente","features_disponiveis":features}
        (LAB_ROOT/"resultados/matching/status_hibrido.json").write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding="utf-8"); return status
    train=components[components.split=="treino"].dropna(subset=features+["label"]); test=components[components.split=="teste"].dropna(subset=features+["label"])
    if train.empty or test.empty:
        return {"status":"SEM_DADOS","oficial":False}
    model=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=20260915).fit(train[features],train.label)
    prob=model.predict_proba(test[features])[:,1]; pred=(prob>=.5).astype(int); y=test.label.to_numpy()
    status={"status":"OK","oficial":True,"features":features,"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"roc_auc":roc_auc_score(y,prob),"brier":brier_score_loss(y,prob)}
    (LAB_ROOT/"resultados/matching/status_hibrido.json").write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding="utf-8"); return status
