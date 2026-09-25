from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from .common import LAB_ROOT, REPO_ROOT
from .overlays import create_matching_overlays


STYLE = """body{font-family:Arial,sans-serif;margin:24px;background:#f4f6f8;color:#17202a}a{color:#145a9c}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}.card{background:white;border:1px solid #dfe4ea;border-radius:10px;padding:12px}.card img{width:100%;height:260px;object-fit:contain;background:#111}.ok{color:#147a3d}.bad{color:#a72525}table{border-collapse:collapse;background:white}td,th{padding:7px;border:1px solid #ddd}nav a{margin-right:14px}small{color:#5d6d7e}.controls{position:sticky;top:0;z-index:5;background:#fff;padding:14px;border-radius:10px;box-shadow:0 2px 8px #0002;margin-bottom:16px}.controls select,.controls input{padding:8px;margin-right:10px}.candidate.correct{border:3px solid #17823b}.candidate h3{margin:7px 0}.candidate p{margin:5px 0}.triptych{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px}.triptych img{height:190px}.query{display:flex;gap:16px;align-items:center;background:white;padding:12px;border-radius:10px;margin-bottom:16px}.query img{width:180px;height:180px;object-fit:contain;background:#111}.badge{display:inline-block;padding:3px 7px;border-radius:12px;background:#e8eef5;margin-right:5px}"""


def _page(title: str, body: str) -> str:
    return f"<!doctype html><html lang='pt-br'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{STYLE}</style></head><body><nav><a href='index.html'>Inicio</a><a href='auditoria.html'>Auditoria</a><a href='cranios.html'>Cranios</a><a href='faces.html'>Faces</a><a href='matching.html'>Matching</a></nav><h1>{html.escape(title)}</h1>{body}</body></html>"


def _relative_asset(path: Path, report_dir: Path) -> str:
    return Path("../..").joinpath(path.relative_to(LAB_ROOT)).as_posix()


def _repo_asset(relative_path: str) -> str:
    return Path("../../..").joinpath(str(relative_path)).as_posix()


def _matching_browser(manifest: pd.DataFrame, ranking: pd.DataFrame, out: Path) -> str:
    if ranking.empty:
        return "<p>Ranking visual ainda nao foi gerado.</p>"
    visual = create_matching_overlays(manifest, ranking, top_k=10)
    meta = manifest.copy(); meta["id_pessoa"] = meta["id_pessoa"].astype(str)
    records = meta.set_index("id_pessoa").to_dict("index")
    grouped = {}
    for query, group in visual.groupby("query_id", sort=True):
        q = records.get(str(query), {})
        candidates = []
        for _, r in group.sort_values("rank").iterrows():
            c = records.get(str(r["candidate_id"]), {})
            candidates.append({
                "id": str(r["candidate_id"]), "rank": int(r["rank"]),
                "score": round(float(r["score_global_two_tower"]), 5), "correto": bool(r["correto"]),
                "sexo": str(c.get("classe_cranio", "INDEFINIDO")),
                "p_masc": round(float(c.get("prob_masculino_cranio", 0) or 0), 3),
                "p_fem": round(float(c.get("prob_feminino_cranio", 0) or 0), 3),
                "cranio": _repo_asset(c.get("imagem_cf", "")),
                "overlay": _relative_asset(LAB_ROOT / str(r["overlay_path"]), out),
                "alinhamento": str(r["overlay_status"]),
            })
        grouped[str(query)] = {
            "face": _repo_asset(q.get("imagem_face", "")), "sexo_face": str(q.get("genero_final", "INDEFINIDO")),
            "ano": str(q.get("ano", "")), "candidates": candidates,
        }
    payload = json.dumps(grouped, ensure_ascii=False).replace("</", "<\\/")
    options = "".join(f"<option value='{html.escape(k)}'>{html.escape(k)}</option>" for k in grouped)
    return f"""
<h2>Consulta visual por face</h2>
<p>Selecione ou pesquise uma identidade. As sobreposicoes sao exploratorias; amarelo indica os pontos usados no alinhamento.</p>
<div class='controls'><label>Face: <select id='querySelect'>{options}</select></label><label>Buscar ID: <input id='querySearch' placeholder='ex.: 2021_'></label><label><input type='checkbox' id='onlySameSex'> sexo craniano compativel</label></div>
<div id='queryBox'></div><div id='candidateGrid' class='grid'></div>
<script>const MATCH={payload};
const sel=document.getElementById('querySelect'), search=document.getElementById('querySearch'), same=document.getElementById('onlySameSex');
function esc(s){{return String(s).replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));}}
function compatible(face,skull){{face=face.toUpperCase();skull=skull.toUpperCase();return face.includes('MASC')?skull.includes('MASC'):face.includes('FEM')?skull.includes('FEM'):true;}}
function render(){{const id=sel.value,d=MATCH[id]; document.getElementById('queryBox').innerHTML=`<div class='query'><img src='${{d.face}}'><div><h2>Face ${{esc(id)}}</h2><span class='badge'>sexo: ${{esc(d.sexo_face)}}</span><span class='badge'>ano: ${{esc(d.ano)}}</span></div></div>`;
let cs=d.candidates.filter(c=>!same.checked||compatible(d.sexo_face,c.sexo));document.getElementById('candidateGrid').innerHTML=cs.map(c=>`<div class='card candidate ${{c.correto?'correct':''}}'><h3>#${{c.rank}} — ${{esc(c.id)}}</h3><p><b>${{esc(c.sexo)}}</b> · M ${{c.p_masc}} / F ${{c.p_fem}}</p><p>similaridade: <b>${{c.score}}</b> ${{c.correto?'· correspondência real':''}}</p><div class='triptych'><div><small>Face</small><img src='${{d.face}}'></div><div><small>Crânio</small><img src='${{c.cranio}}'></div><div><small>Sobreposição</small><img src='${{c.overlay}}'></div></div><small>Alinhamento: ${{esc(c.alinhamento)}}</small></div>`).join('');}}
sel.onchange=render;same.onchange=render;search.oninput=()=>{{const q=search.value.toLowerCase(),o=[...sel.options].find(x=>x.value.toLowerCase().includes(q));if(o){{sel.value=o.value;render();}}}};render();</script>"""


def generate_reports(manifest: pd.DataFrame, skull: pd.DataFrame, face: pd.DataFrame, ranking: pd.DataFrame, metrics: dict, max_cards: int = 250) -> Path:
    out = LAB_ROOT / "resultados/relatorios_html"; out.mkdir(parents=True, exist_ok=True)
    audit = manifest.groupby(["ano", "genero_final"], dropna=False).agg(identidades=("id_pessoa", "count"), elegiveis=("usar_matching", "sum")).reset_index()
    (out / "auditoria.html").write_text(_page("Auditoria do dataset", audit.to_html(index=False)), encoding="utf-8")
    cards=[]
    for _, r in skull.head(max_cards).iterrows():
        p = LAB_ROOT / "resultados/cranios/previews" / f"{r['id_pessoa']}_CF.jpg"; src=_relative_asset(p,out) if p.exists() else ""
        cards.append(f"<div class='card'><img src='{src}'><h3>{html.escape(str(r['id_pessoa']))}</h3><b>{html.escape(str(r.get('status_cranio_geo','')))}</b><p>Core: {r.get('regioes_centrais_detectadas','')} | topo truncado: {r.get('topo_truncado','')}</p></div>")
    (out / "cranios.html").write_text(_page("Cranios CF - pontos e zonas", "<div class='grid'>"+"".join(cards)+"</div>"), encoding="utf-8")
    compare=[]
    for _,r in skull.head(min(max_cards,120)).iterrows():
        p1=LAB_ROOT/"resultados/cranios/previews_v2"/f"{r['id_pessoa']}_CF.jpg"; p2=LAB_ROOT/"resultados/cranios/previews"/f"{r['id_pessoa']}_CF.jpg"
        if p1.exists() and p2.exists():
            compare.append(f"<div class='card'><h3>{html.escape(str(r['id_pessoa']))}</h3><small>V2</small><img src='{_relative_asset(p1,out)}'><small>V3</small><img src='{_relative_asset(p2,out)}'></div>")
    v2_path=LAB_ROOT/"resultados/cranios/metricas_cranios_cf_v2.csv"
    comparison=""
    if v2_path.exists():
        v2=pd.read_csv(v2_path); joined=v2[["id_pessoa","confianca_par_orbital","regioes_centrais_detectadas"]].merge(skull[["id_pessoa","confianca_par_orbital","regioes_centrais_detectadas"]],on="id_pessoa",suffixes=("_v2","_v3"))
        comparison="<p>Par orbital confiavel: V2 {:.1%} → V3 {:.1%}. Core completo: V2 {:.1%} → V3 {:.1%}.</p>".format((joined.confianca_par_orbital_v2>=.45).mean(),(joined.confianca_par_orbital_v3>=.45).mean(),(joined.regioes_centrais_detectadas_v2>=4).mean(),(joined.regioes_centrais_detectadas_v3>=4).mean())
    (out/"cranios_v1_v2.html").write_text(_page("Comparacao visual Cranios V2 x V3",comparison+"<div class='grid'>"+"".join(compare)+"</div>"),encoding="utf-8")
    cards=[]
    for _, r in face.head(max_cards).iterrows():
        p=LAB_ROOT / "resultados/faces/previews" / f"{r['id_pessoa']}_face.jpg"; src=_relative_asset(p,out) if p.exists() else ""
        cards.append(f"<div class='card'><img src='{src}'><h3>{html.escape(str(r['id_pessoa']))}</h3><b>{html.escape(str(r.get('status_face_anatomica','')))}</b></div>")
    (out / "faces.html").write_text(_page("Faces - camada anatomica automatica", "<div class='grid'>"+"".join(cards)+"</div>"), encoding="utf-8")
    top = ranking[ranking["rank"] <= 10].head(1000) if not ranking.empty else ranking
    hybrid_path=LAB_ROOT/"resultados/matching/status_hibrido.json"; hybrid=json.loads(hybrid_path.read_text(encoding="utf-8")) if hybrid_path.exists() else {"status":"NAO_EXECUTADO"}
    tt_path=LAB_ROOT/"resultados/matching/metricas_two_tower.json"; tt=json.loads(tt_path.read_text(encoding="utf-8")) if tt_path.exists() else {"status":"NAO_TREINADO"}
    gate_path=LAB_ROOT/"resultados/cranios/gate_viabilidade.csv"; gate=pd.read_csv(gate_path).iloc[0].to_dict() if gate_path.exists() else {}
    tt_rank_path=LAB_ROOT/"resultados/matching/ranking_two_tower_teste.csv"; tt_rank=pd.read_csv(tt_rank_path) if tt_rank_path.exists() else pd.DataFrame()
    browser = _matching_browser(manifest, tt_rank, out) if not tt_rank.empty else ""
    body = browser+"<h2>Two-tower global</h2><pre>"+html.escape(json.dumps(tt,indent=2,ensure_ascii=False))+"</pre><h2>Gate anatomico</h2><pre>"+html.escape(str(gate))+"</pre><h2>Status hibrido</h2><pre>"+html.escape(str(hybrid))+"</pre><h2>Metricas anatomicas exploratorias</h2><pre>"+html.escape(str(metrics))+"</pre><h2>Top-20 two-tower</h2>"+(tt_rank.head(1000).to_html(index=False) if not tt_rank.empty else "<p>Sem ranking two-tower.</p>")+"<h2>Top-10 anatomico</h2>"+(top.to_html(index=False) if not top.empty else "<p>Sem ranking.</p>")
    (out / "matching.html").write_text(_page("Matching e ranking", body), encoding="utf-8")
    experiments=pd.DataFrame([
        {"experimento":"E0 baseline legado","estado":"IMPORTADO/PENDENTE DE REPRODUCAO OFICIAL","observacao":"pipeline original preservado"},
        {"experimento":"E1 anatomico","estado":"EXPLORATORIO","observacao":f"Recall@1={metrics.get('recall_at_1',0):.3f}; gate={gate.get('cobertura_core',0):.3f}"},
        {"experimento":"E2 two-tower global","estado":"TREINADO" if "ranking" in tt else "NAO_TREINADO","observacao":f"Recall@1={tt.get('ranking',{}).get('recall_at_1',0):.3f}; AUC={tt.get('classification',{}).get('roc_auc',0):.3f}"},
        {"experimento":"E3 two-tower por regioes","estado":"BLOQUEADO","observacao":"depende do gate anatomico"},
        {"experimento":"E4 hibrido","estado":hybrid.get("status","NAO_EXECUTADO"),"observacao":hybrid.get("motivo","")},
        {"experimento":"E5 ablacoes","estado":"PENDENTE","observacao":"executar apos estabilizar E2/E4"},
    ])
    (out/"comparacoes.html").write_text(_page("Comparacao de experimentos",experiments.to_html(index=False)),encoding="utf-8")
    index = "<p>Laboratorio separado dos pipelines originais.</p><ul><li><a href='auditoria.html'>Auditoria</a></li><li><a href='cranios.html'>Cranios V3</a></li><li><a href='cranios_v1_v2.html'>Cranios V2 x V3</a></li><li><a href='faces.html'>Faces</a></li><li><a href='matching.html'>Matching</a></li><li><a href='comparacoes.html'>Comparacao de experimentos</a></li></ul>"
    (out / "index.html").write_text(_page("Laboratorio hibrido cranio-face", index), encoding="utf-8")
    return out / "index.html"
