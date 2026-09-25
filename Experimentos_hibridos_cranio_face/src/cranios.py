from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import LAB_ROOT, abs_repo, read_image, save_csv, write_image


POINT_COLORS = {"orbita_direita": (255, 80, 80), "orbita_esquerda": (80, 160, 255), "nasal": (80, 255, 255), "mento": (80, 255, 80), "gonio_direito": (255, 80, 255), "gonio_esquerdo": (255, 80, 255), "zigio_direito": (255, 200, 80), "zigio_esquerdo": (255, 200, 80)}


def foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([gray[:5].ravel(), gray[-5:].ravel(), gray[:, :5].ravel(), gray[:, -5:].ravel()])
    bg = float(np.median(border)); diff = cv2.absdiff(gray, np.full_like(gray, int(bg)))
    nonzero = diff[diff > 3]
    threshold = max(8, min(80, int(np.percentile(nonzero, 20) * 0.35))) if len(nonzero) else 8
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    candidates = list(range(1, n)); candidates.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    keep = np.zeros_like(mask)
    for i in candidates[:3]:
        if stats[i, cv2.CC_STAT_AREA] >= image.shape[0] * image.shape[1] * 0.01:
            keep[labels == i] = 255
    return keep


def _bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if not len(xs): return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _cavity_contours(image: np.ndarray, mask: np.ndarray, box) -> list[dict]:
    x0, y0, x1, y1 = box; gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = gray[y0:y1 + 1, x0:x1 + 1]; fg = mask[y0:y1 + 1, x0:x1 + 1] > 0
    vals = roi[fg]
    if not len(vals): return []
    threshold = float(np.percentile(vals, 28))
    dark = ((roi <= threshold) & fg).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1; result = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < bw * bh * 0.002 or a > bw * bh * 0.15: continue
        x, y, w, h = cv2.boundingRect(c); cx, cy = x + w / 2, y + h / 2
        result.append({"contour": c + np.array([[[x0, y0]]]), "cx": cx + x0, "cy": cy + y0, "x": x + x0, "y": y + y0, "w": w, "h": h, "area": a, "rx": cx / bw, "ry": cy / bh})
    return result


def detect_geometry_v1(image: np.ndarray) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    h, w = image.shape[:2]; mask = foreground_mask(image); box = _bbox(mask)
    if box is None:
        return {"status_cranio_geo": "FALHA", "motivo_cranio_geo": "SEM_PRIMEIRO_PLANO"}, [], mask, image.copy()
    x0, y0, x1, y1 = box; bw, bh = x1 - x0 + 1, y1 - y0 + 1; cx0 = (x0 + x1) / 2
    touches = {"topo_truncado": y0 <= 3, "base_truncada": y1 >= h - 4, "lateral_esq_truncada": x0 <= 3, "lateral_dir_truncada": x1 >= w - 4}
    cavities = _cavity_contours(image, mask, box)
    orbit_candidates = [c for c in cavities if 0.16 <= c["ry"] <= 0.58 and 0.10 <= c["rx"] <= 0.90]
    right_side = [c for c in orbit_candidates if c["cx"] < cx0]
    left_side = [c for c in orbit_candidates if c["cx"] >= cx0]
    def best(items, target_x):
        if not items: return None
        return max(items, key=lambda c: c["area"] * max(0.1, 1 - abs(c["rx"] - target_x)))
    od, oe = best(right_side, 0.32), best(left_side, 0.68)
    nasal_candidates = [c for c in cavities if 0.38 <= c["ry"] <= 0.76 and abs(c["cx"] - cx0) <= bw * 0.20]
    nasal = max(nasal_candidates, key=lambda c: c["area"], default=None)
    points = []
    def add(name, x, y, conf, method="geometria_2d"):
        points.append({"ponto": name, "x_px": float(x), "y_px": float(y), "x_norm": float((x - x0) / bw), "y_norm": float((y - y0) / bh), "confianca": float(np.clip(conf, 0, 1)), "visivel": bool(conf > 0), "metodo": method, "motivo_ausencia": "" if conf > 0 else "NAO_DETECTADO"})
    for name, c in [("orbita_direita", od), ("orbita_esquerda", oe)]:
        if c: add(name, c["cx"], c["cy"], min(1, 0.45 + c["area"] / (bw * bh * 0.04)))
        else: add(name, np.nan, np.nan, 0)
    if nasal: add("nasal", nasal["cx"], nasal["cy"], min(1, 0.40 + nasal["area"] / (bw * bh * 0.025)))
    else: add("nasal", np.nan, np.nan, 0)
    ys, xs = np.where(mask > 0); lower = ys >= (y0 + int(0.62 * bh)); lx, ly = xs[lower], ys[lower]
    if len(lx):
        bottom = ly.max(); band = np.abs(ly - bottom) <= max(3, int(0.025 * bh)); menton_x = float(np.median(lx[band])) if band.any() else cx0
        add("mento", menton_x, bottom, 0.85 if not touches["base_truncada"] else 0.35)
        yjaw = y0 + int(0.78 * bh); band2 = np.abs(ly - yjaw) <= max(5, int(0.04 * bh))
        if band2.any():
            add("gonio_direito", float(np.percentile(lx[band2], 10)), yjaw, 0.55)
            add("gonio_esquerdo", float(np.percentile(lx[band2], 90)), yjaw, 0.55)
    add("zigio_direito", x0 + 0.08 * bw, y0 + 0.51 * bh, 0.40)
    add("zigio_esquerdo", x0 + 0.92 * bw, y0 + 0.51 * bh, 0.40)

    valid = {p["ponto"]: p for p in points if p["visivel"]}
    core = sum(k in valid and valid[k]["confianca"] >= 0.45 for k in ["orbita_direita", "orbita_esquerda", "nasal", "mento"])
    status = "OK" if core == 4 else ("PARCIAL" if core >= 2 else "INCERTO")
    metrics = {"status_cranio_geo": status, "motivo_cranio_geo": "" if core >= 2 else "REGIOES_CENTRAIS_INSUFICIENTES", **touches,
               "bbox_x": x0, "bbox_y": y0, "bbox_w": bw, "bbox_h": bh, "area_relativa": float((mask > 0).mean()), "eixo_simetria_x": cx0,
               "mandibula_presente": bool(len(lx)), "regioes_centrais_detectadas": core}
    if od and oe:
        metrics["distancia_interorbital_norm"] = float(abs(oe["cx"] - od["cx"]) / bw)
        metrics["assimetria_orbital_area"] = float(abs(od["area"] - oe["area"]) / max(od["area"], oe["area"], 1))
    if nasal: metrics["largura_nasal_norm"] = float(nasal["w"] / bw); metrics["altura_nasal_norm"] = float(nasal["h"] / bh)
    if "mento" in valid and od and oe:
        metrics["orbita_mento_norm"] = float((valid["mento"]["y_px"] - (od["cy"] + oe["cy"]) / 2) / bh)
    gd, ge = valid.get("gonio_direito"), valid.get("gonio_esquerdo")
    if gd and ge:
        metrics["largura_mandibular_norm"] = float(abs(ge["x_px"] - gd["x_px"]) / bw)
    zd, ze = valid.get("zigio_direito"), valid.get("zigio_esquerdo")
    if zd and ze:
        metrics["largura_zigomatica_norm"] = float(abs(ze["x_px"] - zd["x_px"]) / bw)
    preview = image.copy(); cv2.rectangle(preview, (x0, y0), (x1, y1), (180, 180, 180), 2); cv2.line(preview, (int(cx0), y0), (int(cx0), y1), (255, 255, 0), 2)
    for p in points:
        if not p["visivel"]: continue
        xy = (int(p["x_px"]), int(p["y_px"])); color = POINT_COLORS.get(p["ponto"], (0, 255, 0)); cv2.circle(preview, xy, max(4, min(h, w)//160), color, -1); cv2.putText(preview, f"{p['ponto']} {p['confianca']:.2f}", (xy[0]+5, xy[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return metrics, points, mask, preview


def _anatomical_bbox(mask: np.ndarray):
    """Remove linhas finas de plataforma e artefatos afastados antes do bbox."""
    h, w = mask.shape; binary = mask > 0; widths = binary.sum(axis=1)
    valid = widths >= max(12, int(w * 0.07))
    runs, start = [], None
    for i, ok in enumerate(valid):
        if ok and start is None: start = i
        if start is not None and (not ok or i == h - 1):
            end = i if ok and i == h - 1 else i - 1
            if end - start >= max(20, int(h * .12)): runs.append((start, end))
            start = None
    if not runs: return _bbox(mask)
    y0, y1 = max(runs, key=lambda r: (r[1]-r[0]) * np.median(widths[r[0]:r[1]+1]))
    region = binary[y0:y1+1]; ys, xs = np.where(region)
    if not len(xs): return _bbox(mask)
    return int(xs.min()), int(y0), int(xs.max()), int(y1)


def _symmetry_axis(image: np.ndarray, box) -> tuple[float, float]:
    x0, y0, x1, y1 = box; bw, bh = x1-x0+1, y1-y0+1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 35, 110).astype(np.float32)/255
    ya, yb = y0+int(.16*bh), y0+int(.68*bh); half = max(15, int(.40*bw))
    best_x, best_err = (x0+x1)/2, 1e9
    for cx in np.linspace(x0+.43*bw, x0+.57*bw, 31):
        c=int(cx); span=min(half,c,x1-c)
        if span < 10: continue
        left=edges[ya:yb,c-span:c]; right=np.fliplr(edges[ya:yb,c+1:c+1+span])
        if left.shape != right.shape or not left.size: continue
        err=float(np.mean(np.abs(cv2.GaussianBlur(left,(7,7),0)-cv2.GaussianBlur(right,(7,7),0))) )
        if err < best_err: best_x,best_err=cx,err
    confidence=float(np.clip(1-best_err*3,0,1))
    return float(best_x), confidence


def _multi_threshold_candidates(image: np.ndarray, mask: np.ndarray, box) -> list[dict]:
    x0,y0,x1,y1=box; bw,bh=x1-x0+1,y1-y0+1; gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    roi=gray[y0:y1+1,x0:x1+1]; fg=mask[y0:y1+1,x0:x1+1]>0; vals=roi[fg]
    if not len(vals): return []
    allc=[]
    for percentile in (14,20,26,32,38):
        dark=((roi<=np.percentile(vals,percentile))&fg).astype(np.uint8)*255
        k=max(3,int(min(bw,bh)*.008)|1); dark=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
        contours,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area=cv2.contourArea(c)
            if not bw*bh*.001 <= area <= bw*bh*.13: continue
            x,y,w,h=cv2.boundingRect(c); cx=x+w/2+x0; cy=y+h/2+y0
            perimeter=cv2.arcLength(c,True); circular=4*np.pi*area/(perimeter*perimeter) if perimeter else 0
            allc.append({"contour":c+np.array([[[x0,y0]]]),"cx":cx,"cy":cy,"x":x+x0,"y":y+y0,"w":w,"h":h,"area":area,"rx":cx/bw-x0/bw,"ry":cy/bh-y0/bh,"circularity":circular,"percentile":percentile})
    return allc


def _best_orbit_pair(candidates: list[dict], box, axis: float):
    x0,y0,x1,y1=box; bw,bh=x1-x0+1,y1-y0+1
    pool=[c for c in candidates if .16 <= (c["cy"]-y0)/bh <= .55 and .08 <= c["area"]/(bw*bh)*10 <= .75 and .06 <= (c["cx"]-x0)/bw <= .94 and .55 <= c["w"]/max(c["h"],1) <= 2.2 and .12 <= abs(c["cx"]-axis)/bw <= .36]
    right=[c for c in pool if c["cx"]<axis]; left=[c for c in pool if c["cx"]>axis]
    best=None
    for a in right:
        for b in left:
            dy=abs(a["cy"]-b["cy"])/bh; da=abs((axis-a["cx"])-(b["cx"]-axis))/bw
            ratio=min(a["area"],b["area"])/max(a["area"],b["area"]); sep=(b["cx"]-a["cx"])/bw
            if dy>.13 or da>.18 or not .20<=sep<=.62: continue
            shape=(min(a["circularity"],1)+min(b["circularity"],1))/2; target=1-min(1,abs(sep-.40)/.22)
            score=.30*(1-dy/.13)+.20*(1-da/.18)+.20*ratio+.12*shape+.18*target
            if best is None or score>best[0]: best=(score,a,b)
    return best


def _consensus_stability(candidates: list[dict], chosen: dict, box) -> float:
    x0,y0,x1,y1=box; diag=max(np.hypot(x1-x0,y1-y0),1); support=set()
    for c in candidates:
        if np.hypot(c["cx"]-chosen["cx"],c["cy"]-chosen["cy"])/diag < .045: support.add(c["percentile"])
    return len(support)/5


def detect_geometry_v2(image: np.ndarray) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    h,w=image.shape[:2]; mask=foreground_mask(image); box=_anatomical_bbox(mask)
    if box is None: return {"status_cranio_geo":"FALHA","motivo_cranio_geo":"SEM_PRIMEIRO_PLANO","detector_versao":"v2"},[],mask,image.copy()
    x0,y0,x1,y1=box; bw,bh=x1-x0+1,y1-y0+1; axis,axis_conf=_symmetry_axis(image,box); candidates=_multi_threshold_candidates(image,mask,box); pair=_best_orbit_pair(candidates,box,axis)
    points=[]
    def add(name,x=np.nan,y=np.nan,conf=0,reason="NAO_DETECTADO"):
        visible=np.isfinite(x) and np.isfinite(y) and conf>0
        points.append({"ponto":name,"x_px":float(x),"y_px":float(y),"x_norm":float((x-x0)/bw) if visible else np.nan,"y_norm":float((y-y0)/bh) if visible else np.nan,"confianca":float(np.clip(conf,0,1)),"visivel":visible,"metodo":"geometria_2d_v2","motivo_ausencia":"" if visible else reason})
    od=oe=None; pair_conf=0
    if pair:
        pair_score,od,oe=pair; sd=_consensus_stability(candidates,od,box); se=_consensus_stability(candidates,oe,box); pair_conf=np.clip(.55*pair_score+.25*min(sd,se)+.20*axis_conf,0,1)
        add("orbita_direita",od["cx"],od["cy"],pair_conf); add("orbita_esquerda",oe["cx"],oe["cy"],pair_conf)
    else: add("orbita_direita"); add("orbita_esquerda")
    nasal=None; nasal_conf=0
    if od and oe:
        eye_y=(od["cy"]+oe["cy"])/2; inter=oe["cx"]-od["cx"]
        pool=[c for c in candidates if eye_y+.08*bh<c["cy"]<eye_y+.42*bh and .0015<c["area"]/(bw*bh)<.055 and abs(c["cx"]-axis)<max(.12*bw,.32*inter) and c["h"]>=c["w"]*.65]
        if pool:
            nasal=max(pool,key=lambda c:(c["area"]*(1-abs(c["cx"]-axis)/max(.2*bw,1)))); stability=_consensus_stability(candidates,nasal,box)
            pos=max(0,1-abs(nasal["cx"]-axis)/(.18*bw)); nasal_conf=np.clip(.45*stability+.35*pos+.20*pair_conf,0,1); add("nasal",nasal["cx"],nasal["cy"],nasal_conf)
        else: add("nasal")
    else: add("nasal",reason="ORBITAS_INSUFICIENTES")
    binary=mask>0; yy=np.arange(h); row_width=binary.sum(axis=1); central=binary[:,max(0,int(axis-.24*bw)):min(w,int(axis+.24*bw))].sum(axis=1)
    low0=y0+int(.62*bh); plausible=np.where((yy>=low0)&(yy<=y1)&(central>=max(8,int(.12*bw)))&(row_width>=max(15,int(.18*bw))))[0]
    if len(plausible):
        my=int(plausible.max()); xs=np.where(binary[my])[0]; near=xs[(xs>=axis-.28*bw)&(xs<=axis+.28*bw)]; mx=float(np.median(near)) if len(near) else axis; mconf=.85 if my<y1-2 else .48; add("mento",mx,my,mconf)
    else: add("mento")
    orbit_y=(od["cy"]+oe["cy"])/2 if od and oe else y0+.36*bh
    zrow=int(min(y1,max(y0,orbit_y+.12*bh))); band=binary[max(y0,zrow-int(.03*bh)):min(y1+1,zrow+int(.03*bh)+1)].any(axis=0); zx=np.where(band)[0]
    if len(zx): add("zigio_direito",np.percentile(zx,3),zrow,.52); add("zigio_esquerdo",np.percentile(zx,97),zrow,.52)
    else: add("zigio_direito"); add("zigio_esquerdo")
    grow=int(y0+.76*bh); band=binary[max(y0,grow-int(.04*bh)):min(y1+1,grow+int(.04*bh)+1)].any(axis=0); gx=np.where(band)[0]
    if len(gx): add("gonio_direito",np.percentile(gx,8),grow,.50); add("gonio_esquerdo",np.percentile(gx,92),grow,.50)
    else: add("gonio_direito"); add("gonio_esquerdo")
    touches={"topo_truncado":y0<=3,"base_truncada":y1>=h-4,"lateral_esq_truncada":x0<=3,"lateral_dir_truncada":x1>=w-4}
    valid={p["ponto"]:p for p in points if p["visivel"]}; core=sum(k in valid and valid[k]["confianca"]>=.45 for k in ["orbita_direita","orbita_esquerda","nasal","mento"]); status="OK" if core==4 else ("PARCIAL" if core>=2 else "INCERTO")
    metrics={"status_cranio_geo":status,"motivo_cranio_geo":"" if core>=2 else "REGIOES_CENTRAIS_INSUFICIENTES","detector_versao":"v2",**touches,"bbox_x":x0,"bbox_y":y0,"bbox_w":bw,"bbox_h":bh,"area_relativa":float(binary.mean()),"eixo_simetria_x":axis,"confianca_eixo":axis_conf,"confianca_par_orbital":pair_conf,"confianca_nasal":nasal_conf,"mandibula_presente":"mento" in valid,"regioes_centrais_detectadas":core}
    if od and oe: metrics["distancia_interorbital_norm"]=abs(oe["cx"]-od["cx"])/bw; metrics["assimetria_orbital_area"]=abs(od["area"]-oe["area"])/max(od["area"],oe["area"],1)
    if nasal: metrics["largura_nasal_norm"]=nasal["w"]/bw; metrics["altura_nasal_norm"]=nasal["h"]/bh
    if "mento" in valid and od and oe: metrics["orbita_mento_norm"]=(valid["mento"]["y_px"]-(od["cy"]+oe["cy"])/2)/bh
    for a,b,key in [("gonio_direito","gonio_esquerdo","largura_mandibular_norm"),("zigio_direito","zigio_esquerdo","largura_zigomatica_norm")]:
        if a in valid and b in valid: metrics[key]=abs(valid[b]["x_px"]-valid[a]["x_px"])/bw
    preview=image.copy(); cv2.rectangle(preview,(x0,y0),(x1,y1),(180,180,180),2); cv2.line(preview,(int(axis),y0),(int(axis),y1),(255,255,0),2)
    for p in points:
        if p["visivel"]:
            xy=(int(p["x_px"]),int(p["y_px"])); color=POINT_COLORS.get(p["ponto"],(0,255,0)); cv2.circle(preview,xy,max(4,min(h,w)//160),color,-1); cv2.putText(preview,f"{p['ponto']} {p['confianca']:.2f}",(xy[0]+5,xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
    return metrics,points,mask,preview


def _orbital_response(gray: np.ndarray, box) -> np.ndarray:
    """Resposta a cavidades: interior escuro em relacao ao osso vizinho."""
    x0, y0, x1, y1 = box; bw, bh = x1-x0+1, y1-y0+1
    roi = gray[y0:y1+1, x0:x1+1]
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(roi)
    small = max(5, (int(min(bw, bh) * .025) | 1))
    large = max(small + 2, (int(min(bw, bh) * .12) | 1))
    # Box filters sao equivalentes o bastante para a resposta local e muito mais
    # rapidos que GaussianBlur com kernels grandes nas capturas de alta resolucao.
    local = cv2.boxFilter(clahe, cv2.CV_32F, (large, large), normalize=True)
    detail = cv2.boxFilter(clahe, cv2.CV_32F, (small, small), normalize=True)
    response = np.maximum(local - detail, 0)
    return cv2.normalize(response, None, 0, 1, cv2.NORM_MINMAX)


def _template_orbit_pair(image: np.ndarray, mask: np.ndarray, box, axis: float, prior=None):
    """Busca um par simetrico por janelas elipticas multiescala e estabilidade."""
    x0, y0, x1, y1 = box; bw, bh = x1-x0+1, y1-y0+1
    prior = prior or {"ry": .36, "sep": .40, "ww": .18, "hh": .13}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants = [gray, cv2.convertScaleAbs(gray, alpha=1.15, beta=-12), cv2.GaussianBlur(gray, (5, 5), 0)]
    detections = []
    for variant in variants:
        response = _orbital_response(variant, box)
        best = None
        for ry in np.linspace(max(.35, prior["ry"]-.06), min(.52, prior["ry"]+.10), 9):
            cy = y0 + ry*bh
            for sep in np.linspace(max(.32, prior["sep"]-.08), min(.52, prior["sep"]+.10), 9):
                centers = (axis-sep*bw/2, axis+sep*bw/2)
                for ww in (prior["ww"]*.86, prior["ww"]*1.14):
                    hh = prior["hh"]
                    scores=[]
                    for cx in centers:
                        rx, ry0 = int(max(5, ww*bw/2)), int(max(4, hh*bh/2))
                        lx, ly = int(cx-x0), int(cy-y0)
                        if lx-rx<0 or lx+rx>=bw or ly-ry0<0 or ly+ry0>=bh:
                            scores=[]; break
                        patch=response[ly-ry0:ly+ry0+1,lx-rx:lx+rx+1]
                        yy,xx=np.ogrid[-ry0:ry0+1,-rx:rx+1]; ellipse=(xx*xx/(rx*rx)+yy*yy/(ry0*ry0))<=1
                        inner=float(patch[ellipse].mean()); ring=(xx*xx/(rx*rx)+yy*yy/(ry0*ry0)<=1.65)&~ellipse
                        contrast=inner-float(patch[ring].mean()) if ring.any() else inner
                        scores.append(.7*inner+.3*max(contrast,0))
                    if len(scores)!=2: continue
                    balance=min(scores)/max(max(scores),1e-6)
                    symmetry_penalty=abs(scores[0]-scores[1])
                    score=float(np.mean(scores)*(.65+.35*balance)-.12*symmetry_penalty)
                    if best is None or score>best[0]: best=(score,centers,cy,ww,hh,scores)
        if best: detections.append(best)
    if len(detections)<3: return None
    centers=np.asarray([[d[1][0],d[2],d[1][1],d[2]] for d in detections])
    spread=float(np.max(np.std(centers,axis=0)/np.array([bw,bh,bw,bh])))
    consensus=np.median(centers,axis=0); raw=float(np.median([d[0] for d in detections]))
    stability=float(np.clip(1-spread/.035,0,1))
    # O limiar foi deliberadamente alto para nao converter textura em orbita.
    visual=float(np.clip((raw-.10)/.15,0,1))
    confidence=float(np.clip(.35*stability+.65*visual,0,1))
    if raw < .125 or confidence < .45: return None
    ww=float(np.median([d[3] for d in detections])); hh=float(np.median([d[4] for d in detections]))
    def item(cx,cy):
        return {"cx":float(cx),"cy":float(cy),"w":ww*bw,"h":hh*bh,"area":np.pi*ww*bw*hh*bh/4,
                "x":cx-ww*bw/2,"y":cy-hh*bh/2,"circularity":.7,"percentile":-1}
    return confidence,item(consensus[0],consensus[1]),item(consensus[2],consensus[3]),stability,raw


def _orbital_prior(rows: list[tuple[dict, list[dict]]]) -> dict:
    values=[]
    for metrics, points in rows:
        if float(metrics.get("confianca_par_orbital",0)) < .55: continue
        p={x["ponto"]:x for x in points if x.get("visivel")}
        if not all(k in p for k in ("orbita_direita","orbita_esquerda")): continue
        sep=p["orbita_esquerda"]["x_norm"]-p["orbita_direita"]["x_norm"]
        values.append(((p["orbita_esquerda"]["y_norm"]+p["orbita_direita"]["y_norm"])/2,sep))
    if not values: return {"ry":.41,"sep":.40,"ww":.18,"hh":.13,"n":0}
    a=np.asarray(values)
    # O V2 fornece apenas uma inicializacao; limites anatomicos impedem que
    # deteccoes altas demais (rebordo supraorbital) contaminem o template.
    return {"ry":float(np.clip(np.median(a[:,0]),.39,.44)),"sep":float(np.clip(np.median(a[:,1]),.34,.46)),"ww":.18,"hh":.13,"n":len(a)}


def detect_geometry_v3(image: np.ndarray, prior=None) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    metrics, points, mask, _ = detect_geometry_v2(image)
    metrics=dict(metrics); points=[dict(p) for p in points]; metrics["detector_versao"]="v3"
    box=_anatomical_bbox(mask)
    if box is None: return metrics,points,mask,image.copy()
    byname={p["ponto"]:p for p in points}; v2_pair=all(byname.get(k,{}).get("confianca",0)>=.45 for k in ("orbita_direita","orbita_esquerda"))
    fallback=None
    if not v2_pair:
        fallback=_template_orbit_pair(image,mask,box,float(metrics["eixo_simetria_x"]),prior)
        if fallback:
            conf,od,oe,stability,raw=fallback
            for name,c in (("orbita_direita",od),("orbita_esquerda",oe)):
                p=byname[name]; p.update({"x_px":c["cx"],"y_px":c["cy"],"x_norm":(c["cx"]-box[0])/(box[2]-box[0]+1),
                    "y_norm":(c["cy"]-box[1])/(box[3]-box[1]+1),"confianca":conf,"visivel":True,
                    "metodo":"template_orbital_multiescala_v3","motivo_ausencia":""})
            metrics.update({"confianca_par_orbital":conf,"estabilidade_template_orbital":stability,"resposta_template_orbital":raw,
                            "distancia_interorbital_norm":abs(oe["cx"]-od["cx"])/(box[2]-box[0]+1)})
    # Recalcula nariz usando a geometria orbital recuperada, sem forcar deteccao.
    byname={p["ponto"]:p for p in points}; od,oe=byname["orbita_direita"],byname["orbita_esquerda"]
    if fallback and not byname["nasal"].get("visivel"):
        candidates=_multi_threshold_candidates(image,mask,box); axis=float(metrics["eixo_simetria_x"]); bh=box[3]-box[1]+1; bw=box[2]-box[0]+1
        eye_y=(od["y_px"]+oe["y_px"])/2; pool=[c for c in candidates if eye_y+.07*bh<c["cy"]<eye_y+.43*bh and abs(c["cx"]-axis)<.16*bw and c["h"]>=c["w"]*.55]
        if pool:
            nasal=max(pool,key=lambda c:c["area"]*(1-abs(c["cx"]-axis)/(.16*bw)))
            stability=_consensus_stability(candidates,nasal,box); conf=float(np.clip(.48*stability+.32*(1-abs(nasal["cx"]-axis)/(.16*bw))+.20*od["confianca"],0,1))
            if conf>=.45:
                byname["nasal"].update({"x_px":nasal["cx"],"y_px":nasal["cy"],"x_norm":(nasal["cx"]-box[0])/bw,"y_norm":(nasal["cy"]-box[1])/bh,
                    "confianca":conf,"visivel":True,"metodo":"nasal_condicionado_orbitas_v3","motivo_ausencia":""})
                metrics.update({"confianca_nasal":conf,"largura_nasal_norm":nasal["w"]/bw,"altura_nasal_norm":nasal["h"]/bh})
    valid={p["ponto"]:p for p in points if p.get("visivel")}; core=sum(k in valid and valid[k]["confianca"]>=.45 for k in ["orbita_direita","orbita_esquerda","nasal","mento"])
    metrics["regioes_centrais_detectadas"]=core; metrics["status_cranio_geo"]="OK" if core==4 else ("PARCIAL" if core>=2 else "INCERTO")
    metrics["motivo_cranio_geo"]="" if core>=2 else "REGIOES_CENTRAIS_INSUFICIENTES"
    if all(k in valid for k in ("orbita_direita","orbita_esquerda","mento")):
        metrics["orbita_mento_norm"]=(valid["mento"]["y_px"]-(valid["orbita_direita"]["y_px"]+valid["orbita_esquerda"]["y_px"])/2)/(box[3]-box[1]+1)
    preview=image.copy(); cv2.rectangle(preview,(box[0],box[1]),(box[2],box[3]),(180,180,180),2); cv2.line(preview,(int(metrics["eixo_simetria_x"]),box[1]),(int(metrics["eixo_simetria_x"]),box[3]),(255,255,0),2)
    for p in points:
        if p.get("visivel"):
            xy=(int(p["x_px"]),int(p["y_px"])); color=POINT_COLORS.get(p["ponto"],(0,255,0)); cv2.circle(preview,xy,max(4,min(image.shape[:2])//160),color,-1); cv2.putText(preview,f"{p['ponto']} {p['confianca']:.2f}",(xy[0]+5,xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
    return metrics,points,mask,preview


def detect_geometry(image: np.ndarray):
    return detect_geometry_v3(image)


def run_cranios(manifest: pd.DataFrame, config: dict, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = LAB_ROOT / "resultados/cranios/metricas_cranios_cf.csv"; points_path = LAB_ROOT / "resultados/cranios/pontos_geometrico_anatomicos.csv"
    if metrics_path.exists() and points_path.exists() and not force:
        return pd.read_csv(metrics_path), pd.read_csv(points_path)
    # Primeira passagem: V2 gera um prior robusto apenas a partir do treino.
    cached=[]
    for _,rec in manifest.iterrows():
        p=abs_repo(rec.get("imagem_cf")); im=read_image(p) if p and p.exists() else None
        if im is not None:
            cached.append((rec,im,detect_geometry_v2(im)))
    split_path=LAB_ROOT/"dataset/splits/split_identidades.csv"
    if split_path.exists():
        split_df=pd.read_csv(split_path); train_ids=set(split_df.loc[split_df["split"].eq("treino"),"id_pessoa"].astype(str))
    else:
        train_ids=set(manifest.loc[manifest.get("split",pd.Series(index=manifest.index,dtype=object)).eq("treino"),"id_pessoa"].astype(str)) if "split" in manifest else set()
    prior_source=[(result[0],result[1]) for rec,_,result in cached if not train_ids or str(rec["id_pessoa"]) in train_ids]
    prior=_orbital_prior(prior_source)
    rows, pts, rows_v1, rows_v2 = [], [], [], []
    for index,(rec,im,v2) in enumerate(cached,1):
        base = {"ano": rec["ano"], "id_pessoa": rec["id_pessoa"], "imagem_cf": rec.get("imagem_cf", "")}
        if im is None:
            rows.append({**base, "status_cranio_geo": "FALHA", "motivo_cranio_geo": "IMAGEM_INVALIDA"}); continue
        metrics_v1, _, _, preview_v1 = detect_geometry_v1(im); rows_v1.append({**base, **metrics_v1})
        metrics_v2, points_v2, _, preview_v2 = v2; rows_v2.append({**base, **metrics_v2})
        metrics, points, mask, preview = detect_geometry_v3(im,prior); rows.append({**base, **metrics})
        for point in points: pts.append({**base, **point})
        write_image(LAB_ROOT / "resultados/cranios/mascaras" / f"{rec['id_pessoa']}_CF.png", mask)
        write_image(LAB_ROOT / "resultados/cranios/previews" / f"{rec['id_pessoa']}_CF.jpg", preview)
        write_image(LAB_ROOT / "resultados/cranios/previews_v1" / f"{rec['id_pessoa']}_CF.jpg", preview_v1)
        write_image(LAB_ROOT / "resultados/cranios/previews_v2" / f"{rec['id_pessoa']}_CF.jpg", preview_v2)
        if index % 50 == 0 or index == len(cached): print(f"[V3] cranios {index}/{len(cached)}",flush=True)
    mdf, pdf = pd.DataFrame(rows), pd.DataFrame(pts); save_csv(mdf, metrics_path); save_csv(pdf, points_path); save_csv(pd.DataFrame(rows_v1), LAB_ROOT / "resultados/cranios/metricas_cranios_cf_v1.csv"); save_csv(pd.DataFrame(rows_v2), LAB_ROOT / "resultados/cranios/metricas_cranios_cf_v2.csv")
    pd.DataFrame([prior]).to_csv(LAB_ROOT/"resultados/cranios/prior_orbital_v3.csv",index=False)
    eligible = manifest[manifest["usar_matching"]][["id_pessoa"]].merge(mdf, on="id_pessoa", how="left")
    coverage = float((eligible.get("regioes_centrais_detectadas", 0) >= 4).mean()) if len(eligible) else 0
    gate = pd.DataFrame([{"elegiveis": len(eligible), "cobertura_core": coverage, "limite": config["cranios"]["gate_cobertura"], "ativar_anatomia_matching": coverage >= config["cranios"]["gate_cobertura"]}])
    save_csv(gate, LAB_ROOT / "resultados/cranios/gate_viabilidade.csv")
    return mdf, pdf
