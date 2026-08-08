from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from .paths import PROJECT_DIR, REPO_ROOT, relative_repo_path, repo_path


def _stats(path: Path) -> dict:
    with Image.open(path) as raw:
        image=ImageOps.exif_transpose(raw).convert("L"); width,height=image.size
        image.thumbnail((512,512),Image.Resampling.BILINEAR); gray=np.asarray(image,dtype=np.uint8)
        p05,p95=np.percentile(gray,[5,95])
        return {"width":width,"height":height,"aspect":width/max(height,1),"brightness_mean":float(gray.mean()),"brightness_std":float(gray.std()),"p05":float(p05),"p95":float(p95),"dynamic_range":float(p95-p05),"dark_fraction":float(np.mean(gray<20)),"light_fraction":float(np.mean(gray>235))}


def _contact_sheet(items: list[tuple[str,Path]], output: Path, thumb=180, columns=5) -> None:
    rows=math.ceil(len(items)/columns); sheet=Image.new("RGB",(columns*thumb,rows*(thumb+28)),"white"); draw=ImageDraw.Draw(sheet)
    for i,(label,path) in enumerate(items):
        with Image.open(path) as raw:
            image=ImageOps.contain(ImageOps.exif_transpose(raw).convert("RGB"),(thumb-8,thumb-8))
        x=(i%columns)*thumb+(thumb-image.width)//2; y=(i//columns)*(thumb+28)+(thumb-image.height)//2
        sheet.paste(image,(x,y)); draw.text(((i%columns)*thumb+5,(i//columns)*(thumb+28)+thumb+3),label[:25],fill="black")
    output.parent.mkdir(parents=True,exist_ok=True); sheet.save(output,"JPEG",quality=88)


def audit_domains(metadata: pd.DataFrame, samples_per_group: int=10, seed: int=42) -> tuple[pd.DataFrame,pd.DataFrame]:
    valid=metadata[metadata.usar_treino.fillna(False)].copy(); rng=np.random.default_rng(seed); rows=[]; gallery_root=PROJECT_DIR/"reports/domain_audit/galleries"
    columns={"face":"face_path","CF":"cranio_cf_path","CE":"cranio_ce_path","CD":"cranio_cd_path"}
    for _,record in valid.iterrows():
        for modality,column in columns.items():
            path=repo_path(record.get(column))
            if not path or not path.exists(): continue
            try: rows.append({"pair_key":record.pair_key,"ano":str(record.ano),"modality":modality,"path":relative_repo_path(path),**_stats(path)})
            except Exception as exc: rows.append({"pair_key":record.pair_key,"ano":str(record.ano),"modality":modality,"path":relative_repo_path(path),"error":str(exc)})
    details=pd.DataFrame(rows); numeric=["width","height","aspect","brightness_mean","brightness_std","dynamic_range","dark_fraction","light_fraction"]
    summary=details.groupby(["ano","modality"])[numeric].agg(["count","mean","std","min","max"]).reset_index(); summary.columns=["_".join(str(x) for x in c if x).rstrip("_") if isinstance(c,tuple) else c for c in summary.columns]
    out=PROJECT_DIR/"reports/domain_audit"; out.mkdir(parents=True,exist_ok=True); details.to_csv(out/"image_metrics.csv",index=False); summary.to_csv(out/"year_summary.csv",index=False)
    for (year,modality),group in details.dropna(subset=["path"]).groupby(["ano","modality"]):
        take=min(samples_per_group,len(group)); chosen=group.iloc[rng.choice(len(group),take,replace=False)]
        _contact_sheet([(str(r.pair_key),repo_path(r.path)) for _,r in chosen.iterrows()],gallery_root/f"{year}_{modality}.jpg")
    report={"images_audited":int(len(details)),"pairs":int(valid.pair_key.nunique()),"years":sorted(details.ano.unique().tolist()),"modalities":list(columns),"note":"Métricas descrevem domínio de captura; não são medidas anatômicas."}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return details,summary


def _face_landmarks(year: str, image_id: str) -> pd.DataFrame:
    path=REPO_ROOT/"Treinamento_faces/dados_originais/faces"/str(year)/"marcacoes_landmarks/landmarks_csv"/f"{image_id}_landmarks.csv"
    return pd.read_csv(path,encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def _letterbox(image: Image.Image, size: int, background=(127,127,127), occupancy=0.88) -> Image.Image:
    limit=max(1,int(size*occupancy)); image=ImageOps.contain(image,(limit,limit),Image.Resampling.LANCZOS); canvas=Image.new("RGB",(size,size),background); canvas.paste(image,((size-image.width)//2,(size-image.height)//2)); return canvas


def normalize_face(path: Path, year: str, image_id: str, size: int=224, masked: bool=True) -> tuple[Image.Image,str]:
    with Image.open(path) as raw:
        image=ImageOps.exif_transpose(raw).convert("RGB").copy()
    landmarks=_face_landmarks(year,image_id)
    if landmarks.empty or not {"x_px","y_px","idx"}.issubset(landmarks):
        return _letterbox(ImageOps.autocontrast(image),size),"fallback_sem_landmarks"
    xs=landmarks.x_px.to_numpy(float); ys=landmarks.y_px.to_numpy(float); x1,x2=xs.min(),xs.max(); y1,y2=ys.min(),ys.max(); w=x2-x1; h=y2-y1
    margin_x=.20*w; margin_y=.16*h; box=(max(0,int(x1-margin_x)),max(0,int(y1-margin_y)),min(image.width,int(x2+margin_x)),min(image.height,int(y2+margin_y)))
    crop=image.crop(box); lookup=landmarks.set_index("idx")
    status="crop_landmarks"
    if 33 in lookup.index and 263 in lookup.index:
        left=lookup.loc[33]; right=lookup.loc[263]; angle=math.degrees(math.atan2(float(right.y_px-left.y_px),float(right.x_px-left.x_px)))
        crop=crop.rotate(angle,Image.Resampling.BICUBIC,expand=False,fillcolor=(127,127,127)); status="alinhada_olhos"
    crop=ImageOps.autocontrast(crop,cutoff=1); crop=ImageEnhance.Contrast(crop).enhance(1.05)
    output=_letterbox(crop,size)
    if masked:
        mask=Image.new("L",(size,size),0); draw=ImageDraw.Draw(mask); draw.ellipse((int(size*.16),int(size*.06),int(size*.84),int(size*.96)),fill=255)
        output=Image.composite(output,Image.new("RGB",(size,size),(127,127,127)),mask); status+="_mascarada"
    return output,status


def normalize_skull(path: Path, size: int=224) -> tuple[Image.Image,str]:
    with Image.open(path) as raw:
        image=ImageOps.exif_transpose(raw).convert("L").copy()
    return normalize_skull_image(image, size)


def normalize_skull_image(image: Image.Image, size: int=224) -> tuple[Image.Image,str]:
    image=ImageOps.exif_transpose(image).convert("L")
    arr=np.asarray(image,dtype=np.uint8); p10,p99=np.percentile(arr,[10,99]); scaled=np.clip((arr.astype(float)-p10)*255/max(p99-p10,1),0,255).astype(np.uint8)
    # Detecta conteúdo por distância ao tom dominante das bordas.
    border=np.concatenate([scaled[0],scaled[-1],scaled[:,0],scaled[:,-1]]); bg=float(np.median(border)); mask=np.abs(scaled.astype(float)-bg)>12; ys,xs=np.where(mask)
    normalized=Image.fromarray(scaled).convert("RGB")
    status="contraste"
    if len(xs)>50:
        x1,x2,y1,y2=xs.min(),xs.max()+1,ys.min(),ys.max()+1; pad_x=max(2,int((x2-x1)*.06)); pad_y=max(2,int((y2-y1)*.06)); normalized=normalized.crop((max(0,x1-pad_x),max(0,y1-pad_y),min(image.width,x2+pad_x),min(image.height,y2+pad_y))); status="crop_conteudo_contraste"
    return _letterbox(normalized,size,background=(0,0,0),occupancy=.88),status


def normalize_uploaded_face(image: Image.Image, size: int=224) -> Image.Image:
    """Padronização conservadora para uma face frontal já recortada pelo usuário."""
    image=ImageOps.exif_transpose(image).convert("RGB")
    image=ImageOps.autocontrast(image,cutoff=1)
    image=ImageEnhance.Contrast(image).enhance(1.05)
    output=_letterbox(image,size,background=(127,127,127),occupancy=.88)
    mask=Image.new("L",(size,size),0)
    ImageDraw.Draw(mask).ellipse((int(size*.16),int(size*.06),int(size*.84),int(size*.96)),fill=255)
    return Image.composite(output,Image.new("RGB",(size,size),(127,127,127)),mask)


def _multiview_face_path(source_root: Path, year: str, person_id: str, view: str) -> Path | None:
    year_root=source_root/str(year)
    normalized_person=str(person_id).replace(" ","_").casefold()
    folders=[folder for folder in year_root.iterdir() if folder.is_dir() and not folder.name.startswith("_") and folder.name.replace(" ","_").casefold()==normalized_person] if year_root.is_dir() else []
    for folder in folders:
        for candidate in folder.iterdir():
            if not candidate.is_file() or candidate.suffix.lower() not in {".jpg",".jpeg",".png"}: continue
            stem=candidate.stem.replace(" ","_").casefold()
            if stem.startswith(normalized_person) and stem.endswith(f"_{view.casefold()}"): return candidate
    return None


def normalize_face_without_landmarks(path: Path, size: int=224) -> tuple[Image.Image,str]:
    """Normalização conservadora para FFS/FL, que não possuem landmarks homologados."""
    with Image.open(path) as raw:
        image=ImageOps.exif_transpose(raw).convert("RGB").copy()
    image=ImageOps.autocontrast(image,cutoff=1)
    image=ImageEnhance.Contrast(image).enhance(1.05)
    return _letterbox(image,size,background=(127,127,127),occupancy=.92),"letterbox_sem_landmarks"


def normalize_domains(metadata: pd.DataFrame, size: int=224, force: bool=False, limit: int|None=None, face_views=("FFN",), face_source_root: str|Path|None=None) -> pd.DataFrame:
    valid=metadata[metadata.usar_treino.fillna(False)].copy()
    if limit: valid=valid.head(limit)
    root=PROJECT_DIR/"data/domain_normalized"/f"v1_{size}"; rows=[]
    views=tuple(str(view).upper() for view in face_views)
    unknown=set(views)-{"FFN","FFS","FL"}
    if unknown: raise ValueError(f"Modalidades faciais desconhecidas: {sorted(unknown)}")
    source_root=Path(face_source_root).expanduser() if face_source_root else None
    if set(views)-{"FFN"} and (source_root is None or not source_root.is_dir()):
        raise FileNotFoundError(f"Banco facial multivista não encontrado: {source_root}")
    for _,record in valid.iterrows():
        rec={"pair_key":record.pair_key,"ano":str(record.ano),"status":"OK","error":""}
        try:
            face_out=root/"face"/str(record.ano)/f"{record.id_face}.png"
            if force or not face_out.exists(): image,status=normalize_face(repo_path(record.face_path),str(record.ano),str(record.id_face),size); face_out.parent.mkdir(parents=True,exist_ok=True); image.save(face_out)
            else: status="ja_existia"
            rec.update(normalized_face_path=relative_repo_path(face_out),face_normalization=status)
            for view in ("FFS","FL"):
                if view not in views: continue
                source=_multiview_face_path(source_root,str(record.ano),str(record.id_pessoa),view)
                if source is None: raise FileNotFoundError(f"{record.id_pessoa}: fotografia {view} ausente em {source_root}")
                target=root/f"face_{view.lower()}"/str(record.ano)/f"{record.id_pessoa}_{view}.png"
                if force or not target.exists():
                    image,view_status=normalize_face_without_landmarks(source,size); target.parent.mkdir(parents=True,exist_ok=True); image.save(target)
                else: view_status="ja_existia"
                rec[f"normalized_face_{view.lower()}_path"]=relative_repo_path(target)
                rec[f"face_{view.lower()}_source_path"]=str(source)
                rec[f"face_{view.lower()}_normalization"]=view_status
            for view,column in (("CF","cranio_cf_path"),("CE","cranio_ce_path"),("CD","cranio_cd_path")):
                source=repo_path(record.get(column)); target=root/view/str(record.ano)/f"{record.id_pessoa}_{view}.png"
                if force or not target.exists(): image,sk_status=normalize_skull(source,size); target.parent.mkdir(parents=True,exist_ok=True); image.save(target)
                else: sk_status="ja_existia"
                rec[f"normalized_cranio_{view.lower()}_path"]=relative_repo_path(target); rec[f"{view.lower()}_normalization"]=sk_status
        except Exception as exc: rec.update(status="FALHA",error=str(exc))
        rows.append(rec)
        if len(rows)%50==0: print(f"Normalizados: {len(rows)}/{len(valid)}",flush=True)
    result=pd.DataFrame(rows); root.mkdir(parents=True,exist_ok=True); result.to_csv(root/"normalized_manifest.csv",index=False); print(f"Concluído: {int(result.status.eq('OK').sum())} OK, {int(result.status.eq('FALHA').sum())} falhas",flush=True); return result
