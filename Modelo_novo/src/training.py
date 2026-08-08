from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageOps

from .model import TwoTower, require_torch, symmetric_hard_negative_loss, symmetric_infonce, torch
from .paths import PROJECT_DIR, repo_path
from .retrieval import retrieval_metrics, sampled_gallery_metrics
from .experiments import save_provenance


def _stack():
    require_torch()
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    return Image, DataLoader, Dataset, transforms


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_dataset_class(image_size=224, skull_views=("CF",), augmentation="none", use_shape64=False, domain_harmonization="none", face_views=("FFN",)):
    Image, _, Dataset, transforms = _stack()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    evaluation = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])
    # Alterações suaves: não espelhar, pois lateralidade pode carregar informação anatômica.
    robust=domain_harmonization=="grayscale_equalized"
    face_train = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomAffine(5 if robust else 4, translate=(0.035, 0.035) if robust else (0.025, 0.025), scale=(0.95, 1.05) if robust else (0.97, 1.03)), transforms.ColorJitter(brightness=0.16 if robust else 0.08, contrast=0.16 if robust else 0.08), transforms.RandomApply([transforms.GaussianBlur(3,sigma=(0.1,1.0))],p=.20 if robust else 0.0), transforms.ToTensor(), normalize])
    skull_train = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomAffine(4 if robust else 3, translate=(0.03, 0.03) if robust else (0.02, 0.02), scale=(0.96, 1.04) if robust else (0.98, 1.02)), transforms.ColorJitter(brightness=0.14 if robust else 0.05, contrast=0.16 if robust else 0.08), transforms.RandomApply([transforms.GaussianBlur(3,sigma=(0.1,1.0))],p=.20 if robust else 0.0), transforms.ToTensor(), normalize])
    def harmonize(image):
        if not robust: return image
        return ImageOps.equalize(image.convert("L")).convert("RGB")
    class PairDataset(Dataset):
        def __init__(self, frame, training=False): self.frame = frame.reset_index(drop=True); self.training = training
        def __len__(self): return len(self.frame)
        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            ft = face_train if self.training and augmentation != "none" else evaluation
            st = skull_train if self.training and augmentation != "none" else evaluation
            face_columns = {"FFN":"face_path", "FFS":"face_ffs_path", "FL":"face_fl_path"}
            face_images=[]
            for view in face_views:
                path=repo_path(row.get(face_columns[view]))
                if not path or not path.exists(): raise FileNotFoundError(f"Sem vista facial {view} para {row['pair_key']}")
                face_images.append(ft(harmonize(Image.open(path).convert("RGB"))))
            face=face_images[0] if len(face_images)==1 else torch.stack(face_images)
            path_columns = {"CF":"cranio_cf_path", "CE":"cranio_ce_path", "CD":"cranio_cd_path"}
            images = []
            for view in skull_views:
                path = repo_path(row.get(path_columns[view]))
                if path and path.exists(): images.append(st(harmonize(Image.open(path).convert("RGB"))))
            if not images: raise FileNotFoundError(f"Sem vista craniana para {row['pair_key']}")
            skull = torch.stack(images)
            if use_shape64:
                face_shape=torch.tensor(row[[f"face_shape_{i:03d}_{axis}" for i in range(64) for axis in ("x","y")]].to_numpy(dtype=np.float32))
                skull_shape=torch.tensor(row[[f"cranio_shape_{i:03d}_{axis}" for i in range(64) for axis in ("x","y")]].to_numpy(dtype=np.float32))
                return face,skull,face_shape,skull_shape,str(row["pair_key"])
            return face, skull, str(row["pair_key"])
    return PairDataset


def _encode(model, loader, device):
    model.eval(); faces=[]; skulls=[]; ids=[]
    with torch.no_grad():
        for batch in loader:
            if len(batch)==5:
                face,skull,face_shape,skull_shape,keys=batch; f,s=model(face.to(device),skull.to(device),face_shape.to(device),skull_shape.to(device))
            else:
                face,skull,keys=batch; f,s=model(face.to(device),skull.to(device))
            faces.append(f.cpu()); skulls.append(s.cpu()); ids.extend(keys)
    return torch.cat(faces).numpy(), torch.cat(skulls).numpy(), ids


def mine_global_hard_neighbors(face_embeddings: np.ndarray, skull_embeddings: np.ndarray, ids: list[str], top_k: int=5) -> dict[str,list[str]]:
    """Seleciona negativos globais nas duas direções, excluindo o par correto."""
    if len(ids)<2: return {str(key):[] for key in ids}
    faces=np.clip(np.nan_to_num(np.asarray(face_embeddings,dtype=np.float64),nan=0.0,posinf=1.0,neginf=-1.0),-1.0,1.0)
    skulls=np.clip(np.nan_to_num(np.asarray(skull_embeddings,dtype=np.float64),nan=0.0,posinf=1.0,neginf=-1.0),-1.0,1.0)
    similarity=faces@skulls.T
    k=min(max(1,int(top_k)),len(ids)-1); keys=np.asarray(ids,dtype=str); neighbors={str(key):[] for key in keys}
    for matrix in (similarity,similarity.T):
        masked=matrix.copy(); np.fill_diagonal(masked,-np.inf)
        selected=np.argpartition(masked,-k,axis=1)[:,-k:]
        for i,row in enumerate(selected):
            ordered=row[np.argsort(masked[i,row])[::-1]]
            for j in ordered:
                candidate=str(keys[j])
                if candidate not in neighbors[str(keys[i])]: neighbors[str(keys[i])].append(candidate)
    return neighbors


def hard_negative_batches(ids: list[str], neighbors: dict[str,list[str]], batch_size: int, seed: int) -> list[list[int]]:
    """Monta lotes únicos por época, agrupando cada âncora com negativos minerados."""
    if batch_size<2: raise ValueError("batch_size precisa ser pelo menos 2")
    rng=np.random.default_rng(seed); order=np.arange(len(ids)); rng.shuffle(order); index={str(key):i for i,key in enumerate(ids)}
    remaining=set(int(i) for i in order); batches=[]
    while remaining:
        anchor=next(int(i) for i in order if int(i) in remaining); batch_indices=[anchor]; remaining.remove(anchor)
        for key in neighbors.get(str(ids[anchor]),[]):
            candidate=index.get(str(key))
            if candidate in remaining and len(batch_indices)<batch_size: batch_indices.append(candidate); remaining.remove(candidate)
        if len(batch_indices)<batch_size:
            fillers=[int(i) for i in order if int(i) in remaining][:batch_size-len(batch_indices)]
            batch_indices.extend(fillers); remaining.difference_update(fillers)
        batches.append(batch_indices)
    return batches


def apply_domain_normalized_paths(splits: pd.DataFrame, image_size: int=224) -> pd.DataFrame:
    path=PROJECT_DIR/"data/domain_normalized"/f"v1_{image_size}"/"normalized_manifest.csv"
    if not path.exists(): raise FileNotFoundError(f"Manifesto normalizado ausente: {path}. Execute normalize-images primeiro.")
    normalized=pd.read_csv(path); required={"pair_key","normalized_face_path","normalized_cranio_cf_path","normalized_cranio_ce_path","normalized_cranio_cd_path"}
    if not required.issubset(normalized.columns): raise ValueError(f"Manifesto normalizado sem colunas: {sorted(required-set(normalized.columns))}")
    optional=[column for column in ("normalized_face_ffs_path","normalized_face_fl_path") if column in normalized.columns]
    merged=splits.merge(normalized[list(required)+optional],on="pair_key",how="left",validate="one_to_one")
    if merged["normalized_face_path"].isna().any(): raise ValueError("Há pares do split sem normalização")
    assignments=dict(face_path=merged.normalized_face_path,cranio_cf_path=merged.normalized_cranio_cf_path,cranio_ce_path=merged.normalized_cranio_ce_path,cranio_cd_path=merged.normalized_cranio_cd_path)
    for view,column in (("ffs","normalized_face_ffs_path"),("fl","normalized_face_fl_path")):
        if column in merged: assignments[f"face_{view}_path"]=merged[column]
    return merged.assign(**assignments)


def apply_shape64_features(splits: pd.DataFrame) -> pd.DataFrame:
    face=pd.read_csv(PROJECT_DIR.parent/"Treinamento_faces/resultados/features_matching_faces/shape_faces_64pts.csv")
    skull=pd.read_csv(PROJECT_DIR.parent/"Treinamento_cranios/resultados/features_matching_cranios/shape_cranios_64pts.csv")
    for frame in (face,skull): frame["ano"]=frame.ano.astype(str); frame["id_pessoa"]=frame.id_pessoa.astype(str)
    out=splits.copy(); out["ano"]=out.ano.astype(str); out["id_pessoa"]=out.id_pessoa.astype(str)
    fcols=[f"face_shape_{i:03d}_{axis}" for i in range(64) for axis in ("x","y")]
    scols=[f"cranio_shape_{i:03d}_{axis}" for i in range(64) for axis in ("x","y")]
    out=out.merge(face[["ano","id_pessoa",*fcols]],on=["ano","id_pessoa"],how="left",validate="many_to_one")
    out=out.merge(skull[["ano","id_pessoa",*scols]],on=["ano","id_pessoa"],how="left",validate="many_to_one")
    if out[fcols+scols].isna().any().any(): raise ValueError("Há pares sem shape64 facial ou craniano")
    return out


def train_two_tower(splits: pd.DataFrame, config: dict, run_dir: Path | None = None, evaluation_split: str = "test") -> tuple[Path, dict]:
    _, DataLoader, _, _ = _stack()
    seed = int(config.get("seed", 42)); _seed(seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_dir or PROJECT_DIR / "runs" / run_id; run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    log_path = run_dir / "train.log"
    def log(message: str) -> None:
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    def progress(payload: dict) -> None:
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(progress_path)
    tc = config.get("training", {}); mc = config.get("model", {}); dc = config.get("data", {})
    if bool(dc.get("use_domain_normalized",False)):
        splits=apply_domain_normalized_paths(splits,int(dc.get("image_size",224)))
    explicit_exclusions={str(key) for key in dc.get("exclude_pair_keys",[])}
    if explicit_exclusions:
        excluded=splits.pair_key.astype(str).isin(explicit_exclusions)
        splits.loc[excluded,["pair_key","split"]].to_csv(run_dir/"excluded_by_protocol.csv",index=False)
        log(f"Excluidos {int(excluded.sum())} par(es) definidos pelo protocolo experimental")
        splits=splits.loc[~excluded].copy()
    face_views=tuple(dc.get("face_views",["FFN"]))
    if bool(dc.get("require_complete_face_views",False)) and len(face_views)>1:
        required=[{"FFN":"face_path","FFS":"face_ffs_path","FL":"face_fl_path"}[view] for view in face_views]
        complete=splits[required].notna().all(axis=1)
        excluded=splits.loc[~complete,[column for column in ("pair_key","split",*required) if column in splits]].copy()
        if len(excluded):
            excluded.to_csv(run_dir/"excluded_incomplete_face_views.csv",index=False)
            log(f"Excluidos {len(excluded)} par(es) sem todas as vistas faciais; splits originais preservados")
        splits=splits.loc[complete].copy()
    use_shape64=bool(dc.get("use_shape64",False))
    if use_shape64: splits=apply_shape64_features(splits)
    PairDataset = make_dataset_class(int(dc.get("image_size", 224)), tuple(dc.get("skull_views", ["CF"])), str(dc.get("augmentation", "none")),use_shape64,str(dc.get("domain_harmonization","none")),face_views)
    batch = int(tc.get("batch_size", 32)); device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    names = ["train", "val"] + ([evaluation_split] if evaluation_split not in ("train", "val", "none", "") else [])
    frames={name:splits[splits.split.eq(name)].reset_index(drop=True) for name in dict.fromkeys(names)}
    datasets={name:PairDataset(frames[name],training=name=="train") for name in frames}
    loaders = {name: DataLoader(datasets[name], batch_size=batch, shuffle=name == "train", num_workers=0) for name in datasets}
    mining_loader=DataLoader(PairDataset(frames["train"],training=False),batch_size=batch,shuffle=False,num_workers=0)
    model = TwoTower(int(mc.get("embedding_dim", 64)), bool(mc.get("pretrained", True)), bool(mc.get("freeze_backbone", True)), float(mc.get("dropout", 0.15)), str(mc.get("view_fusion","mean")), int(mc.get("max_skull_views",3)),use_shape64,int(mc.get("geometry_hidden_dim",128)),str(mc.get("face_view_fusion","mean")),int(mc.get("max_face_views",3))).to(device)
    def optimizer_for_stage():
        projection, backbone = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad: (backbone if ".backbone." in name else projection).append(parameter)
        groups=[]
        if projection: groups.append({"params":projection,"lr":float(tc.get("learning_rate",1e-3))})
        if backbone: groups.append({"params":backbone,"lr":float(tc.get("backbone_learning_rate",1e-4))})
        return torch.optim.AdamW(groups, weight_decay=1e-4)
    optimizer = optimizer_for_stage()
    save_provenance(run_dir, config)
    selection_metric=str(tc.get("checkpoint_metric","mrr")); best_score=-1.0; patience=0; history=[]; checkpoint=run_dir / "best.pt"
    max_epochs = int(tc.get("epochs", 20)); started = time.monotonic(); epoch_durations=[]
    evaluation_count = len(loaders[evaluation_split].dataset) if evaluation_split in loaders else 0
    log(f"Execucao {run_id} iniciada em {device}; {len(loaders['train'].dataset)} treino, {len(loaders['val'].dataset)} validacao, {evaluation_count} avaliacao ({evaluation_split})")
    log(f"Configuracao: {len(face_views)} vista(s) facial(is), {len(dc.get('skull_views', ['CF']))} vista(s) craniana(s), batch {batch}, maximo {max_epochs} epocas")
    log(f"Selecao de checkpoint: {selection_metric}")
    hard_enabled = str(tc.get("loss", "infonce")) == "infonce_hard_negatives"
    hard_start = int(tc.get("hard_negative_start_epoch", int(tc.get("warmup_epochs", 3)) + 1))
    global_mining=bool(tc.get("global_hard_negative_mining",False)); mining_refresh=max(1,int(tc.get("global_hard_refresh_epochs",2))); mined_neighbors={}
    if hard_enabled:
        log(f"Negativos dificeis: top-{int(tc.get('hard_negative_top_k',3))}, peso {float(tc.get('hard_negative_weight',0.5)):.2f}, margem {float(tc.get('margin',0.2)):.2f}, inicio epoca {hard_start}")
    if global_mining: log(f"Mineracao global ativada: top-{int(tc.get('global_hard_top_k',5))}, atualizacao a cada {mining_refresh} epoca(s)")
    progress({"status":"iniciando","run_id":run_id,"device":str(device),"epoch":0,"max_epochs":max_epochs})
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.monotonic()
        if epoch == int(tc.get("warmup_epochs", 3)) + 1 and bool(tc.get("unfreeze_last_block", True)):
            model.unfreeze_last_blocks(); optimizer = optimizer_for_stage(); log("Ultimo bloco das duas ResNet liberado para fine-tuning")
        stage = "finetune_layer4" if epoch > int(tc.get("warmup_epochs", 3)) else "projection_only"
        if global_mining and epoch>=hard_start and (not mined_neighbors or (epoch-hard_start)%mining_refresh==0):
            log("Atualizando negativos dificeis em todo o conjunto de treino...")
            mined_faces,mined_skulls,mined_ids=_encode(model,mining_loader,device)
            mined_neighbors=mine_global_hard_neighbors(mined_faces,mined_skulls,mined_ids,int(tc.get("global_hard_top_k",5)))
            batches=hard_negative_batches(mined_ids,mined_neighbors,batch,seed+epoch)
            loaders["train"]=DataLoader(datasets["train"],batch_sampler=batches,num_workers=0)
            log(f"Mineracao global concluida: {len(mined_ids)} pares organizados em {len(batches)} lotes")
        log(f"Epoca {epoch}/{max_epochs} iniciada — {stage}")
        model.train(); losses=[]; hard_losses=[]
        for train_batch in loaders["train"]:
            optimizer.zero_grad()
            if len(train_batch)==5:
                face,skull,face_shape,skull_shape,_=train_batch; f,s=model(face.to(device),skull.to(device),face_shape.to(device),skull_shape.to(device))
            else:
                face,skull,_=train_batch; f,s=model(face.to(device),skull.to(device))
            if hard_enabled and epoch >= hard_start:
                loss, hard_loss = symmetric_hard_negative_loss(f, s, float(tc.get("temperature", 0.07)), float(tc.get("label_smoothing", 0.0)), float(tc.get("margin", 0.2)), int(tc.get("hard_negative_top_k", 3)), float(tc.get("hard_negative_weight", 0.5)))
                hard_losses.append(float(hard_loss.item()))
            else:
                loss = symmetric_infonce(f, s, float(tc.get("temperature", 0.07)), float(tc.get("label_smoothing", 0.0)))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.item()))
        vf, vs, ids = _encode(model, loaders["val"], device); val_similarity=vf@vs.T; metrics, _ = retrieval_metrics(val_similarity, ids, ids)
        if selection_metric.startswith("gallery10_"): metrics.update(sampled_gallery_metrics(val_similarity,int(tc.get("selection_gallery_size",10)),int(tc.get("selection_gallery_repeats",50)),int(tc.get("selection_gallery_seed",2026))))
        if selection_metric not in metrics: raise ValueError(f"Métrica de checkpoint desconhecida: {selection_metric}")
        metrics.update(epoch=epoch, train_loss=float(np.mean(losses)))
        metrics["hard_negative_loss"] = float(np.mean(hard_losses)) if hard_losses else 0.0
        metrics["stage"] = stage; history.append(metrics)
        duration = time.monotonic() - epoch_started; epoch_durations.append(duration)
        current_score=float(metrics[selection_metric]); improved=current_score>best_score
        if improved:
            best_score=current_score; patience=0; torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch,"selection_metric":selection_metric,"selection_score":current_score}, checkpoint)
        else: patience += 1
        # Grava a cada epoca: permanece visivel mesmo se houver interrupcao.
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        eta = float(np.mean(epoch_durations[-3:]) * max(0, max_epochs - epoch))
        snapshot={"status":"treinando","run_id":run_id,"device":str(device),"epoch":epoch,"max_epochs":max_epochs,"stage":stage,"train_loss":metrics["train_loss"],"hard_negative_loss":metrics["hard_negative_loss"],"val_mrr":metrics["mrr"],"val_top_1":metrics["top_1"],"val_top_5":metrics["top_5"],"val_recall_at_10":metrics["recall_at_10"],"selection_metric":selection_metric,"selection_score":current_score,"best_selection_score":best_score,"best_epoch":int(max(history,key=lambda x:x[selection_metric])["epoch"]),"patience":patience,"epoch_seconds":duration,"eta_seconds":eta,"checkpoint_updated":improved}
        progress(snapshot)
        marker = " — MELHOR MODELO SALVO" if improved else f" — sem melhora ({patience}/{int(tc.get('patience',5))})"
        hard_text = f" | hard={metrics['hard_negative_loss']:.4f}" if hard_losses else ""
        log(f"Epoca {epoch}/{max_epochs}: loss={metrics['train_loss']:.4f}{hard_text} | MRR={metrics['mrr']:.4f} | Top-1={metrics['top_1']*100:.2f}% | Top-5={metrics['top_5']*100:.2f}% | Top-10={metrics['recall_at_10']*100:.2f}% | {duration:.1f}s{marker}")
        if patience >= int(tc.get("patience", 5)):
            log(f"Early stopping acionado apos {patience} epocas sem melhora"); break
    target_split = evaluation_split if evaluation_split in loaders else "val"
    log(f"Carregando melhor checkpoint e avaliando o conjunto '{target_split}'...")
    state=torch.load(checkpoint, map_location=device); model.load_state_dict(state["model_state"])
    tf, ts, ids = _encode(model, loaders[target_split], device); final_metrics, ranking = retrieval_metrics(tf @ ts.T, ids, ids)
    ranking.to_csv(run_dir / f"ranking_{target_split}.csv", index=False)
    # Cortes descritivos; nunca são usados para selecionar checkpoint.
    test_frame = splits[splits.split.eq(target_split)].set_index("pair_key")
    correct = ranking[ranking["correto"]].set_index("query_id")
    stratified = {}
    for field in ("ano", "sexo_face"):
        for value, keys in test_frame.groupby(field).groups.items():
            ranks = correct.loc[correct.index.intersection(keys), "rank"].to_numpy()
            if len(ranks): stratified[f"{field}={value}"] = {"n":int(len(ranks)),"top_1":float(np.mean(ranks<=1)),"top_5":float(np.mean(ranks<=5)),"recall_at_10":float(np.mean(ranks<=10)),"mrr":float(np.mean(1/ranks))}
    result={"run_id":run_id,"device":str(device),"best_epoch":state["epoch"],"evaluation_split":target_split,target_split:final_metrics,f"stratified_{target_split}":stratified}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress({"status":"concluido","run_id":run_id,"device":str(device),"epoch":len(history),"max_epochs":max_epochs,"best_epoch":state["epoch"],"elapsed_seconds":time.monotonic()-started,"evaluation_split":target_split,target_split:final_metrics})
    log(f"Concluido. Melhor epoca: {state['epoch']} | {target_split} MRR={final_metrics['mrr']:.4f} | Top-1={final_metrics['top_1']*100:.2f}% | Top-5={final_metrics['top_5']*100:.2f}% | Top-10={final_metrics['recall_at_10']*100:.2f}%")
    return checkpoint, result
