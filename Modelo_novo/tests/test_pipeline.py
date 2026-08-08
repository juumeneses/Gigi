import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Modelo_novo.src.index_builder import face_person_id
from Modelo_novo.src.retrieval import retrieval_metrics, sampled_gallery_metrics
from Modelo_novo.src.splits import create_leave_one_year_out, create_splits
from Modelo_novo.src.gallery_simulator import _metrics
from Modelo_novo.src.domain_preprocessing import _letterbox, normalize_skull_image, normalize_uploaded_face
from Modelo_novo.src.training import apply_domain_normalized_paths, apply_shape64_features
from Modelo_novo.src.model import TwoTower, symmetric_hard_negative_loss, torch
from Modelo_novo.src.training import hard_negative_batches, mine_global_hard_neighbors
from Modelo_novo.src.studies import average_similarity_matrices, summarize_multiseed
from Modelo_novo.cli import config as load_config
from PIL import Image


class PipelineTests(unittest.TestCase):
    def test_face_person_id(self):
        self.assertEqual(face_person_id("2023_10_ABC_FFN"), "2023_10_ABC")

    def test_retrieval_perfect(self):
        metrics, ranking = retrieval_metrics(np.eye(3), ["a","b","c"], ["a","b","c"])
        self.assertEqual(metrics["top_1"], 1.0)
        self.assertEqual(len(ranking), 9)

    def test_gallery_metrics(self):
        metrics=_metrics(pd.Series([1,2,5,10]),10)
        self.assertEqual(metrics["top_1"],0.25)
        self.assertEqual(metrics["top_5"],0.75)

    def test_sampled_gallery_metrics_perfect(self):
        metrics=sampled_gallery_metrics(np.eye(12),gallery_size=10,repeats=3,seed=42)
        self.assertEqual(metrics["gallery10_top_1"],1.0)
        self.assertEqual(metrics["gallery10_n"],36)

    @unittest.skipIf(torch is None, "PyTorch não instalado")
    def test_hard_negative_loss_penalizes_confusing_pairs(self):
        faces=torch.tensor([[1.0,0.0],[0.0,1.0]])
        easy=torch.tensor([[1.0,0.0],[0.0,1.0]])
        confusing=torch.tensor([[0.8,0.6],[0.6,0.8]])
        _,easy_hard=symmetric_hard_negative_loss(faces,easy,margin=0.3,top_k=1)
        _,confusing_hard=symmetric_hard_negative_loss(faces,confusing,margin=0.3,top_k=1)
        self.assertGreater(float(confusing_hard),float(easy_hard))

    def test_global_hard_negative_mining(self):
        faces=np.eye(3,dtype=float)
        skulls=np.array([[1,0,0],[.9,.1,0],[0,0,1]],dtype=float)
        mined=mine_global_hard_neighbors(faces,skulls,["a","b","c"],top_k=1)
        self.assertIn("b",mined["a"])
        batches=hard_negative_batches(["a","b","c"],mined,batch_size=2,seed=42)
        flattened=[item for batch in batches for item in batch]
        self.assertEqual(sorted(flattened),[0,1,2])

    @unittest.skipIf(torch is None,"PyTorch não instalado")
    def test_attention_fusion_shape(self):
        model=TwoTower(embedding_dim=8,pretrained=False,view_fusion="attention")
        encoded=torch.randn(6,8)
        fused=model.fuse_skull_views(encoded,batch=2,views=3)
        self.assertEqual(tuple(fused.shape),(2,8))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(fused,dim=1),torch.ones(2),atol=1e-5))

    @unittest.skipIf(torch is None, "PyTorch não instalado")
    def test_face_attention_fusion_shape(self):
        model=TwoTower(embedding_dim=8,pretrained=False,face_view_fusion="attention")
        encoded=torch.randn(6,8)
        fused=model.fuse_face_views(encoded,batch=2,views=3)
        self.assertEqual(tuple(fused.shape),(2,8))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(fused,dim=1),torch.ones(2),atol=1e-5))

    @unittest.skipIf(torch is None,"PyTorch não instalado")
    def test_hybrid_model_forward(self):
        model=TwoTower(embedding_dim=8,pretrained=False,use_geometry=True,geometry_hidden_dim=16)
        face=torch.randn(2,3,64,64); skull=torch.randn(2,3,3,64,64)
        f,s=model(face,skull,torch.randn(2,128),torch.randn(2,128))
        self.assertEqual(tuple(f.shape),(2,8)); self.assertEqual(tuple(s.shape),(2,8))

    def test_multiseed_summary(self):
        rows=[]
        for seed,value in ((42,.1),(123,.3)):
            for year in ("2017","2019"):
                rows.append({"holdout_year":year,"seed":seed,"n_queries":10,"top_1":value,"top_3":value,"top_5":value,"recall_at_10":value,"mrr":value,"median_rank":2,"mean_rank":3})
        summary=summarize_multiseed(pd.DataFrame(rows))
        overall=summary[summary.holdout_year.eq("TODOS")].iloc[0]
        self.assertAlmostEqual(overall.top_1_mean,.2)
        self.assertGreater(overall.top_1_std,0)

    def test_ensemble_uses_raw_score_mean(self):
        averaged=average_similarity_matrices([np.array([[1.,0.]]),np.array([[0.,1.]])])
        np.testing.assert_allclose(averaged,[[.5,.5]])

    def test_research_ablation_configs_change_one_component(self):
        attention=load_config("Modelo_novo/configs/research_attention_only.yaml")
        global_only=load_config("Modelo_novo/configs/research_global_hard_only.yaml")
        self.assertEqual(attention["model"]["view_fusion"],"attention")
        self.assertFalse(attention["training"]["global_hard_negative_mining"])
        self.assertEqual(global_only["model"]["view_fusion"],"mean")
        self.assertTrue(global_only["training"]["global_hard_negative_mining"])
        self.assertTrue(attention["data"]["use_domain_normalized"])

    def test_domain_robust_validation_configs_are_comparable(self):
        control=load_config("Modelo_novo/configs/research_validation_control.yaml")
        robust=load_config("Modelo_novo/configs/research_domain_robust.yaml")
        self.assertEqual(control["training"],{k:v for k,v in robust["training"].items()})
        self.assertEqual(robust["data"]["domain_harmonization"],"grayscale_equalized")
        self.assertNotIn("domain_harmonization",control["data"])

    def test_multiview_face_config_preserves_protocol(self):
        base=load_config("Modelo_novo/configs/research_advanced.yaml")
        multi=load_config("Modelo_novo/configs/research_multiview_faces.yaml")
        self.assertEqual(multi["data"]["face_views"],["FFN","FFS","FL"])
        self.assertEqual(multi["model"]["face_view_fusion"],"attention")
        self.assertEqual(multi["split"],base["split"])
        self.assertEqual(multi["training"],base["training"])

    def test_multiview_control_changes_only_face_branch(self):
        control=load_config("Modelo_novo/configs/research_multiview_faces_control.yaml")
        multi=load_config("Modelo_novo/configs/research_multiview_faces.yaml")
        self.assertEqual(control["training"],multi["training"])
        self.assertEqual(control["split"],multi["split"])
        self.assertEqual(control["data"]["skull_views"],multi["data"]["skull_views"])
        self.assertEqual(control["model"]["view_fusion"],multi["model"]["view_fusion"])
        self.assertEqual(control["data"]["face_views"],["FFN"])

    def test_domain_letterbox(self):
        image=_letterbox(Image.new("RGB",(400,100),"white"),224)
        self.assertEqual(image.size,(224,224))

    def test_uploaded_images_are_normalized(self):
        face=normalize_uploaded_face(Image.new("RGB",(400,300),"white"),224)
        skull,status=normalize_skull_image(Image.new("RGB",(300,400),"black"),224)
        self.assertEqual(face.size,(224,224)); self.assertEqual(skull.size,(224,224)); self.assertTrue(status)

    def test_normalized_manifest_covers_splits(self):
        path=Path("Modelo_novo/data/splits/all_splits.csv")
        if path.exists() and Path("Modelo_novo/data/domain_normalized/v1_224/normalized_manifest.csv").exists():
            out=apply_domain_normalized_paths(pd.read_csv(path),224)
            self.assertFalse(out.face_path.isna().any())

    def test_shape64_covers_all_frozen_splits(self):
        path=Path("Modelo_novo/data/splits/all_splits.csv")
        if path.exists():
            out=apply_shape64_features(pd.read_csv(path))
            shape_cols=[c for c in out if c.startswith(("face_shape_","cranio_shape_"))]
            self.assertEqual(len(shape_cols),256)
            self.assertFalse(out[shape_cols].isna().any().any())

    def test_splits_no_leak(self):
        rows=[]
        for year in ("2019","2021"):
            for sex in ("MASCULINO","FEMININO"):
                for i in range(20): rows.append({"pair_key":f"{year}::{sex}::{i}","ano":year,"sexo_face":sex,"usar_treino":True})
        with tempfile.TemporaryDirectory() as tmp:
            out=create_splits(pd.DataFrame(rows),output_dir=Path(tmp))
            self.assertEqual(out.groupby("pair_key").split.nunique().max(),1)
            self.assertEqual(set(out.split),{"train","val","test"})

    def test_loyo_keeps_holdout_exclusive(self):
        rows=[]
        for year in ("2017","2019","2021"):
            for sex in ("MASCULINO","FEMININO"):
                for i in range(10): rows.append({"pair_key":f"{year}::{sex}::{i}","ano":year,"sexo_face":sex,"usar_treino":True})
        with tempfile.TemporaryDirectory() as tmp:
            paths=create_leave_one_year_out(pd.DataFrame(rows),Path(tmp),seed=42)
            fold=pd.read_csv([p for p in paths if p.stem=="holdout_2019"][0])
            self.assertEqual(set(fold[fold.ano.astype(str)=="2019"].split),{"test"})
            self.assertEqual(set(fold[fold.ano.astype(str)!="2019"].split),{"train","val"})


if __name__ == "__main__": unittest.main()
