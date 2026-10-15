import unittest
import torch
from scripts.piloto_localizador_ia import masked_loss, model


class PilotTest(unittest.TestCase):
    def test_missing_labels_have_no_gradient(self):
        pred=torch.tensor([[[1.,1.],[100.,100.]]],requires_grad=True)
        target=torch.zeros_like(pred)
        loss=masked_loss(pred,target,torch.tensor([[True,False]]))
        self.assertEqual(loss.item(),2.)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad[0,1],torch.zeros(2)))

    def test_shape_and_bounds(self):
        result=model()(torch.zeros(2,3,128,128))
        self.assertEqual(tuple(result.shape),(2,22))
        self.assertTrue(bool(((result>=0)&(result<=1)).all()))
