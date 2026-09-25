import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from ntc import TextureSampler, BC1Texture, FeatureGrid, NeuralTexture, texel_centers
from ntc.bc1 import BLOCK_DTYPE, palette, pack565
from ntc.storage import quantize_uint8, quantize_model, save_model, load_model


class CompressionTests(unittest.TestCase):
    def test_pixel_centers_and_borders_non_square(self):
        a = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 17
        tex = TextureSampler(a)
        uv = texel_centers(2, 3)
        np.testing.assert_allclose(tex.sample(uv[:, 0], uv[:, 1]), a.reshape(-1, 3), atol=1e-7)
        np.testing.assert_allclose(tex.sample(0, 0), a[0, 0])
        np.testing.assert_allclose(tex.sample(1, 1), a[-1, -1])
        np.testing.assert_allclose(tex.sample(1/3, .5), a[:, :2].mean((0, 1)), atol=1e-7)

    def test_grid_lookup_matches_numpy_and_has_gradients(self):
        g = FeatureGrid((3,), 3)
        image = np.random.default_rng(2).random((3, 3, 3), dtype=np.float32)
        with torch.no_grad():
            g.grids[0].copy_(torch.from_numpy(image).permute(2, 0, 1)[None])
        uv = torch.tensor([[0, 0], [1, 1], [.2, .7], [.5, .5]])
        out = g(uv)
        np.testing.assert_allclose(out.detach(), TextureSampler(image).sample(uv[:, 0], uv[:, 1]), atol=1e-6)
        out.sum().backward()
        self.assertGreater(g.grids[0].grad.abs().sum().item(), 0)

    def test_known_bc1_bit_layout(self):
        block = np.zeros(1, dtype=BLOCK_DTYPE)
        block['c0'], block['c1'] = 0xf800, 0x001f
        block['indices'] = 0xe4e4e4e4  # left-to-right indices 0,1,2,3 in every row
        tex = BC1Texture(4, 4, block)
        expected = np.tile(palette(np.uint16(0xf800), np.uint16(0x001f))[None], (4, 1, 1))
        np.testing.assert_allclose(tex.reconstruct(), expected)
        self.assertEqual(block.tobytes()[:4], bytes([0, 248, 31, 0]))

    def test_bc1_block_boundaries_padding_and_roundtrip(self):
        im = np.random.default_rng(7).random((7, 9, 3), dtype=np.float32)
        tex = BC1Texture.compress(im)
        self.assertEqual(tex.stored_bytes, 2 * 3 * 8)
        self.assertTrue((tex.blocks['c0'] > tex.blocks['c1']).all())
        uv = np.array([[4/9, 4/7], [.01, .99], [.72, .46]])
        np.testing.assert_allclose(tex.sample(uv[:, 0], uv[:, 1]), TextureSampler(tex.reconstruct()).sample(uv[:, 0], uv[:, 1]), atol=1e-7)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'test.bc1'
            tex.save(p)
            np.testing.assert_array_equal(tex.reconstruct(), BC1Texture.load(p).reconstruct())

    def test_bc1_flat_extremes(self):
        for value in (0, 1, .5):
            tex = BC1Texture.compress(np.full((4, 4, 3), value, dtype=np.float32))
            self.assertLess(np.abs(tex.reconstruct() - value).max(), .02)
            self.assertGreater(int(tex.blocks['c0'][0]), int(tex.blocks['c1'][0]))

    def test_model_parameter_counts_and_training_gradients(self):
        expected = {'small': 12739, 'medium': 15555, 'large': 92483}
        for name, n in expected.items():
            model = NeuralTexture(name)
            self.assertEqual(model.stored_bytes, n * 4)
            y = model(torch.rand(20, 2))
            self.assertEqual(y.shape, (20, 3))
            ((y - .2) ** 2).mean().backward()
            self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_quantization_constant_and_error_bound(self):
        for a in (torch.ones(17) * 4, torch.linspace(-2, 3, 400)):
            q, lo, scale, restored = quantize_uint8(a)
            self.assertEqual(q.dtype, np.uint8)
            self.assertLessEqual(float((restored - a).abs().max()), float(scale / 2) + 1e-6)
            self.assertTrue(torch.isfinite(restored).all())

    def test_real_uint8_file_and_model_roundtrip(self):
        model = NeuralTexture('medium')
        for quantize_mlp in (False, True):
            quantized, arrays = quantize_model(model, quantize_mlp)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'model.ntc'
                sizes = save_model(quantized, p, 512, 512, arrays)
                decoded, _ = load_model(p)
                self.assertEqual(sizes['file_bytes'], p.stat().st_size)
                expected = sum(v[0].size + 8 for v in arrays.values()) + sum(p.numel()*4 for n,p in model.named_parameters() if n not in arrays)
                self.assertEqual(sizes['stored_bytes'], expected)
                for a, b in zip(quantized.parameters(), decoded.parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                uv = torch.rand(15, 2)
                torch.testing.assert_close(quantized(uv), decoded(uv), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
