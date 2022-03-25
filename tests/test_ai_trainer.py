from unittest.mock import MagicMock

import pytest


@pytest.fixture
def model_file():
    from pathlib import Path

    return Path(__file__).parent.parent / "strainmap_ai" / "ai_config.toml"


class TestDataAugmentation:
    def test_factory(self, model_file):
        import toml

        from strainmap_ai.unet import DataAugmentation

        config = toml.load(model_file)["augmentation"]

        da = DataAugmentation.factory()
        assert len(da.steps) == len([c for c in config["active"]])
        assert da.times == config["times"]
        assert da.axis == config["axis"]
        assert da.include_original == config["include_original"]

    def test_transform(self):
        import numpy as np

        from strainmap_ai.unet import DataAugmentation

        def double(d):
            return 2 * d

        n = 3
        da = DataAugmentation.factory()
        da.steps = [
            double,
        ] * n
        data = np.random.random((5, 5))
        actual = da.transform(data)
        assert actual == pytest.approx(data * 2 ** n)

    def test_augment(self):
        import numpy as np

        from strainmap_ai.unet import DataAugmentation

        def double(d):
            return 2 * d

        n = 10
        c = 4
        h = w = 5
        images = np.random.random((n, h, w, c))
        labels = np.random.random((n, h, w))
        da = DataAugmentation.factory()
        da.steps = [
            double,
        ]

        da.include_original = False
        aug_img, aug_lbl = da.augment(images, labels)
        assert aug_img.shape == (n * da.times, h, w, c)
        assert aug_lbl.shape == (n * da.times, h, w)

        da.include_original = True
        aug_img, aug_lbl = da.augment(images, labels)
        assert aug_img.shape == (n * (da.times + 1), h, w, c)
        assert aug_lbl.shape == (n * (da.times + 1), h, w)

    def test__group(self):
        import numpy as np

        from strainmap_ai.unet import DataAugmentation

        n = 3
        c = 2
        h = w = 5
        images = np.random.random((n, h, w, c))
        labels = np.random.random((n, h, w))
        grouped = DataAugmentation._group(images, labels)
        assert grouped.shape == (n * (c + 1), h, w)
        for i in range(n):
            assert (grouped[i : c * n : n].transpose((1, 2, 0)) == images[i]).all()
        assert (grouped[-n:] == labels).all()

    def test__ungroup(self):
        import numpy as np

        from strainmap_ai.unet import DataAugmentation

        n = 3
        c = 2
        h = w = 5
        images = np.random.random((n, h, w, c))
        labels = np.random.random((n, h, w))
        grouped = DataAugmentation._group(images, labels)
        eimages, elabels = DataAugmentation._ungroup(grouped, c)
        assert (eimages == images).all()
        assert (elabels == labels).all()


class TestNormal:
    def test_avail(self):
        from strainmap_ai.unet import Normal

        assert len(Normal.avail()) >= 1

    def test_register(self):
        from strainmap_ai.unet import Normal

        @Normal.register
        def my_norm():
            pass

        assert "my_norm" in Normal.avail()
        del Normal._normalizers["my_norm"]

    def test_run(self):
        from strainmap_ai.unet import Normal

        fun = MagicMock(__name__="my_norm")
        Normal.register(fun)
        data = [1, 2, 3]
        Normal.run(data, "my_norm")
        fun.assert_called_with(data)
        del Normal._normalizers["my_norm"]


class TestUNet:
    def test_factory(self, model_file, tmpdir):
        import toml

        from strainmap_ai.unet import UNet

        config = toml.load(model_file)
        config["Net"]["model_name"] = "my test model"
        new_model = tmpdir / "new_model.toml"
        toml.dump(config, new_model)

        net = UNet.factory(new_model)
        assert net.model_name == "my test model"

    def test_model(self):
        from strainmap_ai.unet import UNet

        net = UNet()
        with pytest.raises(RuntimeError):
            net.model

        net._model = "a model"
        assert net.model == net._model

    def test_conv_block(self, mocker):
        import numpy as np
        from keras import layers

        from strainmap_ai.unet import UNet

        sp_activation = mocker.spy(layers, "Activation")

        net = UNet()

        tensor = np.random.rand(10, 10, 10, 10)
        repetitions = 5
        actual = net._conv_block(tensor, net.filters, repetitions=repetitions)
        assert actual.shape[0] == tensor.shape[0]
        assert actual.shape[-1] == net.filters
        assert sp_activation.call_count == repetitions

    def test_deconv_block(self, mocker):
        import numpy as np
        from keras import layers

        from strainmap_ai.unet import UNet

        sp_conv_transpose = mocker.spy(layers, "Conv2DTranspose")
        sp_concatenate = mocker.spy(layers, "concatenate")

        net = UNet()
        net._conv_block = MagicMock()

        tensor = np.random.rand(10, 10, 10, 10)
        residual = np.random.rand(10, 20, 20, net.filters)
        net._deconv_block(tensor, residual, net.filters)
        sp_conv_transpose.assert_called()
        sp_concatenate.assert_called()
        net._conv_block.assert_called()

    def test_modelstruct(self):
        """Nothing to test here as the exact sequence might change.

        Just making sure that the return value is a Model object.
        """
        from keras.models import Model

        from strainmap_ai.unet import UNet

        model = UNet()._modelstruct()
        assert isinstance(model, Model)

    def test_compile_model(self):
        from strainmap_ai.unet import UNet

        net = UNet()

        class Mockdel:
            compile = MagicMock()
            summary = MagicMock()
            load_weights = MagicMock()

        model = Mockdel()
        net._modelstruct = MagicMock(return_value=model)

        net.compile_model()
        model.compile.assert_called()
        model.summary.assert_called()
        model.load_weights.assert_not_called()

        net.model_file = "a model path"
        net.compile_model()
        model.load_weights.assert_called_with("a model path")

        net.compile_model(model_file="another path")
        model.load_weights.assert_called_with("another path")

    def test_train(self):
        import numpy as np

        from strainmap_ai.unet import UNet

        net = UNet()
        net._model = MagicMock()
        net._model.fit = MagicMock()
        net._model.save_weights = MagicMock()

        net.train(np.array([]), np.array([]), model_file="a path")
        net._model.fit.assert_called()
        net._model.save_weights.assert_called_with("a path")

    def test_infer(self):
        import numpy as np

        from strainmap_ai.unet import UNet

        result = np.random.rand(10, 10, 10)
        expected = (result[..., 0] > 0.5).astype(float)

        net = UNet()
        net._model = MagicMock()
        net._model.predict = MagicMock(return_value=result)

        actual = net.infer(np.array([]))
        net._model.predict.assert_called()
        np.testing.assert_equal(actual, expected)
