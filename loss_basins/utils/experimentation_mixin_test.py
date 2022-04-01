from loss_basins.utils.experimentation_mixin import *


class TestModel(nn.Module, ExperimentationMixin):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 4, 5), nn.ReLU(), nn.MaxPool2d(2))
        self.flatten = nn.Flatten(-3)
        self.linear_layers = nn.Linear(36, 10)

        self.init_mixin()

    def forward(self, x):
        y1 = self.conv_layers(x)
        y2 = self.flatten(y1)
        y3 = self.linear_layers(y2)
        return y3


def test_params_to_vector():
    model = TestModel()
    params = [p.clone() for p in model.parameters()]
    vector = model.params_to_vector()

    vec_rand = t.rand(vector.shape)
    model.params_from_vector(vec_rand)
    vector_2 = model.params_to_vector()
    assert not vector.allclose(vector_2)
    assert vector_2.equal(vec_rand)

    model.params_from_vector(vector)
    assert all([p1.allclose(p2) for (p1, p2) in zip(params, model.parameters())])


if __name__ == "__main__":
    test_params_to_vector()
