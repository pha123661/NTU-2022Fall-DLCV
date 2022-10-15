import torch
import torch.nn as nn


class GRF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.view_as(x)  # NECESSARY! autograd checks if tensor is modified

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.saved_variables[0]
        return grad_output.neg() * lambda_, None


class FeatureExtractor(nn.Module):
    def __init__(self, in_chans=3, n_features=512) -> None:
        super().__init__()
        self.n_features = n_features
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, n_features, 3, padding=1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        feature = self.conv(x)
        feature = feature.reshape(-1, 512)
        return feature


class LabelPredictor(nn.Module):
    def __init__(self, n_features=512, n_classes=10) -> None:
        super().__init__()
        self.l_clf = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.ReLU(),
            nn.Linear(n_features // 2, n_features // 4),
            nn.ReLU(),
            nn.Linear(n_features // 4, n_classes),
        )

    def forward(self, x):
        x = self.l_clf(x)
        # x = nn.functional.softmax(x)  # using BCEloss later
        return x


class DomainClassifier(nn.Module):
    '''
    A Binary classifier
    '''

    def __init__(self, n_features=512) -> None:
        super().__init__()
        self.d_clf = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(n_features // 2, n_features // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(n_features // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, lambda_):
        x = GRF.apply(x, lambda_)
        x = self.d_clf(x)
        return x


if __name__ == '__main__':
    n = FeatureExtractor()
    n(torch.rand(64, 3, 32, 32))
