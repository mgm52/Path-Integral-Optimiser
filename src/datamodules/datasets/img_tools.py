# pylint: skip-file
import numpy as np
import torch


def prepare_image(
    rgb, crop=None, embed=None, white_cutoff=225, gauss_sigma=5, background=0.01
):
    """Transforms rgb image array into 2D-density and energy
    Parameters
    ----------
    density : ndarray(width, height)
        Probability density
    energy : ndarray(width, height)
        Energy
    """
    img = rgb

    # make one channel
    img = img.mean(axis=2)

    # make background white
    img = img.astype(np.float32)
    img[img > white_cutoff] = 255

    # normalize
    img /= img.max()

    if crop is not None:
        # crop
        img = img[crop[0] : crop[1], crop[2] : crop[3]]

    if embed is not None:
        tmp = np.ones((embed[0], embed[1]), dtype=np.float32)
        shift_x = (embed[0] - img.shape[0]) // 2
        shift_y = (embed[1] - img.shape[1]) // 2
        tmp[shift_x : img.shape[0] + shift_x, shift_y : img.shape[1] + shift_y] = img
        img = tmp

    # convolve with Gaussian
    from scipy.ndimage import gaussian_filter

    img2 = gaussian_filter(img, sigma=gauss_sigma)

    # add background
    background1 = gaussian_filter(img, sigma=10)
    background2 = gaussian_filter(img, sigma=20)
    background3 = gaussian_filter(img, sigma=50)
    density = (1.0 - img2) + background * (background1 + background2 + background3)

    U = -np.log(density)
    U -= U.min()

    return density, U


class Energy(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.0
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self.energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x)[0]


class ImageEnergy(Energy):
    def __init__(
        self, pixel_energy, mean=[350, 350], scale=[350, 350], outside_penalty=1.0
    ):
        """Evaluates image energy at continuous coordinates
        Parameters
        ----------
        pixel_energy : ndarray(width, height)
            Image energy
        mean : (int, int)
            center pixel
        scale : (int, int)
            number of pixels to scale to 1.0 (in x and y direction)
        outside_penalty : float
            prefactor for x^2 penalty when being x away from image boundary.
        """
        super().__init__(2)
        self.pixel_energy = torch.tensor(pixel_energy)
        self.mean = torch.tensor([mean])
        self.scale = torch.tensor([scale])
        self.maxindex_x = torch.tensor([pixel_energy.shape[1] - 1])
        self.maxindex_y = torch.tensor([pixel_energy.shape[0] - 1])
        self.outside_penalty = outside_penalty

    def energy(self, X, **kwargs):
        Xs = X * self.scale.to(dtype=torch.float32) + self.mean.to(dtype=torch.float32)
        I = Xs.to(dtype=torch.long)
        Ix = I[:, 0]
        Iy = I[:, 1]
        pixel_split_dx = Xs[:, 0] - Ix.to(dtype=torch.float32)
        pixel_split_dy = Xs[:, 1] - Iy.to(dtype=torch.float32)

        zero = torch.tensor([0.0])
        # select closest pixel inside image
        # Ix_inside = torch.max(torch.min(Ix, self.maxindex_x), zero)
        # Iy_inside = torch.max(torch.min(Iy, self.maxindex_y), zero)
        Ix_inside = torch.max(torch.min(Ix, self.maxindex_x - 1), zero + 1).to(
            dtype=torch.long
        )
        Iy_inside = torch.max(torch.min(Iy, self.maxindex_y - 1), zero + 1).to(
            dtype=torch.long
        )
        E0 = self.pixel_energy[Iy_inside, Ix_inside]
        dEdx = 0.5 * (
            self.pixel_energy[Iy_inside, Ix_inside + 1]
            - self.pixel_energy[Iy_inside, Ix_inside - 1]
        )
        dEdy = 0.5 * (
            self.pixel_energy[Iy_inside + 1, Ix_inside]
            - self.pixel_energy[Iy_inside - 1, Ix_inside]
        )

        # image_energy = E0 + pixel_split_dx * dEdx + pixel_split_dy * dEdy
        # image_energy = image_energy.unsqueeze(-1)

        # penalty factor from being outside image
        dx_left = torch.max(-Xs[:, 0], zero)
        dx_right = torch.max(Xs[:, 0] - self.maxindex_x, zero)
        dx = torch.max(dx_left, dx_right) / self.scale[0][0]
        dy_down = torch.max(-Xs[:, 1], zero)
        dy_up = torch.max(Xs[:, 1] - self.maxindex_y, zero)
        dy = torch.max(dy_down, dy_up) / self.scale[0][1]
        penalty = self.outside_penalty * (dx ** 2 + dy ** 2).to(dtype=torch.float32)
        # penalty = penalty.unsqueeze(-1)

        img_e = (
            E0
            + (pixel_split_dx * dEdx + pixel_split_dy * dEdy) * (penalty < 1e-6)
            + penalty
        )
        return img_e.unsqueeze(-1)
        # return image_energy + penalty

    def density(self, nbins):
        xmax = 0.5 * self.pixel_energy.shape[1] / self.scale[0, 1].numpy()
        ymax = 0.5 * self.pixel_energy.shape[0] / self.scale[0, 0].numpy()
        xrange = (-xmax, xmax)
        yrange = (-ymax, ymax)
        x_pos = np.linspace(-xmax, xmax, num=nbins, endpoint=True)
        y_pos = np.linspace(-ymax, ymax, num=nbins, endpoint=True)

        probe = []
        for x in x_pos:
            for y in y_pos:
                probe.append([x, y])
        Eprobe = self.energy(torch.tensor(probe)).numpy()
        hist_X = np.exp(-Eprobe).reshape(nbins, nbins)
        hist_X /= hist_X.sum()

        return hist_X, xrange, yrange
