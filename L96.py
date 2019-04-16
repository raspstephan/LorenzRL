"""
Definition of the Lorenz96 model.

Created on 2019-04-16-12-28
Author: Stephan Rasp, raspstephan@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm_notebook as tqdm


class L96OneLevel(object):
    def __init__(self, K=36, J=10, h=1, F=10, c=10, b=10, dt=0.01,
                 X_init=None, noprog=False):
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt
        self.params = [self.F]
        self.noprog = noprog
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros((self.K, self.J))
        self._history_X = [self.X.copy()]

    def _rhs(self, X):
        """Compute the right hand side of the ODE."""
        dXdt = (
                -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) -
                X + self.F
        )
        return dXdt

    def step(self):
        """Step forward one time step with RK4."""
        k1 = self.dt * self._rhs(self.X)
        k2 = self.dt * self._rhs(self.X + k1 / 2)
        k3 = self.dt * self._rhs(self.X + k2 / 2)
        k4 = self.dt * self._rhs(self.X + k3)
        self.X += 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self._history_X.append(self.X.copy())

    def iterate(self, steps):
        for n in tqdm(range(steps), disable=self.noprog):
            self.step()

    @property
    def state(self):
        return self.X

    def set_state(self, x):
        self.X = x

    @property
    def history(self):
        return xr.DataArray(self._history_X, dims=['time', 'x'], name='X')


class L96TwoLevelUncoupled(object):
    def __init__(self, K=36, J=10, h=1, F=10, c=10, b=10, dt=0.001,
                 X_init=None, noprog=False):
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt
        self.noprog=noprog
        self.X = np.random.rand(self.K) if X_init is None else X_init
        self.Y = np.zeros(self.K * self.J)
        self._history_X = [self.X.copy()]
        self._history_Y = [self.Y.copy()]
        self._history_B = [-self.h * self.c * self.Y.reshape(self.K, self.J).mean(1)]

    def _rhs_X(self, X, B):
        """Compute the right hand side of the X-ODE."""
        dXdt = (
                -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) -
                X + self.F + B
        )
        return dXdt

    def _rhs_Y(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = (
                       -self.b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) -
                       Y + self.h / self.J * np.repeat(X, self.J)
               ) * self.c
        return dYdt

    def step(self):
        # First get solution for X without updating Y
        B = -self.h * self.c * self.Y.reshape(self.K, self.J).mean(1)
        Y_mean = self.Y.reshape(self.K, self.J).mean(1)
        k1_X = self.dt * self._rhs_X(self.X, B)
        k2_X = self.dt * self._rhs_X(self.X + k1_X / 2, B)
        k3_X = self.dt * self._rhs_X(self.X + k2_X / 2, B)
        k4_X = self.dt * self._rhs_X(self.X + k3_X, B)

        # Then update Y with unupdated X
        k1_Y = self.dt * self._rhs_Y(self.X, self.Y)
        k2_Y = self.dt * self._rhs_Y(self.X, self.Y + k1_Y / 2)
        k3_Y = self.dt * self._rhs_Y(self.X, self.Y + k2_Y / 2)
        k4_Y = self.dt * self._rhs_Y(self.X, self.Y + k3_Y)

        # Then update both
        self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        self.Y += 1 / 6 * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        self._history_X.append(self.X.copy())
        self._history_Y.append(self.Y.copy())
        self._history_B.append(B.copy())

    def iterate(self, steps):
        for n in tqdm(range(steps), disable=self.noprog):
            self.step()

    @property
    def state(self):
        return np.concatenate([self.X, self.Y])

    def set_state(self, x):
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def history(self):
        da_X = xr.DataArray(self._history_X, dims=['time', 'x'], name='X')
        da_B = xr.DataArray(self._history_B, dims=['time', 'x'], name='B')
        da_X_repeat = xr.DataArray(np.repeat(self._history_X, self.J, 1),
                                   dims=['time', 'y'], name='X_repeat')
        da_Y = xr.DataArray(self._history_Y, dims=['time', 'y'], name='Y')
        return xr.Dataset({'X': da_X, 'B': da_B, 'Y': da_Y, 'X_repeat': da_X_repeat})


class L96TwoLevelCoupled(object):
    def __init__(self, K=36, J=10, h=1, F=10, c=10, b=10, dt=0.001,
                 X_init=None):
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt
        self.X = np.random.rand(self.K) if X_init is None else X_init
        self.Y = np.zeros(self.K * self.J)
        self._history_X = self.X[None, :]
        self._history_Y = self.Y[None, :]

    def _rhs_X(self, X, Y):
        """Compute the right hand side of the X-ODE."""
        dXdt = (
                -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) -
                X + self.F - self.h * self.c * Y.reshape(self.K, self.J).mean(1)
        )
        return dXdt

    def _rhs_Y(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = (
                       -self.b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) -
                       Y + self.h / self.J * np.repeat(X, self.J)
               ) / self.c
        return dYdt

    def _rhs_dt(self, X, Y):
        return self.dt * self._rhs_X(X, Y), self.dt * self._rhs_Y(X, Y)

    def step(self):
        k1_X, k1_Y = self._rhs_dt(self.X, self.Y)
        k2_X, k2_Y = self._rhs_dt(self.X + k1_X / 2, self.Y + k1_Y / 2)
        k3_X, k3_Y = self._rhs_dt(self.X + k2_X / 2, self.Y + k2_Y / 2)
        k4_X, k4_Y = self._rhs_dt(self.X + k3_X, self.Y + k3_Y)
        self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        self.Y += 1 / 6 * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        self._history_X = np.concatenate([self._history_X, self.X[None, :]])
        self._history_Y = np.concatenate([self._history_Y, self.Y[None, :]])

    def iterate(self, steps):
        for n in tqdm(range(steps)):
            self.step()

    @property
    def history(self):
        da_X = xr.DataArray(self._history_X, dims=['time', 'x'], name='X')
        da_X_repeat = xr.DataArray(np.repeat(self._history_X, self.J, 1),
                                   dims=['time', 'y'], name='X_repeat')
        da_Y = xr.DataArray(self._history_Y, dims=['time', 'y'], name='Y')
        return xr.Dataset({'X': da_X, 'Y': da_Y, 'X_repeat': da_X_repeat})





class L96TwoLevelNN(object):
    def __init__(self, K=36, J=10, h=1, F=10, c=10, b=10, dt=0.001,
                 X_init=None, model=None, mean_x=None, mean_y=None, std_x=None, std_y=None):
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt
        self.model = model
        self.mean_x, self.mean_y, self.std_x, self.std_y = mean_x, mean_y, std_x, std_y
        self.X = np.random.rand(self.K) if X_init is None else X_init
        self.Y = np.zeros(self.K * self.J)
        self._history_X = [self.X.copy()]
        self._history_Y = [self.Y.copy()]
        self._history_B = [-self.h * self.c * self.Y.reshape(self.K, self.J).mean(1)]

    def _rhs_X(self, X, B):
        """Compute the right hand side of the X-ODE."""
        dXdt = (
                -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) -
                X + self.F + B
        )
        return dXdt

    def _rhs_Y(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = (
                       -self.b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) -
                       Y + self.h / self.J * np.repeat(X, self.J)
               ) * self.c
        return dYdt

    def step(self):
        # First get solution for X without updating Y
        #         B = -self.h * self.c * self.Y.reshape(self.K, self.J).mean(1)
        B = model.predict_on_batch((self.X - self.mean_x) / self.std_x).squeeze()
        B = B * self.std_y + self.mean_y
        k1_X = self.dt * self._rhs_X(self.X, B)
        k2_X = self.dt * self._rhs_X(self.X + k1_X / 2, B)
        k3_X = self.dt * self._rhs_X(self.X + k2_X / 2, B)
        k4_X = self.dt * self._rhs_X(self.X + k3_X, B)

        #         # Then update Y with unupdated X
        #         k1_Y = self.dt * self._rhs_Y(self.X, self.Y)
        #         k2_Y = self.dt * self._rhs_Y(self.X, self.Y + k1_Y/2)
        #         k3_Y = self.dt * self._rhs_Y(self.X, self.Y + k2_Y/2)
        #         k4_Y = self.dt * self._rhs_Y(self.X, self.Y + k3_Y)

        # Then update both
        self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        #         self.Y += 1/6 * (k1_Y + 2*k2_Y + 2*k3_Y + k4_Y)
        self._history_X.append(self.X.copy())
        #         self._history_Y.append(self.Y.copy())
        self._history_B.append(B.copy())

    def iterate(self, steps):
        for n in tqdm(range(steps)):
            self.step()

    @property
    def history(self):
        da_X = xr.DataArray(self._history_X, dims=['time', 'x'], name='X')
        da_B = xr.DataArray(self._history_B, dims=['time', 'x'], name='B')
        da_X_repeat = xr.DataArray(np.repeat(self._history_X, self.J, 1),
                                   dims=['time', 'y'], name='X_repeat')
        #         da_Y = xr.DataArray(self._history_Y, dims=['time', 'y'], name='Y')
        return xr.Dataset({'X': da_X, 'B': da_B, 'X_repeat': da_X_repeat})