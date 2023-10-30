import numpy as np


class Uvar:
    def __init__(self, n_terms, R, cjs, t):
        self.n_terms = n_terms
        self.R = R
        self.theta = np.zeros((R, n_terms))
        self.cjs = cjs
        self.t = t
        self.test = np.zeros((R, R))
        for r in range(R):
            for s in range(R):
                self.test[r, s] = -1 if s > r else 1

    def update_theta(self, new_array):
        if new_array.shape != (self.R - 1, self.n_terms):
            raise ValueError("Wrong length provided to update theta.")
        for j in range(self.n_terms):
            for r in range(self.R - 1):
                self.theta[r, j] = new_array[r, j]
            self.theta[self.R - 1, j] = self.t * self.cjs[j]
            for r in range(self.R - 1):
                self.theta[self.R - 1, j] -= new_array[r, j]

    def get_initial_trotter_vector(self):
        theta_trotter = np.zeros((self.R - 1, self.n_terms))
        for j in range(self.n_terms):
            for r in range(self.R - 1):
                theta_trotter[r, j] = self.cjs[j] * self.t / self.R
        return theta_trotter

    def flatten_theta(self, theta):
        return np.array(theta).reshape((self.R - 1) * self.n_terms)

    def set_theta_to_Trotter(self):
        theta_trotter = self.get_initial_trotter_vector()
        self.update_theta(theta_trotter)

    def chi(self, j, m):
        cc1 = self.theta[:, j].transpose() @ self.test @ self.theta[:, m]
        return cc1
