class Uvar:
    def __init__(self, n_terms, R, cjs, t):
        self.n_terms = n_terms
        self.R = R
        self.theta = np.zeros((R, n_terms))
        self.cjs = cjs
        self.t = t

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

    def set_theta_to_Trotter(self):
        theta_trotter = self.get_initial_trotter_vector()
        self.update_theta(theta_trotter)

    def chi(self, j, m):
        c = 0
        for r in range(self.R):
            c += self.theta[r, j] * self.theta[r, m]
        for q in range(self.R):
            for r in range(q + 1, self.R):
                c += (
                    self.theta[r, j] * self.theta[q, m]
                    - self.theta[r, m] * self.theta[q, j]
                )
        return c
