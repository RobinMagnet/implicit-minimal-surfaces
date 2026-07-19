import numpy as np


class GLLogger:
    def __init__(
        self,
        Lap,
        M,
        epsilon,
        lam,
        alpha,
        log_interval=10,
        log_values=False,
        log_values_interval=None,
        save_values_dir=None,
    ):
        self.Lap = Lap
        self.M = M
        self.epsilon = epsilon
        self.lam = lam
        self.alpha = alpha
        self.log_interval = log_interval
        self.log_values = log_values

        self.log_values_interval = (
            log_values_interval if log_values_interval is not None else log_interval
        )
        self.save_values_dir = save_values_dir

        # if log_values:
        self.history_values = []

        self.history_total = []
        self.history_dirichlet = []
        self.history_potential = []

    def __call__(self, xk):
        # print("logging", xk)
        # 1. Compute raw energies using your existing functions
        # xk is phi_r2 (real, imag stacked)
        # print(type(xk), xk.shape, xk.dtype)

        iteration = len(self.history_total) + 1
        first_it = iteration == 1

        if iteration % self.log_interval == 0 or first_it:
            E_dir = dirichlet_energy(xk, self.Lap)
            E_pot = potential_energy(xk, self.M)

            # 2. Compute weighted total energy (the objective function)
            # Formula: alpha * Dir + (lam / eps^2) * Pot
            weight_pot = self.lam / (self.epsilon**2)
            total = self.alpha * E_dir + weight_pot * E_pot

            # 3. Store
            self.history_total.append(total)
            self.history_dirichlet.append(E_dir)
            self.history_potential.append(E_pot)

            # Optional: Print progress

            print(
                f"Iter {iteration}: Total={total:.2e} | "
                f"Dir={E_dir:.2e} | Pot={E_pot:.2e} | Mean norm={np.abs(cast_to_complex(xk)).mean():.2e}"
            )
        else:
            self.history_total.append(None)
            self.history_dirichlet.append(None)
            self.history_potential.append(None)

        if self.log_values and (
            (iteration % self.log_values_interval == 0) or first_it
        ):
            if self.save_values_dir is None:
                self.history_values.append(cast_to_complex(xk).copy())
            else:
                np.save(
                    f"{self.save_values_dir}/phi_iter_{iteration}.npy",
                    cast_to_complex(xk),
                )


def cast_to_complex(phi_r2):
    N = phi_r2.shape[0] // 2
    return phi_r2[:N] + 1j * phi_r2[N:]


def cast_to_r2(phi):
    return np.concatenate((np.real(phi), np.imag(phi)))


def dirichlet_energy(phi_r2, Lap):
    phi = cast_to_complex(phi_r2)
    return 0.5 * np.real(phi.conj().T @ (Lap @ phi))


def dirichlet_energy_grad(phi_r2, Lap):
    phi = cast_to_complex(phi_r2)
    grad = Lap @ phi
    grad_r2 = cast_to_r2(grad)
    return grad_r2


def dirichlet_energy_and_grad(phi_r2, Lap):
    phi = cast_to_complex(phi_r2)  # (N)

    grad = Lap @ phi  # (N,)
    energy = 0.5 * np.real(phi.conj().T @ grad)
    grad_r2 = cast_to_r2(grad)
    return energy, grad_r2


def potential_energy(phi_r2, M):
    phi = cast_to_complex(phi_r2)
    displacement = np.abs(phi) ** 2 - 1.0

    return 0.25 * (displacement.T @ (M @ displacement))


def potential_energy_grad(phi_r2, M):

    phi = cast_to_complex(phi_r2)

    # displacement = 1 - np.abs(phi) ** 2
    # grad = -M @ (displacement * phi)

    displacement = np.abs(phi) ** 2 - 1.0
    grad = phi * (M @ displacement)
    grad_r2 = cast_to_r2(grad)
    return grad_r2


def potential_energy_and_grad(phi_r2, M):
    phi = cast_to_complex(phi_r2)

    displacement = np.abs(phi) ** 2 - 1.0

    energy = 0.25 * (displacement.T @ (M @ displacement))

    # displacement_grad = displacement * phi
    # grad = M @ displacement_grad

    grad = phi * (M @ displacement)
    grad_r2 = cast_to_r2(grad)

    return energy, grad_r2


def potential_energy_and_grad_lmks(phi_r2, M, weights):
    phi = cast_to_complex(phi_r2)  # (N,)

    displacement = np.abs(phi) ** 2 - 1.0  # (N,)

    # print(displacement.shape, weights.shape)

    sqrtw = np.sqrt(weights)

    weighted_displacement = sqrtw * displacement

    energy = 0.25 * (weighted_displacement.T @ (M @ weighted_displacement))

    # displacement_grad = displacement * phi
    # grad = M @ displacement_grad

    grad = phi * (sqrtw * (M @ (weighted_displacement)))
    grad_r2 = cast_to_r2(grad)

    return energy, grad_r2


def potential_energy_and_grad_lmks2(phi_r2, M, weights):
    phi = cast_to_complex(phi_r2)  # (N,)

    displacement = np.abs(phi) ** 2 - weights  # (N,)

    # print(displacement.shape, weights.shape)

    if isinstance(M, np.ndarray) and M.ndim == 1:
        energy = 0.25 * np.dot(M, displacement**2)
        grad = M * displacement * phi
    else:
        integrated_displacement = M @ displacement
        energy = 0.25 * (displacement.T @ integrated_displacement)

        grad = phi * integrated_displacement

    grad_r2 = cast_to_r2(grad)
    return energy, grad_r2


def gl_energy_and_grad(phi_r2, Lap, M, epsilon, lam, alpha):
    dir_energy, dir_grad = dirichlet_energy_and_grad(phi_r2, Lap)
    pot_energy, pot_grad = potential_energy_and_grad(phi_r2, M)

    energy = alpha * dir_energy + (lam / epsilon**2) * pot_energy
    grad = alpha * dir_grad + (lam / epsilon**2) * pot_grad

    return energy, grad


def gl_energy_and_grad_lmks(phi_r2, Lap, M, epsilon, lam, alpha, weights):
    dir_energy, dir_grad = dirichlet_energy_and_grad(phi_r2, Lap)
    pot_energy, pot_grad = potential_energy_and_grad_lmks(phi_r2, M, weights)

    energy = alpha * dir_energy + (lam / epsilon**2) * pot_energy
    grad = alpha * dir_grad + (lam / epsilon**2) * pot_grad

    return energy, grad


def gl_energy_and_grad_lmks2(phi_r2, Lap, M, epsilon, lam, alpha, weights):
    dir_energy, dir_grad = dirichlet_energy_and_grad(phi_r2, Lap)
    pot_energy, pot_grad = potential_energy_and_grad_lmks2(phi_r2, M, weights)

    energy = alpha * dir_energy + (lam / epsilon**2) * pot_energy
    grad = alpha * dir_grad + (lam / epsilon**2) * pot_grad

    return energy, grad


def gl_energy(phi_r2, Lap, M, epsilon, lam, alpha):
    return alpha * dirichlet_energy(phi_r2, Lap) + (
        lam / epsilon**2
    ) * potential_energy(phi_r2, M)


def gl_energy_grad(phi_r2, Lap, M, epsilon, lam, alpha):
    return alpha * dirichlet_energy_grad(phi_r2, Lap) + (
        lam / epsilon**2
    ) * potential_energy_grad(phi_r2, M)
