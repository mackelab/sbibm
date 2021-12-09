import os
from pathlib import Path


from julia import Julia
from warnings import warn

JULIA_PROJECT = str(Path(__file__).parent / "julia")
os.environ["JULIA_PROJECT"] = JULIA_PROJECT


def find_sysimage():
    if "JULIA_SYSIMAGE_DIFFMODELS" in os.environ:
        environ_path = Path(os.environ["JULIA_SYSIMAGE_DIFFMODELS"])
        if environ_path.exists():
            return str(environ_path)
        else:
            warn("JULIA_SYSIMAGE_DIFFMODELS is set but image does not exist")
            return None
    else:
        warn("JULIA_SYSIMAGE_DIFFMODELS not set")
        default_path = Path("~/.julia_sysimage_diffmodels.so").expanduser()
        if default_path.exists():
            warn(f"Defaulting to {default_path}")
            return str(default_path)
        else:
            return None


class DDMJulia:
    def __init__(
        self,
        dt: float = 0.001,
        num_trials: int = 1,
        dim_parameters: int = 4,
        seed: int = -1,
    ) -> None:
        """Wrapping DDM simulation and likelihood computation from Julia.

        Based on Julia package DiffModels.jl

        https://github.com/DrugowitschLab/DiffModels.jl

        Calculates likelihoods via Navarro and Fuss 2009.

        Args:
            dt: integration step size
            num_trials: number of iid trials to simulator per parameter.
            seed: seed passed to Julia rng.
        """

        self.dt = dt
        self.num_trials = num_trials
        self.seed = seed

        self.jl = Julia(
            compiled_modules=False,
            sysimage=find_sysimage(),
            runtime="julia",
        )
        self.jl.eval("using DiffModels")
        self.jl.eval("using Random")

        # forward model and likelihood for two-param case, symmetric bounds.
        if dim_parameters == 2:
            self.simulate = self.jl.eval(
                f"""
                    function simulate(vs, as; dt={self.dt}, num_trials={self.num_trials}, seed={self.seed})
                        num_parameters = size(vs)[1]
                        rt = fill(NaN, (num_parameters, num_trials))
                        c = fill(NaN, (num_parameters, num_trials))

                        # seeding
                        if seed > 0
                            Random.seed!(seed)
                        end
                        for i=1:num_parameters
                            drift = ConstDrift(vs[i], dt)
                            # Pass 0.5a to get bound from boundary separation.
                            bound = ConstSymBounds(0.5 * as[i], dt)
                            s = sampler(drift, bound)
                        
                            for j=1:num_trials
                                rt[i, j], cj = rand(s)
                                c[i, j] = cj ? 1.0 : 0.0
                            end
                        end
                        return rt, c
                    end
                """
            )
            self.log_likelihood = self.jl.eval(
                f"""
                    function log_likelihood(vs, as, rts, cs; dt={self.dt}, l_lower_bound=1e-29)
                        batch_size = size(vs)[1]
                        num_trials = size(rts)[1]

                        logl = zeros(batch_size)

                        for i=1:batch_size
                            drift = ConstDrift(vs[i], dt)
                            # Pass 0.5a to get bound from boundary separation.
                            bound = ConstSymBounds(0.5 * as[i], dt)

                            for j=1:num_trials
                                if cs[j] == 1.0
                                    logl[i] += log(max(l_lower_bound, pdfu(drift, bound, rts[j])))
                                else
                                    logl[i] += log(max(l_lower_bound, pdfl(drift, bound, rts[j])))
                                end
                            end
                        end
                        return logl
                    end
                """
            )
            # forward model and likelihood for four-param case via asymmetric bounds
            # as in LAN paper, "simpleDDM".
        else:
            self.simulate_simpleDDM = self.jl.eval(
                f"""
                    function simulate_simpleDDM(v, bl, bu; dt={self.dt}, num_trials={self.num_trials}, seed={self.seed})
                        num_parameters = size(v)[1]
                        rt = fill(NaN, (num_parameters, num_trials))
                        c = fill(NaN, (num_parameters, num_trials))

                        # seeding
                        if seed > 0
                            Random.seed!(seed)
                        end

                        for i=1:num_parameters
                            drift = ConstDrift(v[i], dt)
                            bound = ConstAsymBounds(bu[i], bl[i], dt)
                            s = sampler(drift, bound)

                            for j=1:num_trials
                                # Simulate DDM.
                                rt[i, j], cj = rand(s)
                                c[i, j] = cj ? 1.0 : 0.0
                            end

                        end
                        return rt, c
                    end
                """
            )
            self.log_likelihood_simpleDDM = self.jl.eval(
                f"""
                    function log_likelihood_simpleDDM(v, bl, bu, rts, cs; ndt=0, dt={self.dt}, l_lower_bound=1e-29)
                        # eps is the numerical lower bound for the likelihood used in HDDM.
                        parameter_batch_size = size(v)[1]
                        num_trials = size(rts)[1]
                        # If no ndt is passed, use zeros without effect.
                        if ndt == 0
                            ndt = zeros(parameter_batch_size)
                        end

                        logl = zeros(parameter_batch_size)

                        for i=1:parameter_batch_size
                            drift = ConstDrift(v[i], dt)
                            bound = ConstAsymBounds(bu[i], bl[i], dt)

                            for j=1:num_trials
                                # Subtract the current ndt from rt to get correct likelihood.
                                rt = rts[j] - ndt[i]
                                # If rt negative (too high ndt) likelihood is 0.
                                if rt < 0
                                    # 1e-29 is the lower bound for negative rts used in HDDM.
                                    logl[i] += log(l_lower_bound)
                                else
                                    if cs[j] == 1.0
                                        logl[i] += log(max(l_lower_bound, pdfu(drift, bound, rt)))
                                    else
                                        logl[i] += log(max(l_lower_bound, pdfl(drift, bound, rt)))
                                    end
                                end
                            end
                        end
                        return logl
                    end
                """
            )
