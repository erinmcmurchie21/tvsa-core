import scipy.optimize as opt

def objectivefunction(x):
    # Two parameter Rosenbrock function
    f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    return f

# Find global optimum using differential evolution
result = opt.differential_evolution(
    objectivefunction,
    bounds = [(-100, 100), (-100, 100)],
    seed = 1337,
    maxiter = 10000,
    popsize = 24,
    tol = 1e-14,
    atol = 1e-14,
    disp = True,
    init = 'latinhypercube',
    updating = 'immediate',
    workers = 1,
    )

print(result)
print(result.x)