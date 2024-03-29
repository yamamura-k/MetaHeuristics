# MetaHeuristics

Python and C++ implementation of several metaheuristic algorithms.

## Example
```python
from algorithm import optimize
result = optimize(dimension, function, maximum_iteration, **options)
best_solution = (result.best_obj, result.best_x)
```

## Component

```bash
.
├── LICENSE
├── README.md
├── requirements.txt
├── tests
│   ├── test_all.py
│   ├── utils -> ../utils/
│   ├── algorithm -> ../algorithm
│   └── benchmarks.py -> ../benchmarks.py
├── utils
│   ├── __init__.py
│   ├── logging.py
│   ├── grad_based.py
│   ├── parameter_search.py
│   ├── common.py
│   ├── parallel.py
│   └── base.py
├── algorithm
│   ├── __init__.py
│   ├── grad_based
│   │   ├── __init__.py
│   │   ├── gradient_descent.py
│   │   ├── newton.py
│   │   ├── nesterov.py
│   │   └── conjugate.py
│   ├── metaheuristics
│   │   ├── __init__.py
│   │   ├── TLBO.py
│   │   ├── GWO.py
│   │   ├── paraBA.py
│   │   ├── BA.py
│   │   ├── paraABC.py
│   │   ├── FA.py
│   │   └── ABC.py
│   └── nelder_mead.py
├── cpp
│   ├── CMakeLists.txt
│   ├── ABC.cpp
│   └── main.cpp
├── check_grad.py
├── compare_result.py
└── benchmarks.py
```

### Main algorithms

#### Metaheuristics
| File           | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| ABC.py         | Implementation of sequential Artificial Bee Colony algorithm                |
| paraABC.py     | Implementation of parallel Artificial Bee Colony algorithm                  |
| BA.py          | Implementation of sequential Bat algorithm                                  |
| paraBA.py      | Implementation of parallel Bat algorithm                                    |
| GWO.py         | Implementation of sequential Gray Wolf minimizer algorithm(under develop)   |
| FA.py          | Implementation of firefly algorithm                                         |
| nelder_mead.py | Implementation of nelder-mead algorithm                                     |
| cpp/ABC.cpp    | Implementation of sequential Artificial Bee Colony algorithm                |

#### gradient based algorithms
| File                     | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| gradient_descent.py      | Implementation of gradient descent algorithm                 |
| newton.py                | Implementation of newton algorithm                           |
| nesterov.py              | Implementation of nesterov algorithm                         |
| conjugate.py             | Implementation of conjugate gradient algorithm               |


### Benchmark functions

These functions are implemented in `benchmarks.py`. (Base class is implemented in `base.py`)
Reference : https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda (Japanese)

| Name                            | Description                                       |
| ------------------------------- | ------------------------------------------------- |
| Ackley function                 | Multimodal function.                              |
| Sphere function                 | Basic function.                                   |
| Rosenbrock function             | Popular benchmark function. (Banana function)     |
| Styblinski-Tang function        | Multimodal function.                              |
| k-tablet function               | Unimodal function.                                |
| Weighted Sphere function        | Unimodal function.                                |
| Sum of different power function | Unimodal function. Absolute value is used.        |
| Griewank function               | Multimodal function. This has so many local opts. |

### for test
`tests/`
| File                  | Description                                       |
| ----------------------| ------------------------------------------------- |
| test_all.py           | test metaheuristics and gradient based algorithms |
| cpp/main.cpp          | test cpp                                          |

### Others

| File               | Description       |
| ------------------ | ----------------- |
| utils              | Utility functions |
| compare_result.py  | Example           |
| cpp/CMakeLists.txt | For cmake         |


## References

- [A New Metaheuristic Bat-Inspired Algorithm](https://arxiv.org/pdf/1004.4170.pdf)
- [A powerful and efficient algorithm for numerical
function optimization: artificial bee colony (ABC)
algorithm](https://link.springer.com/content/pdf/10.1007/s10898-007-9149-x.pdf)
- [Fireﬂy Algorithms for Multimodal Optimization](https://www.researchgate.net/publication/45904853_Firefly_Algorithms_for_Multimodal_Optimization)
