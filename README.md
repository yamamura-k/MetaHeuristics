# MetaHeuristics

Python and C++ implementation of several metaheuristic algorithms.

## Component

```bash
├── README.md
├── requrements.txt
├── ABC.py
├── paraABC.py
├── BA.py
├── paraBA.py
├── grad_based
│   ├── gradient_descent.py
│   ├── newton.py
│   ├── nesterov.py
│   └── conjugate.py
├── cpp
│   ├── CMakeLists.txt
│   ├── ABC.cpp
│   └── main.cpp
├── benchmarks.py
├── base.py
├── utils.py
├── test_metaheuristics.py
├── test_grad_based.py
├── compare_result_metaheuristics.py
└──compare_result_grad_based.py
```

### Main algorithms

#### Metaheuristics
| File        | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| ABC.py      | Implementation of sequential Artificial Bee Colony algorithm |
| paraABC.py  | Implementation of parallel Artificial Bee Colony algorithm   |
| BA.py       | Implementation of sequential Bat algorithm                   |
| paraBA.py   | Implementation of parallel Bat algorithm                     |
| cpp/ABC.cpp | Implementation of sequential Artificial Bee Colony algorithm |

#### gradient based algorithms
| File        | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| gradient_descent.py      | Implementation of gradient descent algorithm |
| newton.py  | Implementation of newton algorithm   |
| nesterov.py       | Implementation of nesterov algorithm                   |
| conjugate.py   | Implementation of conjugate gradient algorithm                     |


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
| File               | Description                         |
| ------------------ | ----------------------------------- |
| test_metaheuristics.py||
| test_grad_based.py||
| cpp/main.cpp       | for test                            |

### Others

| File               | Description                         |
| ------------------ | ----------------------------------- |
| utils.py           | Utility functions for parallization |
|compare_result_metaheuristics.py| Example |
|compare_result_grad_based.py|Example|
| cpp/CMakeLists.txt | For cmake                           |


## References

- [A New Metaheuristic Bat-Inspired Algorithm](https://arxiv.org/pdf/1004.4170.pdf)
- [A powerful and efficient algorithm for numerical
function optimization: artificial bee colony (ABC)
algorithm](https://link.springer.com/content/pdf/10.1007/s10898-007-9149-x.pdf)
