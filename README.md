# MetaHeuristics

Python and C++ implementation of several metaheuristic algorithms.

## Component

├── README.md
├── ABC.py
├── paraABC.py
├── BA.py
├── paraBA.py
├── grad_based/
├── base.py
├── benchmarks.py
├── test.py 
├── utils.py
└──cpp
     ├── ABC.cpp
     ├── CMakeLists.txt
     └── main.cpp

### Main algorithms

| File        | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| ABC.py      | Implementation of sequential Artificial Bee Colony algorithm |
| paraABC.py  | Implementation of parallel Artificial Bee Colony algorithm   |
| BA.py       | Implementation of sequential Bat Algorithm                   |
| paraBA.py   | Implementation of parallel Bat Algorithm                     |
| cpp/ABC.cpp | Implementation of sequential Artificial Bee Colony algorithm |
| grad_based/ | gradient based algorithms (Not implemented)                  |

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

### Others

| File               | Description                         |
| ------------------ | ----------------------------------- |
| utils.py           | Utility functions for parallization |
| test.py            | for test                            |
| cpp/main.cpp       | for test                            |
| cpp/CMakeLists.txt | For cmake                           |

## References

Update Later