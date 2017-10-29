# sklearn-hogwild
An implementation of the Hogwild! algorithm for asynchronous SGD that interfaces with sci-kit learn. 

## Requirements

A requirements.txt file is included and dependencies can be installed via:

```python
pip install -r requirements.txt
```

## Usage

```python
from hogwildsgd import HogWildRegressor
hwsgd = HogWildRegressor(n_epochs = 5,
                         batch_size = 1, 
                         chunk_size = 32,
                         learning_rate = .001,
                         verbose=2)
hwsgd.fit(X,y)
```

## Example

For an example, see the test.py file.