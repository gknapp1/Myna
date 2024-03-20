# Contributing

Contributing to Myna is easy: just open a [pull
request](https://help.github.com/articles/using-pull-requests/). Make
`main` the destination branch on the Myna repository and add a full
description of the intended changes.

Your pull request must pass Myna's tests, be formatted to match Myna, and be
reviewed by at least one Myna developer.

Additional dependencies are needed for formatting and testing:
```
pip install -e .[dev]
```

## Automatic commit checks

You can use `pre-commit` to ensure local changes are ready to be submitted in a
pull request. It can be installed locally using: `pip install pre-commit`, then it
must be setup for the repository: `pre-commit install`. Formatting checks are then
run for every commit.

`PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit ...` can be used to selectively skip
these checks.

Checks can instead be done manually, described below.

## Code formatting

Myna is formatted with `black`. You can run `black .` from the top level directory.

## Unit testing

Myna uses `pytest`. You can run `python -m pytest` from the top level directory.