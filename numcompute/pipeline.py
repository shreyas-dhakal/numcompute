from typing import Any, Iterable, Sequence


class Pipeline:
    """
    Chain transformers and an optional final estimator.
    Parameters
    steps : Sequence[tuple[str, Any]]

    Raises
    ValueError
        If steps is empty, names are invalid/duplicate, or required methods
        are not implemented.

    Time Complexity
    Initialization: O(n), where n is the number of steps.

    Space Complexity
    O(n), for storing the steps.
    """

    def __init__(self, steps: Sequence[tuple[str, Any]]) -> None:
        if not steps:
            raise ValueError("steps must contain at least one (name, object) pair.")

        self.steps = list(steps)
        self._validate_steps()

    def _validate_steps(self) -> None:
        """
        Validate pipeline step structure and interface requirements.

        Raises
        ValueError
            If any step violates naming or interface constraints.

        Time Complexity
        O(n), where n is the number of steps.

        Space Complexity
        O(n), for tracking seen step names.
        """
        seen = set()
        for name, step in self.steps:
            if not isinstance(name, str) or not name:
                raise ValueError("Each step name must be a non-empty string.")
            if name in seen:
                raise ValueError(f"Duplicate step name: {name}.")
            seen.add(name)

            has_fit = hasattr(step, "fit")
            has_transform = hasattr(step, "transform")
            has_predict = hasattr(step, "predict")
            if not has_fit:
                raise ValueError(f"Step '{name}' must implement fit().")

            is_last = (name, step) == self.steps[-1]
            if not is_last and not has_transform:
                raise ValueError(f"Non-final step '{name}' must implement transform().")
            if is_last and not (has_transform or has_predict):
                raise ValueError(
                    f"Final step '{name}' must implement transform() or predict()."
                )

    def fit(self, X: Any, y: Any = None) -> "Pipeline":
        """
        Fit all steps in the pipeline.

        Parameters
        X : Any
            Input data.
        y : Any, optional
            Target values.

        Returns
        Pipeline
            The fitted pipeline instance.

        Raises
        AttributeError
            If a step is missing required methods during execution.

        Time Complexity
        O(n * T_fit + n * T_transform), where n is number of steps.

        Space Complexity
        O(m), where m is the size of intermediate transformed data.
        """
        x_curr = X
        for i, (_, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1
            if is_last:
                if y is None:
                    step.fit(x_curr)
                else:
                    step.fit(x_curr, y)
            else:
                if hasattr(step, "fit_transform"):
                    x_curr = step.fit_transform(x_curr)
                else:
                    step.fit(x_curr)
                    x_curr = step.transform(x_curr)
        return self

    def transform(self, X: Any) -> Any:
        """
        Apply transformations sequentially to the input data.

        Parameters
        X : Any
            Input data.

        Returns
        Any
            Transformed data after applying all steps.

        Raises
        ValueError
            If the final step does not implement `transform()`.

        Time Complexity
        O(n * T_transform), where n is the number of steps.

        Space Complexity
        O(m), where m is the size of intermediate transformed data.
        """
        x_curr = X
        for i, (name, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1
            if is_last and not hasattr(step, "transform"):
                raise ValueError(
                    f"Final step '{name}' does not implement transform(); use predict()."
                )
            x_curr = step.transform(x_curr)
        return x_curr

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """Fit the pipeline and transform the data.

        Parameters
        X : Any
            Input data.
        y : Any, optional
            Target values.

        Returns
        Any
            Transformed data after fitting the pipeline.

        Time Complexity
        O(n * (T_fit + T_transform)), where n is the number of steps.

        Space Complexity
        O(m), where m is the size of intermediate transformed data.
        """
        self.fit(X, y=y)
        return self.transform(X)

    def predict(self, X: Any) -> Any:
        """Generate predictions using the final step.

        Parameters
        X : Any
            Input data.

        Returns
        Any
            Predicted output.

        Raises
        ValueError
            If the final step does not implement `predict()`.

        Time Complexity
        O((n-1) * T_transform + T_predict), where n is the number of steps.

        Space Complexity
        O(m), where m is the size of intermediate transformed data.
        """
        x_curr = X
        for _, step in self.steps[:-1]:
            x_curr = step.transform(x_curr)

        name, final_step = self.steps[-1]
        if not hasattr(final_step, "predict"):
            raise ValueError(f"Final step '{name}' does not implement predict().")
        return final_step.predict(x_curr)
    

__all__ = ["Pipeline"]
