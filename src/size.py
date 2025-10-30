# Isaac Joffe, 2025
# Defines the rules by which grid size can be transformed


# General grid size setup
# -----------------------------------------------------------------------------
# System hyperparameter
N_GRIDSIZE_HYPOTHESES = 3       # Identity, constant, function

# Universal identifiers of different object hypotheses
IDENTITY_SIZE_ID = 0
CONSTANT_SIZE_ID = 1
FUNCTION_SIZE_ID = 2

# Priorities of different object hypotheses
IDENTITY_SIZE_PRIORITY = 0
CONSTANT_SIZE_PRIORITY = 1
FUNCTION_SIZE_PRIORITY = 2

# Mapping from object hypothesis identifiers to printable names
GRIDSIZE_HYPOTHESIS_MAP = {
    IDENTITY_SIZE_ID: "Identity",
    CONSTANT_SIZE_ID: "Constant",
    FUNCTION_SIZE_ID: "Function",
}
# -----------------------------------------------------------------------------


# Standard way of representing a grid size hypothesis
# -----------------------------------------------------------------------------
# Abstract class representing a general grid size hypothesis
class ARCGridSizeHypothesis():
    # Construct the grid size hypothesis object
    def __init__(self, identifier, priority):
        # Store general information about what grid size hypothesis this is
        self.__identifier = identifier
        self.__priority = priority
        return

    # Convert the object hypothesis into a printable format
    # Final
    def __str__(self):
        return f"{GRIDSIZE_HYPOTHESIS_MAP[self.get_identifier()]} Grid Size Hypothesis"

    # Return the identifier of this grid size hypothesis
    # Final
    def get_identifier(self):
        # Identifier should not be changed later
        return self.__identifier

    # Return the execution priority of this grid size hypothesis
    # Final
    def get_priority(self):
        # Priority should not be changed later
        return self.__priority

    # Execute the grid size hypothesis
    # Abstract
    def apply(self):
        raise NotImplementedError("Grid size hypothesis must implement an apply() method")
# -----------------------------------------------------------------------------


# Basic grid size hypotheses
# -----------------------------------------------------------------------------
# Identity (same) grid size hypothesis
class IdentitySizeHypothesis(ARCGridSizeHypothesis):
    # No parameters needed, just passes through input grid size
    def __init__(self):
        super().__init__(
            IDENTITY_SIZE_ID,
            IDENTITY_SIZE_PRIORITY,
        )
        return

    # Predicts grid size stays the same as the input grid
    def apply(self, in_grid_size):
        return in_grid_size


# Constant (same) grid size hypothesis
class ConstantSizeHypothesis(ARCGridSizeHypothesis):
    # Parameter is the constant grid size to always predict
    def __init__(self, constant_size):
        super().__init__(
            CONSTANT_SIZE_ID,
            CONSTANT_SIZE_PRIORITY,
        )
        self.constant_size = constant_size
        return

    # Predicts grid size is soem constant value
    def apply(self, none=None):
        return self.constant_size
# -----------------------------------------------------------------------------


# More involved grid size hypothesis
# -----------------------------------------------------------------------------
# Function (operation, transform) grid size hypothesis
class FunctionSizeHypothesis(ARCGridSizeHypothesis):
    # Parameters are the sizes of all grids in the demonstrations
    def __init__(self, train_in_sizes, train_out_sizes):
        super().__init__(
            FUNCTION_SIZE_ID,
            FUNCTION_SIZE_PRIORITY,
        )
        # TODO: Compute some learned function (for now, just identity)
        self.size_predictor = (lambda x: x)
        return

    # Predicts grid size according to some function
    def apply(self, in_grid_size):
        return self.size_predictor(in_grid_size)
# -----------------------------------------------------------------------------


def main():
    return


if __name__ == "__main__":
    main()
