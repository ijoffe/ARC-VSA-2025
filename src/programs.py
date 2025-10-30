# Isaac Joffe, 2025
# Defines the domain-specific language (DSL) and all its primitives


# User-defined libraries
from grid import *
from object import *
from size import *
from vsa import *
# VSA-related dependencies
import numpy as np
# Utilities
import copy


# Utilities for operation storage and I/O
# -----------------------------------------------------------------------------
# System hyperparameter
N_OPERATIONS = 11            # Generate, Extract, Identity, Recolour, Recentre, Reshape, Gravity, Grow, Fill, Hollow

# Universal identifiers of different programs
GENERATE_OPERATION_ID = 0
EXTRACT_OPERATION_ID = 1
IDENTITY_OPERATION_ID = 2
RECOLOUR_OPERATION_ID = 3
RECENTRE_OPERATION_ID = 4
RESHAPE_OPERATION_ID = 5
MOVE_OPERATION_ID = 6
GRAVITY_OPERATION_ID = 7
GROW_OPERATION_ID = 8
FILL_OPERATION_ID = 9
HOLLOW_OPERATION_ID = 10

# Priorities of different programs
GENERATE_OPERATION_PRIORITY = 1
EXTRACT_OPERATION_PRIORITY = 0
IDENTITY_OPERATION_PRIORITY = 2
RECOLOUR_OPERATION_PRIORITY = 3
RECENTRE_OPERATION_PRIORITY = 3
RESHAPE_OPERATION_PRIORITY = 3
MOVE_OPERATION_PRIORITY = 4
GRAVITY_OPERATION_PRIORITY = 5
GROW_OPERATION_PRIORITY = 5
FILL_OPERATION_PRIORITY = 4
HOLLOW_OPERATION_PRIORITY = 4

# Simplicities of different programs
GENERATE_OPERATION_SIMPLICITY = 5
EXTRACT_OPERATION_SIMPLICITY = 1
IDENTITY_OPERATION_SIMPLICITY = 1
RECOLOUR_OPERATION_SIMPLICITY = 3
RECENTRE_OPERATION_SIMPLICITY = 3
RESHAPE_OPERATION_SIMPLICITY = 3
MOVE_OPERATION_SIMPLICITY = 2
GRAVITY_OPERATION_SIMPLICITY = 2
GROW_OPERATION_SIMPLICITY = 2
FILL_OPERATION_SIMPLICITY = 2
HOLLOW_OPERATION_SIMPLICITY = 2

# Number of parameters to different programs
GENERATE_OPERATION_N_PARAMETERS = 3
EXTRACT_OPERATION_N_PARAMETERS = 0
IDENTITY_OPERATION_N_PARAMETERS = 0
RECOLOUR_OPERATION_N_PARAMETERS = 1
RECENTRE_OPERATION_N_PARAMETERS = 1
RESHAPE_OPERATION_N_PARAMETERS = 1
MOVE_OPERATION_N_PARAMETERS = 1
GRAVITY_OPERATION_N_PARAMETERS = 1
GROW_OPERATION_N_PARAMETERS = 1
FILL_OPERATION_N_PARAMETERS = 0
HOLLOW_OPERATION_N_PARAMETERS = 0

# Mapping from operation identifiers to printable names
OPERATION_MAP = {
    GENERATE_OPERATION_ID: "Generate",
    EXTRACT_OPERATION_ID: "Extract",
    IDENTITY_OPERATION_ID: "Identity",
    RECOLOUR_OPERATION_ID: "Recolour",
    RECENTRE_OPERATION_ID: "Recentre",
    RESHAPE_OPERATION_ID: "Reshape",
    MOVE_OPERATION_ID: "Move",
    GRAVITY_OPERATION_ID: "Gravity",
    GROW_OPERATION_ID: "Grow",
    FILL_OPERATION_ID: "Fill",
    HOLLOW_OPERATION_ID: "Hollow",
}
# -----------------------------------------------------------------------------


# Standard way of representing a DSL operation with associated data
# -----------------------------------------------------------------------------
# Abstract class representing a general action
class ARCOperation():
    # Construct the action object
    def __init__(self, identifier, priority, simplicity, n_parameters, input_object_representation, input_parameters_representation):
        # Store general information about what operation this is
        self.__identifier = identifier
        self.__priority = priority
        self.__simplicity = simplicity
        self.__n_parameters = n_parameters

        # Action execution is defined by the object it operates on and the parameter it takes in
        self.input_object_representation = input_object_representation
        self.input_parameters_representation = input_parameters_representation
        assert (len(input_parameters_representation) == self.get_n_parameters())

        # Store grid-related information needed to execute some actions
        self.__input_grid_size_representation = None
        self.__current_grid = None
        return

    # Convert the operation into a printable format
    # Final
    def __str__(self):
        return f"{OPERATION_MAP[self.get_identifier()]} Operation"

    # Convert the action into a unique integer format
    # Final
    def __hash__(self):
        my_hash = self.get_identifier()
        for parameter in self.input_parameters_representation:
            my_hash += int(1_000_000 * parameter[0]) + int(1_000_000 * parameter[-1])
        return my_hash

    # Return the identifier of this operation
    # Final
    def get_identifier(self):
        # Identifier should not be changed later
        return self.__identifier

    # Return the execution priority of this operation
    # Final
    def get_priority(self):
        # Priority should not be changed later
        return self.__priority

    # Return the expressive simplicity of this operation
    # Final
    def get_simplicity(self):
        # Simplicity should not be changed later
        return self.__simplicity

    # Return the number of parameters this operation requires
    # Final
    def get_n_parameters(self):
        # Number of operations should not be changed later
        return self.__n_parameters

    # Get the dimensions of the input grid
    # Final
    def get_input_grid_size_representation(self):
        return self.__input_grid_size_representation

    # Set the dimensions of the input grid
    # Final
    def set_input_grid_size_representation(self, input_grid_size_representation):
        self.__input_grid_size_representation = input_grid_size_representation
        return

    # Get the partially-generated grid prior to applying this action
    # Final
    def get_current_grid(self):
        return self.__current_grid

    # Get the pixel data partially-generated grid prior to applying this action
    # Final
    def get_current_grid_data(self):
        return self.__current_grid.get_data()

    # Set the partially-generated grid prior to applying this action
    # Final
    def set_current_grid(self, current_grid):
        self.__current_grid = current_grid
        return

    # Execute the action
    # Abstract
    def execute(self):
        raise NotImplementedError("Action must implement an execute() method")
# -----------------------------------------------------------------------------


# Basic actions
# -----------------------------------------------------------------------------
# Generate (create, morph, transform) operation
class GenerateOperation(ARCOperation):
    # Parameters are the COLOUR, CENTRE, and SHAPE of the new object
    def __init__(self, input_object_representation=None, input_parameters_representation=[np.zeros((N_DIMENSIONS)), np.zeros((N_DIMENSIONS)), np.zeros((N_DIMENSIONS))]):
        super().__init__(
            GENERATE_OPERATION_ID,
            GENERATE_OPERATION_PRIORITY,
            GENERATE_OPERATION_SIMPLICITY,
            GENERATE_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Produces a new object in the ouptut set
    def execute(self):
        output_object_representation = ARCObject(
            self.input_parameters_representation[0],
            self.input_parameters_representation[1],
            self.input_parameters_representation[2],
        )
        return output_object_representation


# Extract (keep and crop) operation
class ExtractOperation(ARCOperation):
    # No parameter needed
    def __init__(self, input_object_representation=None, input_parameters_representation=[]):
        super().__init__(
            EXTRACT_OPERATION_ID,
            EXTRACT_OPERATION_PRIORITY,
            EXTRACT_OPERATION_SIMPLICITY,
            EXTRACT_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Reproduce an object from the input set, but centred at the origin with the grid bounds cropped
    def execute(self):
        input_centre = cleanup_centre(self.input_object_representation.get_centre_representation())[1]
        input_shape = self.input_object_representation.get_shape_representation()
        input_position = cleanup_position(SSP_SPACE.bind(input_centre, input_shape), self.get_input_grid_size_representation())[0]

        # Do nothing if input object is nothing
        if (not input_position):
            print("WARNING: Extracting a null object.")
            return self.input_object_representation

        # Determine the bounds of the object
        y_coordinates = [pair[0] for pair in input_position]
        x_coordinates = [pair[1] for pair in input_position]
        height = max(y_coordinates) - min(y_coordinates) + 1
        width = max(x_coordinates) - min(x_coordinates) + 1
        output_grid_size_representation = bundle_size(height, width)

        # Update current grid to new, cropped size
        self.get_current_grid().set_size_representation(output_grid_size_representation)

        output_object_representation = ARCObject(
            self.input_object_representation.get_colour_representation(),
            scale_centre(SSP_SPACE.encode([0, 0]).flatten()),
            self.input_object_representation.get_shape_representation(),
        )
        return output_object_representation


# Identity (select, keep) operation
class IdentityOperation(ARCOperation):
    # No parameter needed
    def __init__(self, input_object_representation=None, input_parameters_representation=[]):
        super().__init__(
            IDENTITY_OPERATION_ID,
            IDENTITY_OPERATION_PRIORITY,
            IDENTITY_OPERATION_SIMPLICITY,
            IDENTITY_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Exactly reproduces an old object from the input set in the ouptut set
    def execute(self):
        output_object_representation = ARCObject(
            self.input_object_representation.get_colour_representation(),
            self.input_object_representation.get_centre_representation(),
            self.input_object_representation.get_shape_representation(),
        )
        return output_object_representation
# -----------------------------------------------------------------------------


# Fundamental actions
# -----------------------------------------------------------------------------
# Recolour operation
class RecolourOperation(ARCOperation):
    # Parameter is the COLOUR of the new object
    def __init__(self, input_object_representation=None, input_parameters_representation=[np.zeros((N_DIMENSIONS))]):
        super().__init__(
            RECOLOUR_OPERATION_ID,
            RECOLOUR_OPERATION_PRIORITY,
            RECOLOUR_OPERATION_SIMPLICITY,
            RECOLOUR_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Reproduce an object from the input set, but with the colour property changed
    def execute(self):
        output_object_representation = ARCObject(
            self.input_parameters_representation[0],
            self.input_object_representation.get_centre_representation(),
            self.input_object_representation.get_shape_representation(),
        )
        return output_object_representation


# Recentre (move to) operation
class RecentreOperation(ARCOperation):
    # Parameter is the CENTRE of the new object
    def __init__(self, input_object_representation=None, input_parameters_representation=[np.zeros((N_DIMENSIONS))]):
        super().__init__(
            RECENTRE_OPERATION_ID,
            RECENTRE_OPERATION_PRIORITY,
            RECENTRE_OPERATION_SIMPLICITY,
            RECENTRE_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Reproduce an object from the input set, but with the centre property changed
    def execute(self):
        output_object_representation = ARCObject(
            self.input_object_representation.get_colour_representation(),
            self.input_parameters_representation[0],
            self.input_object_representation.get_shape_representation(),
        )
        return output_object_representation


# Reshape operation
class ReshapeOperation(ARCOperation):
    # Parameter is the SHAPE of the new object
    def __init__(self, input_object_representation=None, input_parameters_representation=[np.zeros((N_DIMENSIONS))]):
        super().__init__(
            RESHAPE_OPERATION_ID,
            RESHAPE_OPERATION_PRIORITY,
            RESHAPE_OPERATION_SIMPLICITY,
            RESHAPE_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Reproduce an object from the input set, but with the shape property changed
    def execute(self):
        output_object_representation = ARCObject(
            self.input_object_representation.get_colour_representation(),
            self.input_object_representation.get_centre_representation(),
            self.input_parameters_representation[0],
        )
        return output_object_representation
# -----------------------------------------------------------------------------


# Special motion-based actions
# -----------------------------------------------------------------------------
# Move (move static, shift) operation
class MoveOperation(ARCOperation):
    # Parameter is the representation of the difference to move by
    def __init__(self, input_object_representation=None, input_parameters_representation=[np.zeros((N_DIMENSIONS))]):
        super().__init__(
            MOVE_OPERATION_ID,
            MOVE_OPERATION_PRIORITY,
            MOVE_OPERATION_SIMPLICITY,
            MOVE_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Reproduce an object from the input set, but with the centre property shifted
    def execute(self):
        output_centre = SSP_SPACE.bind(self.input_object_representation.get_centre_representation(), self.input_parameters_representation[0]).flatten()

        output_object_representation = ARCObject(
            self.input_object_representation.get_colour_representation(),
            output_centre,
            self.input_object_representation.get_shape_representation(),
        )
        return output_object_representation


# Gravity (move dynamically until contact) operation
class GravityOperation(ARCOperation):
    # Parameter is the representation of the direction to move in
    def __init__(self, input_object_representation=None, input_parameters_representation=[np.zeros((N_DIMENSIONS))]):
        super().__init__(
            GRAVITY_OPERATION_ID,
            GRAVITY_OPERATION_PRIORITY,
            GRAVITY_OPERATION_SIMPLICITY,
            GRAVITY_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Reproduce an object from the input set, but with the object moved until contact with another object or the bounds of the grid
    def execute(self):
        input_centre = cleanup_centre(self.input_object_representation.get_centre_representation())[1]
        input_shape = self.input_object_representation.get_shape_representation()
        input_position = cleanup_position(SSP_SPACE.bind(input_centre, input_shape), self.get_input_grid_size_representation())[0]

        # Convert representation of old position to grid-pixel space
        n_rows, _, n_cols, _ = cleanup_size(self.get_input_grid_size_representation())
        grid = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                if ((i, j) in input_position):
                    grid[i][j] = 1

        # Determine how to move object until collision
        direction_coordinates, direction_representation = cleanup_centre(self.input_parameters_representation[0])
        horizontal_diff = int(direction_coordinates[0][0])
        vertical_diff = -int(direction_coordinates[0][1])

        # Repeatedly move object until it goes out of bounds
        n_iterations = 0
        while True:
            try:
                next_grid = np.zeros((n_rows, n_cols))
                for i in range(n_rows):
                    for j in range(n_cols):
                        if grid[i][j]:
                            if out_of_bounds(self.get_current_grid(), i + vertical_diff, j + horizontal_diff) or out_of_bounds(ARCGrid(next_grid), i + vertical_diff, j + horizontal_diff):
                                raise OutOfBoundsException
                            next_grid[i + vertical_diff][j + horizontal_diff] = 1
            except OutOfBoundsException:
                break
            except Exception as e:
                raise e
            grid = next_grid
            n_iterations += 1
            if (n_iterations > MAX_GRID_SIZE):
                break

        # Move object centre the number of iterations required, its shape need not change
        output_centre = self.input_object_representation.get_centre_representation()
        for i in range(n_iterations):
            output_centre = SSP_SPACE.bind(output_centre, direction_representation).flatten()

        output_object_representation = ARCObject(
            self.input_object_representation.get_colour_representation(),
            output_centre,
            self.input_object_representation.get_shape_representation(),
        )
        return output_object_representation


# Grow (stretch, extend, scale) operation
class GrowOperation(ARCOperation):
    # Parameter is the representation of the direction to grow in
    def __init__(self, input_object_representation=None, input_parameters_representation=[np.zeros((N_DIMENSIONS))]):
        super().__init__(
            GROW_OPERATION_ID,
            GROW_OPERATION_PRIORITY,
            GROW_OPERATION_SIMPLICITY,
            GROW_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Reproduce an object from the input set, but with the object grown until contact with another object or the bounds of the grid
    def execute(self):
        input_colour = cleanup_colour(self.input_object_representation.get_colour_representation())[0]
        input_centre = cleanup_centre(self.input_object_representation.get_centre_representation())[1]
        input_shape = self.input_object_representation.get_shape_representation()
        input_position = cleanup_position(SSP_SPACE.bind(input_centre, input_shape), self.get_input_grid_size_representation())[0]

        # Convert representation of old position to grid-pixel space
        n_rows, _, n_cols, _ = cleanup_size(self.get_input_grid_size_representation())
        grid = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                if ((i, j) in input_position):
                    grid[i][j] = 1

        # Determine how to move object until collision
        direction_coordinates = cleanup_centre(self.input_parameters_representation[0])[0]
        horizontal_diff = int(direction_coordinates[0][0])
        vertical_diff = -int(direction_coordinates[0][1])

        # Repeatedly move (and keep) object until it goes out of bounds
        n_iterations = 0
        while True:
            try:
                next_grid = np.zeros((n_rows, n_cols))
                for i in range(n_rows):
                    for j in range(n_cols):
                        if grid[i][j]:
                            if out_of_bounds(self.get_current_grid(), i + vertical_diff, j + horizontal_diff) or out_of_bounds(ARCGrid(next_grid), i + vertical_diff, j + horizontal_diff):
                                raise OutOfBoundsException
                            next_grid[i][j] = 1
                            next_grid[i + vertical_diff][j + horizontal_diff] = 1
            except OutOfBoundsException:
                break
            except Exception as e:
                raise e
            grid = next_grid
            n_iterations += 1
            if (n_iterations > MAX_GRID_SIZE):
                break

        output_object_representation = ARCObject(*encode_object(grid, input_colour))
        return output_object_representation
# -----------------------------------------------------------------------------


# Special shape-based programs
# -----------------------------------------------------------------------------
# Fill (connect) operation
class FillOperation(ARCOperation):
    # Parameter is the representation of the colour to fill with
    def __init__(self, input_object_representation=None, input_parameters_representation=[]):
        super().__init__(
            FILL_OPERATION_ID,
            FILL_OPERATION_PRIORITY,
            FILL_OPERATION_SIMPLICITY,
            FILL_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Convert the pixels within a hollow object to a certain colour
    def execute(self):
        input_centre = cleanup_centre(self.input_object_representation.get_centre_representation())[1]
        input_shape = self.input_object_representation.get_shape_representation()
        input_position = cleanup_position(SSP_SPACE.bind(input_centre, input_shape), self.get_input_grid_size_representation())[0]

        # Convert representation of old position into grid-pixel space
        n_rows, _, n_cols, _ = cleanup_size(self.get_input_grid_size_representation())
        grid = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                if ((i, j) in input_position):
                    grid[i][j] = 1

        # Construct a representation of the old and new positions in embedding space
        input_position = np.zeros((N_DIMENSIONS))
        output_position = np.zeros((N_DIMENSIONS))
        for i in range(n_rows):
            for j in range(n_cols):
                if (grid[i][j]):
                    input_position += SSP_SPACE.encode([j - (n_cols - 1) / 2, (n_rows - 1) / 2 - i]).flatten()
                    output_position += SSP_SPACE.encode([j - (n_cols - 1) / 2, (n_rows - 1) / 2 - i]).flatten()
                # If missing pixel is bounded above and below or left and right, then fill it in
                elif ((not grid[i][j]) and (
                    ((grid[i,:j] > 0).any() and (grid[i,(j+1):] > 0).any()) or
                    ((grid[:i,j] > 0).any() and (grid[(i+1):,j] > 0).any())
                )):
                    output_position += SSP_SPACE.encode([j - (n_cols - 1) / 2, (n_rows - 1) / 2 - i]).flatten()
        output_shape = normalize(SSP_SPACE.bind(output_position, SSP_SPACE.invert(input_centre)))

        output_object_representation = ARCObject(
            self.input_object_representation.get_colour_representation(),
            self.input_object_representation.get_centre_representation(),
            output_shape,
        )
        return output_object_representation


# Hollow (empty, fill black) operation
class HollowOperation(ARCOperation):
    # No parameter needed
    def __init__(self, input_object_representation=None, input_parameters_representation=[]):
        super().__init__(
            HOLLOW_OPERATION_ID,
            HOLLOW_OPERATION_PRIORITY,
            HOLLOW_OPERATION_SIMPLICITY,
            HOLLOW_OPERATION_N_PARAMETERS,
            input_object_representation,
            input_parameters_representation,
        )
        return

    # Convert the pixels within the bounds of an object into background
    def execute(self):
        input_centre = cleanup_centre(self.input_object_representation.get_centre_representation())[1]
        input_shape = self.input_object_representation.get_shape_representation()
        input_position = cleanup_position(SSP_SPACE.bind(input_centre, input_shape), self.get_input_grid_size_representation())[0]

        # Convert representation of old position to grid-pixel space
        n_rows, _, n_cols, _ = cleanup_size(self.get_input_grid_size_representation())
        grid = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                if ((i, j) in input_position):
                    grid[i][j] = 1

        # Construct a representation of the old and new positions in embedding space
        input_position = np.zeros((N_DIMENSIONS))
        output_position = np.zeros((N_DIMENSIONS))
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(grid, pad_width=1, mode="constant"), (3, 3))
        for i in range(n_rows):
            for j in range(n_cols):
                if (grid[i][j]):
                    input_position += SSP_SPACE.encode([j - (n_cols - 1) / 2, (n_rows - 1) / 2 - i]).flatten()
                    output_position += SSP_SPACE.encode([j - (n_cols - 1) / 2, (n_rows - 1) / 2 - i]).flatten()
                # If existing pixel is surrounded above and below or left and right, then hollow it out
                if (windows[i][j][1][1]) and (
                    (windows[i][j][1][0] and windows[i][j][1][2] and windows[i][j][0][1] and windows[i][j][2][1])
                ) or (
                    (windows[i][j][1][0] and windows[i][j][1][2]) and ((not (grid[:i,:] > 0).any()) and (not (grid[(i + 1):,:] > 0).any()))
                ) or (
                    (windows[i][j][0][1] and windows[i][j][2][1]) and ((not (grid[:,:j] > 0).any()) and (not (grid[:,(j + 1):] > 0).any()))
                ):
                    output_position -= SSP_SPACE.encode([j - (n_cols - 1) / 2, (n_rows - 1) / 2 - i]).flatten()
        output_shape = normalize(SSP_SPACE.bind(output_position, SSP_SPACE.invert(input_centre)))

        output_object_representation = ARCObject(
            self.input_object_representation.get_colour_representation(),
            self.input_object_representation.get_centre_representation(),
            output_shape,
        )
        return output_object_representation
# -----------------------------------------------------------------------------


# Action utilities
# -----------------------------------------------------------------------------
# To detect invalid moves
class OutOfBoundsException(Exception): pass


# Check if an object's new location is out-of-bounds according to previously-made objects
def out_of_bounds(partial_grid, row, col):
    return (
        (row < 0) or
        (col < 0) or
        (row >= partial_grid.get_n_rows()) or
        (col >= partial_grid.get_n_cols()) or
        (partial_grid.get_pixel(row, col))
    )
# -----------------------------------------------------------------------------


# Infer what transform relates two objects
# -----------------------------------------------------------------------------
# Determine the programs that exactly connect two objects
def program_heuristic(input_object, output_object, in_grid, out_grid, grid_size_hypothesis):
    EPSILON = 0.01
    N_OBJ_PROPERTIES = 3
    program_candidates = []
    program_descriptions = {}

    # Query for approximations of each object property
    in_colour = input_object.get_colour_representation()
    out_colour = output_object.get_colour_representation()
    in_centre = input_object.get_centre_representation()
    out_centre = output_object.get_centre_representation()
    in_shape = input_object.get_shape_representation()
    out_shape = output_object.get_shape_representation()

    # Compute similarities between each object property
    similarities = input_object.get_similarity_to(output_object)

    # Object is unchanged, so identity program applies
    if (abs(np.sum(similarities) - N_OBJ_PROPERTIES) < EPSILON):
        program_hash = hash(IdentityOperation(input_object))
        program_candidates.append(program_hash)
        program_descriptions[program_hash] = (IdentityOperation, [])

        if np.all(cleanup_centre(out_centre)[0] == [[0, 0]]):
            program = ExtractOperation(input_object)
            program.set_input_grid_size_representation(in_grid.get_size_representation())
            program.set_current_grid(copy.deepcopy(out_grid))
            program_result = program.execute()
            if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON) and (np.linalg.norm(out_grid.get_size_representation() - program.get_current_grid().get_size_representation()) < EPSILON) and (type(grid_size_hypothesis) == FunctionSizeHypothesis):
                program_hash = hash(program)
                program_candidates.append(program_hash)
                program_descriptions[program_hash] = (program.__class__, [])

    # Object has colour changed, so some recolour-based program applies
    elif (min(similarities) == similarities[0]):
        program_parameter = [out_colour]
        program = RecolourOperation(input_object, program_parameter)
        program.set_input_grid_size_representation(in_grid.get_size_representation())
        program_result = program.execute()
        if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON):
            program_hash = hash(program)
            program_candidates.append(program_hash)
            program_descriptions[program_hash] = (program.__class__, program_parameter)

    # Object has centre changed, so some recentre-based program applies
    elif (min(similarities) == similarities[1]):
        if np.all(cleanup_centre(out_centre)[0] == [[0, 0]]):
            program = ExtractOperation(input_object)
            program.set_input_grid_size_representation(in_grid.get_size_representation())
            program.set_current_grid(copy.deepcopy(out_grid))
            program_result = program.execute()
            if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON) and (np.linalg.norm(out_grid.get_size_representation() - program.get_current_grid().get_size_representation()) < EPSILON) and (type(grid_size_hypothesis) == FunctionSizeHypothesis):
                program_hash = hash(program)
                program_candidates.append(program_hash)
                program_descriptions[program_hash] = (program.__class__, [])

        program_parameter = [out_centre]
        program = RecentreOperation(input_object, program_parameter)
        program.set_input_grid_size_representation(in_grid.get_size_representation())
        program_result = program.execute()
        if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON):
            program_hash = hash(program)
            program_candidates.append(program_hash)
            program_descriptions[program_hash] = (program.__class__, program_parameter)

        program_parameter = [cleanup_centre(SSP_SPACE.bind(out_centre, SSP_SPACE.invert(in_centre)))[1].flatten()]
        program = MoveOperation(input_object, program_parameter)
        program.set_input_grid_size_representation(in_grid.get_size_representation())
        program_result = program.execute()
        if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON):
            program_hash = hash(program)
            program_candidates.append(program_hash)
            program_descriptions[program_hash] = (program.__class__, program_parameter)

        program_parameter = cleanup_centre(SSP_SPACE.bind(out_centre, SSP_SPACE.invert(in_centre)))[0]
        if (np.linalg.norm(program_parameter)):
            program_parameter = [SSP_SPACE.encode(program_parameter / np.linalg.norm(program_parameter)).flatten()]
            program = GravityOperation(input_object, program_parameter)
            program.set_input_grid_size_representation(in_grid.get_size_representation())
            program.set_current_grid(copy.deepcopy(out_grid).remove_object(output_object))
            program_result = program.execute()
            if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON):
                program_hash = hash(program)
                program_candidates.append(program_hash)
                program_descriptions[program_hash] = (program.__class__, program_parameter)

        program_parameter = cleanup_centre(SSP_SPACE.bind(out_centre, SSP_SPACE.invert(in_centre)))[0]
        if (np.linalg.norm(program_parameter)):
            program_parameter = [SSP_SPACE.encode(program_parameter / np.linalg.norm(program_parameter)).flatten()]
            program = GrowOperation(input_object, program_parameter)
            program.set_input_grid_size_representation(in_grid.get_size_representation())
            program.set_current_grid(copy.deepcopy(out_grid).remove_object(output_object))
            program_result = program.execute()
            if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON):
                program_hash = hash(program)
                program_candidates.append(program_hash)
                program_descriptions[program_hash] = (program.__class__, program_parameter)

    # Object has changed shape, so some reshape-based program applies
    elif (min(similarities) == similarities[2]):
        program_parameter = [out_shape]
        program = ReshapeOperation(copy.deepcopy(input_object), program_parameter)
        program.set_input_grid_size_representation(in_grid.get_size_representation())
        program.set_current_grid(copy.deepcopy(out_grid))
        program_result = program.execute()
        if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON):
            program_hash = hash(program)
            program_candidates.append(program_hash)
            program_descriptions[program_hash] = (program.__class__, program_parameter)

        program_parameter = cleanup_centre(SSP_SPACE.bind(out_centre, SSP_SPACE.invert(in_centre)))[0]
        if (np.linalg.norm(program_parameter)):
            program_parameter = [SSP_SPACE.encode(program_parameter / np.linalg.norm(program_parameter)).flatten()]
            program = GrowOperation(input_object, program_parameter)
            program.set_input_grid_size_representation(in_grid.get_size_representation())
            program.set_current_grid(copy.deepcopy(out_grid).remove_object(output_object))
            program_result = program.execute()
            if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON):
                program_hash = hash(program)
                program_candidates.append(program_hash)
                program_descriptions[program_hash] = (program.__class__, program_parameter)

        program = FillOperation(copy.deepcopy(input_object))
        program.set_input_grid_size_representation(in_grid.get_size_representation())
        program.set_current_grid(copy.deepcopy(out_grid))
        program_result = program.execute()
        if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON):
            program_hash = hash(program)
            program_candidates.append(program_hash)
            program_descriptions[program_hash] = (program.__class__, [])

        program = HollowOperation(input_object)
        program.set_input_grid_size_representation(in_grid.get_size_representation())
        program.set_current_grid(copy.deepcopy(out_grid))
        program_result = program.execute()
        if (abs(np.sum(program_result.get_similarity_to(output_object)) - N_OBJ_PROPERTIES) < EPSILON):
            program_hash = hash(program)
            program_candidates.append(program_hash)
            program_descriptions[program_hash] = (program.__class__, [])

        # TODO: Solve the always-consider-Generate problem better
        program_parameter = [out_colour, out_centre, out_shape]
        program_hash = hash(GenerateOperation(input_object, program_parameter))
        program_candidates.append(program_hash)
        program_descriptions[program_hash] = (GenerateOperation, program_parameter)

    if (not program_candidates):
        program_parameter = [out_colour, out_centre, out_shape]
        program_hash = hash(GenerateOperation(input_object, program_parameter))
        program_candidates.append(program_hash)
        program_descriptions[program_hash] = (GenerateOperation, program_parameter)

    return set(program_candidates), program_descriptions
# -----------------------------------------------------------------------------


def main():
    return


if __name__ == "__main__":
    main()
