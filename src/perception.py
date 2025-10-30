# Isaac Joffe, 2025
# Defines the rules by which grids can be parsed into objects


# VSA-related dependencies
import numpy as np
# Machine-learned-related dependencies
import skimage


# General object perception setup
# -----------------------------------------------------------------------------
# System hyperparameter
N_OBJECT_HYPOTHESES = 6       # 2-connected, 1-connected, colour, pixel, vertical, horizontal

# Universal identifiers of different object hypotheses
EIGHTCONN_ID = 2
FOURCONN_ID = 3
COLOUR_ID = 1
PIXEL_ID = 0
VERTICAL_ID = 4
HORIZONTAL_ID = 5

# Priorities of different object hypotheses
EIGHTCONN_PRIORITY = 0
FOURCONN_PRIORITY = 2
COLOUR_PRIORITY = 1
PIXEL_PRIORITY = 4
VERTICAL_PRIORITY = 3
HORIZONTAL_PRIORITY = 3

# Mapping from object hypothesis identifiers to printable names
OBJECT_HYPOTHESIS_MAP = {
    EIGHTCONN_ID: "2-Connected",
    FOURCONN_ID: "1-Connected",
    COLOUR_ID: "Colour",
    PIXEL_ID: "Pixel",
    VERTICAL_ID: "Vertical",
    HORIZONTAL_ID: "Horizontal",
}
# -----------------------------------------------------------------------------


# Standard way of representing an object hypothesis
# -----------------------------------------------------------------------------
# Abstract class representing a general object hypothesis
class ARCObjectHypothesis():
    # Construct the object hypothesis object
    def __init__(self, identifier, priority, input_grid):
        # Store general information about what object hypothesis this is
        self.__identifier = identifier
        self.__priority = priority

        # Object hypothesis deduction is defined by the grid it operates on
        self.grid = input_grid
        return

    # Convert the object hypothesis into a printable format
    # Final
    def __str__(self):
        return f"{OBJECT_HYPOTHESIS_MAP[self.get_identifier()]} Object Hypothesis"

    # Return the identifier of this object hypothesis
    # Final
    def get_identifier(self):
        # Identifier should not be changed later
        return self.__identifier

    # Return the execution priority of this object hypothesis
    # Final
    def get_priority(self):
        # Priority should not be changed later
        return self.__priority

    # Execute the object hypothesis
    # Abstract
    def perceive(self):
        raise NotImplementedError("Object hypothesis must implement a perceive() method")
# -----------------------------------------------------------------------------


# Default object hypotheses
# -----------------------------------------------------------------------------
# Default object hypothesis, defines objects as groups of contiguous pixels of the same colour
class EightConnectedHypothesis(ARCObjectHypothesis):
    # No parameter needed
    def __init__(self, input_grid):
        super().__init__(
            EIGHTCONN_ID,
            EIGHTCONN_PRIORITY,
            input_grid,
        )
        return

    # Group pixels into objects based on 8-connectivity
    def perceive(self):
        # Partition grid into groups of pixels adjacent by edge or corner
        obj_grid, n = skimage.measure.label(self.grid.get_data(), background=0, return_num=True)
        obj_grids = [(obj_grid == i) * i for i in range(n + 1)][1:]
        obj_colours = {}
        for i in range(1, n + 1):
            indices = np.where(obj_grid == i)
            obj_colours[i] = int(self.grid.get_pixel(indices[0][0], indices[1][0]))
        return obj_grids, obj_colours


# Another object hypothesis, defines objects as groups of contiguous pixels of the same colour
class FourConnectedHypothesis(ARCObjectHypothesis):
    # No parameter needed
    def __init__(self, input_grid):
        super().__init__(
            FOURCONN_ID,
            FOURCONN_PRIORITY,
            input_grid,
        )
        return

    # Group pixels into objects based on 1-connectivity
    def perceive(self):
        # Partition grid into groups of pixels adjacent by edge
        obj_grid, n = skimage.measure.label(self.grid.get_data(), background=0, return_num=True, connectivity=1)
        obj_grids = [(obj_grid == i) * i for i in range(n + 1)][1:]
        obj_colours = {}
        for i in range(1, n + 1):
            indices = np.where(obj_grid == i)
            obj_colours[i] = int(self.grid.get_pixel(indices[0][0], indices[1][0]))
        return obj_grids, obj_colours
# -----------------------------------------------------------------------------


# Naive object hypotheses
# -----------------------------------------------------------------------------
# Colour object hypothesis, defines objects as groups of pixels of the same colour
class ColourHypothesis(ARCObjectHypothesis):
    # No parameter needed
    def __init__(self, input_grid):
        super().__init__(
            COLOUR_ID,
            COLOUR_PRIORITY,
            input_grid,
        )
        return

    # Group pixels into objects based purely on colour
    def perceive(self):
        colours = np.unique(np.append(np.unique(self.grid.get_data()), 0))
        obj_grids = [(self.grid.get_data() == colour) * i for i, colour in enumerate(colours)][1:]
        obj_colours = {}
        for i, colour in enumerate(colours):
            if (colour == 0):
                continue
            obj_colours[i] = int(colour)
        return obj_grids, obj_colours


# Pixel object hypothesis, defines objects as individual pixels
class PixelHypothesis(ARCObjectHypothesis):
    # No parameter needed
    def __init__(self, input_grid):
        super().__init__(
            PIXEL_ID,
            PIXEL_PRIORITY,
            input_grid,
        )
        return

    # Group individual pixels as objects
    def perceive(self):
        obj_grids = []
        obj_colours = {}
        n_objects = 0
        for i in range(self.grid.get_n_rows()):
            for j in range(self.grid.get_n_cols()):
                if self.grid.get_pixel(i, j):
                    # Each coloured pixel is a new object
                    n_objects += 1
                    pixel_grid = (self.grid.get_data() * 0).copy()
                    pixel_grid[i][j] = n_objects
                    obj_grids.append(pixel_grid)
                    obj_colours[n_objects] = int(self.grid.get_pixel(i, j))
        return obj_grids, obj_colours
# -----------------------------------------------------------------------------


# Direction-based object hypotheses
# -----------------------------------------------------------------------------
# Vertical object hypothesis, defines objects as pixels of the same colour connected vertically
class VerticalHypothesis(ARCObjectHypothesis):
    # No parameter needed
    def __init__(self, input_grid):
        super().__init__(
            VERTICAL_ID,
            VERTICAL_PRIORITY,
            input_grid,
        )
        return

    # Group pixels into objects based on vertical connectivity
    def perceive(self):
        obj_grids = []
        obj_colours = {}
        n_objects = 0

        # Split the grid into columns
        for column in range(self.grid.get_n_cols()):
            column_grid = self.grid.get_data().copy()
            column_grid[:, [i for i in range(self.grid.get_n_cols()) if i != column]] = 0

            # Partition these columns into different objects of same colour
            obj_grid, n = skimage.measure.label(column_grid, background=0, return_num=True, connectivity=2)
            for i in range(1, n+1):
                n_objects += 1
                obj_grids.append((obj_grid == i) * n_objects)
                indices = np.where(obj_grid == i)
                obj_colours[n_objects] = int(self.grid.get_pixel(indices[0][0], indices[1][0]))
        return obj_grids, obj_colours


# Horizontal object hypothesis, defines objects as pixels of the same colour connected horizontally
class HorizontalHypothesis(ARCObjectHypothesis):
    # No parameter needed
    def __init__(self, input_grid):
        super().__init__(
            HORIZONTAL_ID,
            HORIZONTAL_PRIORITY,
            input_grid,
        )
        return

    # Group pixels into objects based on horizontal connectivity
    def perceive(self):
        obj_grids = []
        obj_colours = {}
        n_objects = 0

        # Split the grid into rows
        for row in range(self.grid.get_n_cols()):
            row_grid = self.grid.get_data().copy()
            row_grid[[i for i in range(self.grid.get_n_rows()) if i != row], :] = 0

            # Partition these columns into different objects of same colour
            obj_grid, n = skimage.measure.label(row_grid, background=0, return_num=True, connectivity=2)
            for i in range(1, n+1):
                n_objects += 1
                obj_grids.append((obj_grid == i) * n_objects)
                indices = np.where(obj_grid == i)
                obj_colours[n_objects] = int(self.grid.get_pixel(indices[0][0], indices[1][0]))
        return obj_grids, obj_colours
# -----------------------------------------------------------------------------


# Perform perception
# -----------------------------------------------------------------------------
# Convert pixel grids into sets of object masks and object colours
def perceive_grid(grid, object_hypothesis):
    if (object_hypothesis == EIGHTCONN_ID):
        hypothesis = EightConnectedHypothesis(grid)
    elif (object_hypothesis == FOURCONN_ID):
        hypothesis = FourConnectedHypothesis(grid)
    elif (object_hypothesis == COLOUR_ID):
        hypothesis = ColourHypothesis(grid)
    elif (object_hypothesis == VERTICAL_ID):
        hypothesis = VerticalHypothesis(grid)
    elif (object_hypothesis == HORIZONTAL_ID):
        hypothesis = HorizontalHypothesis(grid)
    elif (object_hypothesis == PIXEL_ID):
        hypothesis = PixelHypothesis(grid)
    else:
        hypothesis = EightConnectedHypothesis(grid)
    # Deduce objects in grids based on hypothesis
    obj_grids, obj_colours = hypothesis.perceive()
    return obj_grids, obj_colours


# Factory to generate hypotheses in a structured way
def generate_object_hypothesizer(object_hypothesis):
    if (object_hypothesis == EIGHTCONN_ID):
        return EightConnectedHypothesis
    elif (object_hypothesis == FOURCONN_ID):
        return FourConnectedHypothesis
    elif (object_hypothesis == COLOUR_ID):
        return ColourHypothesis
    elif (object_hypothesis == PIXEL_ID):
        return PixelHypothesis
    elif (object_hypothesis == VERTICAL_ID):
        return VerticalHypothesis
    elif (object_hypothesis == HORIZONTAL_ID):
        return HorizontalHypothesis
    else:
        print("WARNING: Reverting back to default object hypothesis")
        return EightConnectedHypothesis
# -----------------------------------------------------------------------------


def main():
    return


if __name__ == "__main__":
    main()
