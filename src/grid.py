# Isaac Joffe, 2025
# Defines the storage medium of an ARC grid


# User-defined libraries
from vsa import *
# VSA-related dependencies
import numpy as np


# Standard way of representing an ARC grid with basic operations
# -----------------------------------------------------------------------------
# Represents an ARC grid
class ARCGrid():
    # Construct a given ARC grid
    def __init__(self, data=None, size_representation=None):
        # Mode: Constructor reproduces existing grid
        if data is not None:
            # Copy over data to base grid off of, ensure NumPy array
            self.__data = np.array(data)
            # Compute grid sizes based off of this
            self.__n_rows = self.__data.shape[0]
            self.__n_cols = self.__data.shape[1]
            self.__size_representation = bundle_size(self.__n_rows, self.__n_cols)

        # Mode: Constructor generates empty grid of some size
        elif size_representation is not None:
            # Store the grid size in both embedding and pixel space
            self.__size_representation = size_representation.flatten()
            self.__n_rows, _, self.__n_cols, _ = cleanup_size(size_representation)
            # Create an empty grid of the desired size
            self.__data = np.zeros((self.__n_rows, self.__n_cols))

        # Mode: Constructor generates a null grid (empty 1x1 grid)
        else:
            self.__data = np.zeros((1, 1))
            self.__n_rows = 1
            self.__n_cols = 1
            self.__size_representation = bundle_size(self.__n_rows, self.__n_cols)
        return

    # Convert the grid into a printable format
    def __str__(self):
        string_representation = ""
        string_representation += "+" + "--" * self.get_n_cols() + "+" + "\n"
        for i in range(self.get_n_rows()):
            string_representation += "|"
            for j in range(self.get_n_cols()):
                string_representation += COLOUR_MAP[self.get_pixel(i, j)]
            string_representation += "|" + "\n"
        string_representation += "+" + "--" * self.get_n_cols() + "+"
        return string_representation

    # Determine if two ARC grids are equivalent (sizes match, each pixel matches)
    def __eq__(self, other):
        return bool(
            (self.get_n_rows() == other.get_n_rows()) and
            (self.get_n_cols() == other.get_n_cols()) and
            (self.get_data() == other.get_data()).all()
        )

    # Get the pixel data representing the grid
    def get_data(self):
        return self.__data

    # Get a pixel at a certain position in the grid
    def get_pixel(self, row, col):
        return int(self.__data[row][col])

    # Get the size of the grid in embedding space
    def get_size_representation(self):
        return self.__size_representation

    # Get the height of the grid
    def get_n_rows(self):
        return self.__n_rows

    # Get the width of the grid
    def get_n_cols(self):
        return self.__n_cols

    # Get the size of the grid in pixel space (as height and width)
    def get_size(self):
        return self.__n_rows, self.__n_cols

    # Update the dimensions of the grid
    def set_size_representation(self, size_representation):
        # Update both ways of storing the grid size
        self.__size_representation = size_representation.flatten()
        self.__n_rows, _, self.__n_cols, _ = cleanup_size(size_representation)

        # Crop the grid (naively)
        self.__data = self.__data[:self.get_n_rows(), :self.get_n_cols()]
        return

    # Add a represented object to the cumulative grid
    def add_object(self, object_representation):
        colour = cleanup_colour(object_representation.get_colour_representation())[0]
        centre = cleanup_centre(object_representation.get_centre_representation())[1]
        shape = object_representation.get_shape_representation()
        position = cleanup_position(SSP_SPACE.bind(centre, shape), self.get_size_representation())[0]
        # Colour each pixel belonging to the object
        for pixel in position:
            self.__data[pixel[0]][pixel[1]] = colour
        return self

    # Remove a pixelated object from the cumulative grid
    def remove_object(self, object_representation):
        colour = cleanup_colour(object_representation.get_colour_representation())[0]
        centre = cleanup_centre(object_representation.get_centre_representation())[1]
        shape = object_representation.get_shape_representation()
        position = cleanup_position(SSP_SPACE.bind(centre, shape), self.get_size_representation())[0]
        # Black out each pixel belonging to the object
        for pixel in position:
            if (self.get_pixel(pixel[0], pixel[1]) == colour):
                self.__data[pixel[0]][pixel[1]] = 0
        return self
# -----------------------------------------------------------------------------


# Utility functions for grid I/O
# -----------------------------------------------------------------------------
# Print a demonstration pair of grids in ARC format
def print_demonstration(in_grid, out_grid):
    # If grids have the same number of rows, then print left to right
    if (in_grid.get_n_rows() == out_grid.get_n_rows()):
        print("+" + "--" * in_grid.get_n_cols() + "+" + "      " + "+" + "--" * out_grid.get_n_cols() + "+")
        for i in range(in_grid.get_n_rows()):
            print("|", end="")
            for j in range(in_grid.get_n_cols()):
                print(COLOUR_MAP[in_grid.get_pixel(i, j)], end="")
            if (i == in_grid.get_n_rows() // 2):
                print("| ---> |", end="")
            else:
                print("|      |", end="")
            for j in range(out_grid.get_n_cols()):
                print(COLOUR_MAP[out_grid.get_pixel(i, j)], end="")
            print("|")
        print("+" + "--" * in_grid.get_n_cols() + "+" + "      " + "+" + "--" * out_grid.get_n_cols() + "+")

    # Otherwise, print up to down
    else:
        print(in_grid)
        print(" " * min(in_grid.get_n_cols(), out_grid.get_n_cols()) + "|")
        print(" " * min(in_grid.get_n_cols(), out_grid.get_n_cols()) + "v")
        print(out_grid)
    return


# Print summary of results on an ARC task
def print_grid_summary(in_grids, gt_out_grids, gen_out_grids):
    n_correct = 0
    for i in range(len(in_grids)):
        # Print desired transformation
        print(f"Demonstration #{i+1}:")
        print_demonstration(in_grids[i], gt_out_grids[i])

        # Print results from model
        is_correct = (gen_out_grids[i] == gt_out_grids[i])
        print("CORRECT" if (is_correct) else "INCORRECT")
        if (is_correct):
            n_correct += 1
        else:
            print(gen_out_grids[i])
        print()

    accuracy = n_correct / len(in_grids)
    print(f"Total Accuracy: {accuracy * 100:.1f}%")
    return accuracy
# -----------------------------------------------------------------------------


def main():
    return


if __name__ == "__main__":
    main()
