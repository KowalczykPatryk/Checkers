"""
Contains functions and classes related to plotting.
"""

import matplotlib.pyplot as plt

class Plot:
    """
    Dynamically Updating Plot.
    """
    def __init__(self, title: str, xlabel: str, ylabel: str) -> None:
        plt.ion()  # turning interactive mode on
        self.values = []
        self.graph = None
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        # using a dedicated Figure and Axes per Plot instance to avoid shared state
        self.fig, self.ax = plt.subplots()

    def update(self, value: float) -> None:
        """
        Appends value to the previous values and displays.
        """
        self.values.append(value)
        if self.graph is not None:
            self.graph.remove()
        (self.graph,) = self.ax.plot(self.values)
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.fig.canvas.draw()
        plt.pause(0.25)
    def save(self, filename: str) -> None:
        """
        Saves plot with the provided filename.
        """
        self.fig.savefig(filename)
        plt.close(self.fig)
        print("Plot saved successfully!")
