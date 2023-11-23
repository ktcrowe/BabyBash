# Plotting and visualization of audio data
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting data
from matplotlib.animation import FuncAnimation  # For animating the plot


class AudioDataPlotter:
    def __init__(self, n_mfcc, mfcc_range):
        self.n_mfcc = n_mfcc
        self.mfcc_range = mfcc_range
        self.prediction_text = ''  # the text to be displayed on the GUI to indicate whether a baby is detected TODO: fix this
        self.plt = plt
        self.mfccs = np.zeros(self.n_mfcc)  # Initialize the MFCCs to zeros

        # Define the plot to display the input audio in real-time
        self.plt.style.use('fast')  # Set a fast plotting style for real-time updates
        self.fig, self.ax = self.plt.subplots()  # Initialize the plot figure and axis
        self.x_axis = np.arange(n_mfcc)  # Generate x-axis values corresponding to the MFCC coefficients
        self.line, = self.ax.plot(self.x_axis, np.zeros(n_mfcc))  # Create an initial line object with x-axis and y-axis data for the plot

        # Set the y-axis and x-axis limits for the plot
        self.ax.set_ylim(-self.mfcc_range, self.mfcc_range)
        self.ax.set_xlim(0, n_mfcc - 1)

        # Label the x-axis and y-axis of the plot
        self.ax.set_xlabel('MFCC Coefficients')
        self.ax.set_ylabel('Amplitude')

        self.ax.set_title('Real-time MFCC')  # Set the title for the plot

        # Initialize the text element on the plot to display when a baby is crying
        self.text_element = self.ax.text(0.5, 0.1, '', horizontalalignment='center', verticalalignment='center',
                                         transform=self.ax.transAxes)

    # Update the MFCC data
    def update_mfcc_data(self, new_mfccs):
        self.mfccs = new_mfccs

    # Update the prediction text
    def update_prediction_text(self, text):
        self.prediction_text = text

    # Update the plot with new data
    def update_plot(self, frame):
        """
        This function gets called by the animation framework with a new frame
        (essentially an increment in time/intervals).
        """
        self.line.set_ydata(self.mfccs)  # Set new y-data for the line object
        self.text_element.set_text(self.prediction_text)  # Update the text based on the prediction

        """
        Return the line object that has been modified, which is necessary for blitting to work.
        "Blitting" means that the animation only re-draws the parts that have actually changed to improve the animation
        """
        return self.line, self.text_element

    # animate the plot
    def animate(self):
        ani = FuncAnimation(self.fig, self.update_plot, blit=True, interval=10, cache_frame_data=False)
        self.plt.show()  # Display the matplotlib plot and start the animation
        return ani
