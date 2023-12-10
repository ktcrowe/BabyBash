import tkinter as tk
import tkinter.font as tkfont
import sounddevice as sd


# GUI for selecting input and output devices
class DeviceSelector:
    def __init__(self):
        self.master = tk.Tk()
        self.master.title("BabyBash")

        self.input_menu = None
        self.output_menu = None

        self.device_list = sd.query_devices()
        self.input_devices = [device['name'] for device in self.device_list if device['max_input_channels'] > 0]
        self.output_devices = [device['name'] for device in self.device_list if device['max_output_channels'] > 0]

        self.input_var = tk.StringVar(self.master)
        self.output_var = tk.StringVar(self.master)

        self.create_widgets()

    # Create the GUI layout
    def create_widgets(self):
        label_instructions = tk.Label(self.master, text="Select your audio devices to begin baby bashing.")
        font: dict[str, any] = tkfont.Font(font=label_instructions['font']).actual()
        label_instructions.configure(font=(font['family'], '16', 'bold'))
        label_instructions.pack()

        tk.Label(self.master, text="Select Input Device:").pack()

        self.input_menu = tk.OptionMenu(self.master, self.input_var, *self.input_devices)
        self.input_menu.pack()

        tk.Label(self.master, text="Select Output Device:").pack()

        self.output_menu = tk.OptionMenu(self.master, self.output_var, *self.output_devices)
        self.output_menu.pack()

        tk.Button(self.master, text="Confirm", command=self.confirm).pack()

    def confirm(self):
        input_device = self.input_var.get()
        output_device = self.output_var.get()
        print(f"Input Device: {input_device}")
        print(f"Output Device: {output_device}")
        self.master.destroy()

    # Prompt the user to select input and output devices
    def select_devices(self):
        self.master.mainloop()
        return self.input_var.get(), self.output_var.get()
