import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import pkg_resources
from fronpy.funcs import estimate
import numpy as np
import textwrap
from fronpy import __version__

# Alias common NumPy functions for direct use in the GUI
ln = log = np.log
exp = np.exp
sqrt = np.sqrt
abs = np.abs
mean = np.mean
median = np.median

datasets = {}  # Dictionary to store loaded datasets
models = {}    # Dictionary to store estimated models

def set_window_icon(window):
    """Set the FronPy.ico icon for a window."""
    icon_path = pkg_resources.resource_filename('fronpy', 'misc/FronPy.ico')
    try:
        window.iconbitmap(icon_path)
    except:
        print("Icon not found; default icon will be used.")


def load_dataset():
    """Load a dataset and store it in the datasets dictionary."""
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        def confirm_load(first_row_headers):
            try:
                dataset_name = file_path.split("/")[-1]
                if first_row_headers:
                    datasets[dataset_name] = pd.read_csv(file_path)
                else:
                    datasets[dataset_name] = pd.read_csv(file_path, header=None)
                    datasets[dataset_name].columns = [f"x{i+1}" for i in range(datasets[dataset_name].shape[1])]
                messagebox.showinfo("Success", f"Loaded {dataset_name}!")
                options_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")
        
        # Pop-up window for column options
        options_window = tk.Toplevel()
        options_window.title("Load Dataset Options")
        set_window_icon(options_window)
        options_window.geometry("300x150")
        
        tk.Label(options_window, text="Does the first row contain column names?").pack(pady=10)
        tk.Button(options_window, text="Yes", command=lambda: confirm_load(True)).pack(pady=5)
        tk.Button(options_window, text="No", command=lambda: confirm_load(False)).pack(pady=5)


def open_data_viewer():
    """Pop-out window to display datasets in a scrollable, spreadsheet-like view with add/remove column functionality."""

    def populate_table(dataset_name):
        dataset = datasets.get(dataset_name)
        if dataset is not None:
            # Clear the table
            for row in tree.get_children():
                tree.delete(row)

            # Insert column headers
            tree["columns"] = list(dataset.columns)
            tree["show"] = "headings"
            for col in dataset.columns:
                tree.heading(col, text=col)
                tree.column(col, width=70, minwidth=70, anchor="w")  # Set minimum column width

            # Insert rows
            for _, row in dataset.iterrows():
                tree.insert("", tk.END, values=list(row))
        else:
            messagebox.showerror("Error", f"Dataset {dataset_name} not found.")

    def add_column():
        dataset_name = dataset_dropdown.get()
        dataset = datasets.get(dataset_name)
        if not dataset_name or dataset is None:
            messagebox.showerror("Error", "No dataset selected!")
            return

        column_name = col_name_entry.get()
        formula = formula_entry.get()
        if not column_name or not formula:
            messagebox.showerror("Error", "Please provide a column name and formula.")
            return

        try:
            # Define a safe evaluation environment
            safe_globals = {"__builtins__": None}
            safe_locals = {
                "log": np.log,
                "exp": np.exp,
                "sqrt": np.sqrt,
                "abs": np.abs,
                "mean": np.mean,
                "median": np.median,
                **{col: dataset[col].values for col in dataset.columns}  # Map columns as variables
            }

            # Evaluate the formula
            datasets[dataset_name][column_name] = eval(formula, safe_globals, safe_locals)
            messagebox.showinfo("Success", f"Added column '{column_name}' to {dataset_name}.")
            populate_table(dataset_name)  # Refresh table
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add column: {e}")

    def remove_column():
        dataset_name = dataset_dropdown.get()
        dataset = datasets.get(dataset_name)
        if not dataset_name or dataset is None:
            messagebox.showerror("Error", "No dataset selected!")
            return

        column_name = remove_col_entry.get()
        if not column_name or column_name not in dataset.columns:
            messagebox.showerror("Error", f"Column '{column_name}' not found in dataset.")
            return

        try:
            datasets[dataset_name].drop(columns=[column_name], inplace=True)
            messagebox.showinfo("Success", f"Removed column '{column_name}' from {dataset_name}.")
            populate_table(dataset_name)  # Refresh table
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove column: {e}")

    # Viewer window
    viewer = tk.Toplevel()
    viewer.title("Data Viewer")
    set_window_icon(viewer)
    viewer.geometry("1000x700")

    dataset_names = list(datasets.keys())
    if not dataset_names:
        messagebox.showerror("Error", "No datasets loaded!")
        viewer.destroy()
        return

    tk.Label(viewer, text="Select Dataset:").pack(pady=5)
    dataset_dropdown = ttk.Combobox(viewer, values=dataset_names)
    dataset_dropdown.pack(pady=5)
    dataset_dropdown.bind("<<ComboboxSelected>>", lambda e: populate_table(dataset_dropdown.get()))

    # Scrollable table frame
    frame = ttk.Frame(viewer)
    frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    # Scrollbars
    scrollbar_y = ttk.Scrollbar(frame, orient=tk.VERTICAL)
    scrollbar_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)

    # Treeview
    tree = ttk.Treeview(
        frame,
        yscrollcommand=scrollbar_y.set,
        xscrollcommand=scrollbar_x.set,
    )
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Configure scrollbars
    scrollbar_y.config(command=tree.yview)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

    scrollbar_x.config(command=tree.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    # Add a custom scroll speed (faster scrolling)
    def fast_scrollbar(event):
        """Override scrollbar speed for faster scrolling."""
        # Increase the number of units scrolled per event (e.g., scroll 5 units at a time)
        scroll_units = 5  # Change this number for faster/slower scrolling

        # Check if the event is vertical or horizontal
        if event.delta != 0:
            if event.state & 0x0001:  # Check if it's a horizontal scroll event
                tree.xview_scroll(int(-event.delta / 120) * scroll_units, "units")
            else:  # It's a vertical scroll event
                tree.yview_scroll(int(-event.delta / 120) * scroll_units, "units")

    # Bind the fast scroll to the mouse wheel
    tree.bind("<MouseWheel>", fast_scrollbar)
    # Binding the fast scroll to Button-4 and Button-5, i.e., touchpads for certain systems
    tree.bind("<Button-4>", fast_scrollbar)
    tree.bind("<Button-5>", fast_scrollbar)

    # Column manipulation - Add
    add_frame = ttk.Frame(viewer)
    add_frame.pack(fill=tk.X, pady=5, padx=10)

    tk.Label(add_frame, text="Add a new column:").grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
    tk.Label(add_frame, text="Column Name:").grid(row=1, column=0, sticky="e", padx=5)
    col_name_entry = tk.Entry(add_frame, width=20)
    col_name_entry.grid(row=1, column=1, sticky="w")

    tk.Label(add_frame, text="Formula:", anchor="w").grid(row=1, column=2, sticky="e", padx=5)
    formula_entry = tk.Entry(add_frame, width=50)
    formula_entry.grid(row=1, column=3, sticky="w")

    tk.Button(add_frame, text="Add Column", command=add_column).grid(row=1, column=4, sticky="w", padx=5)

    # Column manipulation - Remove
    remove_frame = ttk.Frame(viewer)
    remove_frame.pack(fill=tk.X, pady=5, padx=10)

    tk.Label(remove_frame, text="Remove a column:").grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
    tk.Label(remove_frame, text="Column Name:").grid(row=1, column=0, sticky="e", padx=5)
    remove_col_entry = tk.Entry(remove_frame, width=20)
    remove_col_entry.grid(row=1, column=1, sticky="w")

    tk.Button(remove_frame, text="Remove Column", command=remove_column).grid(row=1, column=2, sticky="w", padx=5)


def open_model_estimation():
    """Pop-out window for model estimation."""
    global models  # Use the global models dictionary
    # Initialize the models dictionary if not already initialized
    if not models:
        models = {}

    def update_visibility(*args):
        """Update visibility of lnmu and mu entries based on model type."""
        model_type = model_type_dropdown.get()
        if model_type in ['ng', 'nnak','ntn']:
            if model_type in ['ng', 'nnak']:
                lnmu_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")  # Show lnmu label
                lnmu_entry.grid(row=5, column=1, padx=10, pady=5)  # Show lnmu entry
                mu_label.grid_forget()  # Hide mu label
                mu_entry.grid_forget()  # Hide mu entry
            elif model_type in ['ntn']:
                mu_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")  # Show mu label
                mu_entry.grid(row=5, column=1, padx=10, pady=5)  # Show mu entry
                lnmu_label.grid_forget()  # Hide lnmu label
                lnmu_entry.grid_forget()  # Hide lnmu entry
            model_name_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
            model_name_entry.grid(row=6, column=1, padx=10, pady=5)
            cost_label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
            cost_entry.grid(row=7, column=1, padx=10, pady=5)
            starting_values_label.grid(row=8, column=0, padx=10, pady=5, sticky="w")
            starting_values_entry.grid(row=8, column=1, padx=10, pady=5)
            estimate_model_button.grid(row=9, column=0, columnspan=2, pady=10)

        else:
            lnmu_label.grid_forget()  # Hide lnmu label
            lnmu_entry.grid_forget()  # Hide lnmu entry
            mu_label.grid_forget()  # Hide mu label
            mu_entry.grid_forget()  # Hide mu entry
            model_name_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
            model_name_entry.grid(row=5, column=1, padx=10, pady=5)
            cost_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
            cost_entry.grid(row=6, column=1, padx=10, pady=5)
            starting_values_label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
            starting_values_entry.grid(row=7, column=1, padx=10, pady=5)
            estimate_model_button.grid(row=8, column=0, columnspan=2, pady=10)


    def estimate_model():
        dataset_name = dataset_dropdown.get()
        formula = equation_entry.get()
        lnsigmav = lnsigmav_entry.get()
        lnsigmau = lnsigmau_entry.get()
        model_type = model_type_dropdown.get()
        cost = cost_var.get()
        starting_values_input = starting_values_entry.get()

        # Debugging output
        print(f"Selected Dataset: {dataset_name}")
        print(f"Formula: {formula}")
        print(f"Selected Model Type: {model_type}")
        print(f"Cost Function: {cost}")
        print(f"Starting Values Input: {starting_values_input}")

        if not dataset_name or dataset_name not in datasets:
            messagebox.showerror("Error", "Please select a valid dataset.")
            return

        if not formula:
            messagebox.showerror("Error", "Please enter a formula.")
            return

        if not model_type:
            messagebox.showerror("Error", "Please select a model type.")
            return

        # Convert starting values input to a NumPy array, if provided
        starting_values = None
        if starting_values_input:
            try:
                starting_values = np.fromstring(starting_values_input, sep=',')
            except ValueError:
                messagebox.showerror("Error", "Invalid starting values. Please enter a comma-separated list of numbers.")
                return

        # Handle lnmu and mu for models in which they do not appear
        lnmu_value = lnmu_entry.get() if model_type in ['ng', 'nnak'] else None
        mu_value = mu_entry.get() if model_type in ['ntn'] else None

        try:
            dataset = datasets[dataset_name]
            result = estimate(
                frontier=formula,
                data=dataset,
                lnsigmav=lnsigmav,
                lnsigmau=lnsigmau,
                lnmu=lnmu_value,
                mu = mu_value,
                model=model_type,
                cost=cost,
                startingvalues=starting_values,
            )

            # Use the user-defined model name, or generate a default name
            model_name = model_name_entry.get() if model_name_entry.get() else f"{dataset_name}_{model_type}_{len(models) + 1}"
            models[model_name] = result

            messagebox.showinfo("Success", f"Model {model_name} estimated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Model estimation failed: {e}")

    # Main estimation window
    estimation_window = tk.Toplevel()
    estimation_window.title("Model Estimation")
    set_window_icon(estimation_window)
    estimation_window.geometry("500x600")

    # Add padding around the entire grid
    estimation_window.grid_columnconfigure(0, weight=1, minsize=200)
    estimation_window.grid_columnconfigure(1, weight=1, minsize=300)

    # Configure rows to remove excessive vertical spacing
    row_height = 30  # Set a consistent minimum row height

    for i in range(13):
        estimation_window.grid_rowconfigure(i, weight=0, minsize=row_height)

    # Select dataset
    tk.Label(estimation_window, text="Dataset:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    dataset_dropdown = ttk.Combobox(estimation_window, values=list(datasets.keys()))
    dataset_dropdown.grid(row=0, column=1, padx=10, pady=5)

    # Model type dropdown
    tk.Label(estimation_window, text="Model Type:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    model_type_dropdown = ttk.Combobox(estimation_window, values=['ng', 'nnak', 'ntn', 'nhn', 'nexp', 'nr'])
    model_type_dropdown.grid(row=1, column=1, padx=10, pady=5)
    model_type_dropdown.bind("<<ComboboxSelected>>", update_visibility)

    # Formula entry
    tk.Label(estimation_window, text="Frontier equation:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
    equation_entry = tk.Entry(estimation_window, width=50)
    equation_entry.grid(row=2, column=1, padx=10, pady=5)

    # lnsigmav entry
    tk.Label(estimation_window, text="Equation for lnsigmav:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
    lnsigmav_entry = tk.Entry(estimation_window, width=50)
    lnsigmav_entry.grid(row=3, column=1, padx=10, pady=5)

    # lnsigmau entry
    tk.Label(estimation_window, text="Equation for lnsigmau:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
    lnsigmau_entry = tk.Entry(estimation_window, width=50)
    lnsigmau_entry.grid(row=4, column=1, padx=10, pady=5)

    # lnmu label and entry (initially packed inside the grid, but will be hidden when necessary)
    lnmu_label = tk.Label(estimation_window, text="Equation for lnmu:")
    lnmu_entry = tk.Entry(estimation_window, width=50)

    # lnmu label and entry (initially packed inside the grid, but will be hidden when necessary)
    mu_label = tk.Label(estimation_window, text="Equation for mu:")
    mu_entry = tk.Entry(estimation_window, width=50)

    # Model name entry
    model_name_label = tk.Label(estimation_window, text="Model Name:")
    model_name_entry = tk.Entry(estimation_window, width=50)

    # Cost checkbox
    cost_label = tk.Label(estimation_window, text="Cost Function:")
    cost_var = tk.BooleanVar()
    cost_entry = tk.Checkbutton(estimation_window, text="Cost=True", variable=cost_var)

    # Starting values entry
    starting_values_label = tk.Label(estimation_window, text="Starting Values (comma-separated):")
    starting_values_entry = tk.Entry(estimation_window, width=50)

    # Estimate button
    estimate_model_button = tk.Button(estimation_window, text="Estimate Model", command=estimate_model)

    # Initially update the lnmu visibility based on default selection
    update_visibility()


def open_view_models():
    """Open a window to view all estimated models."""
    global models  # Access the global models dictionary
    view_window = tk.Toplevel()
    view_window.title("View Models")
    set_window_icon(view_window)

    # Create a listbox to display the model names
    tk.Label(view_window, text="Available Models:").pack(pady=5)
    models_listbox = tk.Listbox(view_window, width=50, height=10)
    models_listbox.pack(pady=5)

    # Add models to the listbox
    for model_name in models:
        models_listbox.insert(tk.END, model_name)

    def view_selected_model():
        """Display the selected model's output in a new window."""
        selected_model = models_listbox.get(models_listbox.curselection())
        if selected_model not in models:
            messagebox.showerror("Error", "No model selected or model not found.")
            return

        result = models[selected_model]

        # Create a new window to display model results
        viewer = tk.Toplevel()
        viewer.title(f"Model: {selected_model}")
        set_window_icon(viewer)

        tk.Label(viewer, text=f"Model: {selected_model}", font=("Arial", 14)).pack(pady=10)

        # Create the text widget
        text = tk.Text(viewer, wrap=tk.WORD, width=80, height=20)
        text.pack(padx=10, pady=10)

        # Enable copying by making the text widget editable for selection
        text.configure(state=tk.NORMAL)

        # Check if the result is a string or an object that needs to be converted to string
        try:
            model_output = str(result)
        except Exception as e:
            model_output = f"Error displaying result: {e}"

        # Insert the model output into the text box
        text.insert(tk.END, model_output)
        text.configure(state=tk.DISABLED)  # Disable editing while still allowing selection

        def copy_to_clipboard():
            """Copy the model output to the clipboard."""
            viewer.clipboard_clear()
            viewer.clipboard_append(model_output)
            viewer.update()  # Ensure the clipboard is updated
            messagebox.showinfo("Copied", "Model output copied to clipboard.")

        # Add a Copy button
        tk.Button(viewer, text="Copy to Clipboard", command=copy_to_clipboard).pack(pady=5)

    # View button
    tk.Button(view_window, text="View Selected Model", command=view_selected_model).pack(pady=5)

def show_about_window():
    """Display the 'About' window with author, version, and license info."""
    about_window = tk.Toplevel()
    about_window.title("About FronPy")

    # Set the window size and position
    about_window.geometry("500x400")
    about_window.resizable(False, False)

    tk.Label(
        about_window,
        text="FronPy",
        font=("Arial", 18, "bold"),
    ).pack(pady=10)

    tk.Label(
        about_window,
        text=f"Version: {__version__}",
        font=("Arial", 14),
    ).pack(pady=5)

    tk.Label(
        about_window,
        text="Author: Alexander D. Stead",
        font=("Arial", 12),
    ).pack(pady=5)

    email_label = tk.Label(
        about_window,
        text="Email: a.d.stead@leeds.ac.uk",
        font=("Arial", 12),
        fg="blue",  # Makes the email visually distinct
        cursor="hand2",
    )
    email_label.pack(pady=5)

    # Add clickable email functionality
    def open_email(event):
        import webbrowser
        webbrowser.open("mailto:a.d.stead@leeds.ac.uk")

    # Bind only the email label to the click event
    email_label.bind("<Button-1>", open_email)

    license_text = """MIT License

Copyright (c) 2023 Alexander D. Stead"""

    newblock = """
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
"""
    newblock = textwrap.dedent(newblock).replace("\n", " ")
    license_text = license_text + "\n" + "\n" + newblock
    

    newblock = """
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
"""
    newblock = textwrap.dedent(newblock).replace("\n", " ")
    license_text = license_text + "\n" + "\n" + newblock

    newblock = """
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
    newblock = textwrap.dedent(newblock).replace("\n", " ")
    license_text = license_text + "\n" + "\n" + newblock

    license_label = tk.Text(
        about_window,
        wrap="word",
        height=15,
        width=60,
        font=("Arial", 10),
        padx=10,
        pady=10,
    )
    license_label.insert("1.0", license_text)
    license_label.configure(state="disabled")  # Make it read-only
    license_label.pack(pady=10)

    # Close button
    tk.Button(about_window, text="Close", command=about_window.destroy).pack(pady=10)



def launch_gui():
    """Launch the main GUI."""
    
    # Splash screen setup
    splash_root = tk.Tk()
    splash_root.title("FronPy")
    
    # Remove the top bar (title bar)
    splash_root.overrideredirect(True)
    
    # Load the logo image
    logo_path = pkg_resources.resource_filename('fronpy', 'misc/FronPyGUI_splash.png')  # Update with your logo path
    logo = tk.PhotoImage(file=logo_path)
    logo = logo.subsample(2, 2)  # Resize logo as needed

    logo_label = tk.Label(splash_root, image=logo)
    logo_label.image = logo  # Keep reference to the image
    logo_label.pack()  # Get rid of padding for splash screen

    # Set the splash window size and background color
    splash_root.geometry("512x640")  # Increased size for splash screen
    splash_root.configure(bg='white')

    # Center the splash screen on the screen
    window_width = 512
    window_height = 640
    screen_width = splash_root.winfo_screenwidth()
    screen_height = splash_root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_left = int(screen_width / 2 - window_width / 2)
    splash_root.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')

    # Close the splash window after 3 seconds
    splash_root.after(3000, splash_root.destroy)  # 3000 ms = 3 seconds
    
    splash_root.mainloop()

    # Main GUI window setup
    root = tk.Tk()
    root.title("FronPy")
    set_window_icon(root)
    root.geometry("400x300")
    tk.Label(root, text="FronPy GUI", font=("Arial", 16)).pack(pady=10)

    tk.Button(root, text="Load Dataset", command=load_dataset).pack(pady=5)
    tk.Button(root, text="View Datasets", command=open_data_viewer).pack(pady=5)
    tk.Button(root, text="Estimate Model", command=open_model_estimation).pack(pady=5)
    tk.Button(root, text="View Models", command=open_view_models).pack(pady=5)
    tk.Button(root, text="About", command=show_about_window).pack(pady=5)

    root.mainloop()