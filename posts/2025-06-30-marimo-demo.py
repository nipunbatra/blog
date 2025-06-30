import marimo

__generated_with = "0.9.34"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        # Marimo Interactive Notebook Demo

        This is a simple demonstration of marimo's interactive capabilities integrated with Quarto.

        Marimo is a reactive notebook where cells automatically update when their dependencies change.
        """
    )
    return


@app.cell
def __(mo):
    # Interactive slider for controlling parameters
    n_points = mo.ui.slider(
        start=10, 
        stop=100, 
        step=5, 
        value=50, 
        label="Number of points:"
    )
    n_points
    return n_points,


@app.cell
def __(mo):
    # Interactive dropdown for selecting function type
    func_type = mo.ui.dropdown(
        options=["sine", "cosine", "tangent"],
        value="sine",
        label="Function type:"
    )
    func_type
    return func_type,


@app.cell
def __(func_type, mo, n_points, np, plt):
    # This cell automatically updates when sliders change
    x = np.linspace(0, 2*np.pi, n_points.value)
    
    if func_type.value == "sine":
        y = np.sin(x)
        title = "Sine Wave"
    elif func_type.value == "cosine":
        y = np.cos(x)
        title = "Cosine Wave"
    else:  # tangent
        y = np.tan(x)
        title = "Tangent Wave"
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title(f"{title} with {n_points.value} points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    
    # Limit y-axis for tangent to avoid extreme values
    if func_type.value == "tangent":
        plt.ylim(-10, 10)
    
    plot = plt.gca()
    plt.show()
    
    mo.md(f"**Current settings:** {n_points.value} points, {func_type.value} function")
    return plot, title, x, y


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Data Analysis Example
        
        Let's create some sample data and analyze it interactively:
        """
    )
    return


@app.cell
def __(mo):
    # Interactive inputs for data generation
    sample_size = mo.ui.slider(
        start=100, 
        stop=1000, 
        step=50, 
        value=500,
        label="Sample size:"
    )
    noise_level = mo.ui.slider(
        start=0.1, 
        stop=2.0, 
        step=0.1, 
        value=0.5,
        label="Noise level:"
    )
    
    mo.hstack([sample_size, noise_level])
    return noise_level, sample_size


@app.cell
def __(mo, noise_level, np, plt, sample_size):
    # Generate and visualize data
    np.random.seed(42)  # For reproducibility
    
    # Generate sample data
    x_data = np.linspace(0, 10, sample_size.value)
    y_true = 2 * x_data + 1
    y_noisy = y_true + noise_level.value * np.random.randn(sample_size.value)
    
    # Simple linear regression
    coeffs = np.polyfit(x_data, y_noisy, 1)
    y_pred = np.polyval(coeffs, x_data)
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Data and fit
    plt.subplot(1, 2, 1)
    plt.scatter(x_data, y_noisy, alpha=0.6, s=1, label='Noisy data')
    plt.plot(x_data, y_true, 'r-', linewidth=2, label='True function')
    plt.plot(x_data, y_pred, 'g--', linewidth=2, label=f'Fitted line (slope={coeffs[0]:.2f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Linear Regression')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    residuals = y_noisy - y_pred
    plt.scatter(x_data, residuals, alpha=0.6, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Display statistics
    mse = np.mean(residuals**2)
    r_squared = 1 - np.sum(residuals**2) / np.sum((y_noisy - np.mean(y_noisy))**2)
    
    mo.md(f"""
    **Regression Results:**
    - Sample size: {sample_size.value}
    - Noise level: {noise_level.value}
    - Fitted slope: {coeffs[0]:.3f} (true slope: 2.000)
    - Fitted intercept: {coeffs[1]:.3f} (true intercept: 1.000)
    - Mean Squared Error: {mse:.3f}
    - R-squared: {r_squared:.3f}
    """)
    return (
        coeffs,
        mse,
        r_squared,
        residuals,
        x_data,
        y_noisy,
        y_pred,
        y_true,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Interactive Features

        Marimo's key features demonstrated here:

        1. **Reactive execution**: Change any slider above and watch all dependent cells update automatically
        2. **UI elements**: Sliders, dropdowns, and other interactive widgets
        3. **Automatic dependency tracking**: Cells know which other cells they depend on
        4. **Clean namespace**: No hidden state or execution order issues
        5. **Integration with Quarto**: This notebook renders seamlessly in your blog

        Try adjusting the parameters above to see the plots update in real-time!
        """
    )
    return


if __name__ == "__main__":
    app.run()