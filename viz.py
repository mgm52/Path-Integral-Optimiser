from src.datamodules.datasets.opt_mini import OptMini

# If this file is run directly, run viz_pdf
if __name__ == "__main__":
    opt_mini = OptMini(1000, sigma=1)
    opt_mini.viz_pdf()