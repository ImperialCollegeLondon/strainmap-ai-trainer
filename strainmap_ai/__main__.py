from pathlib import Path

if __name__ == "__main__":
    from .scripts import train

    filenames = Path(__file__).parent.parent.parent / "Data"
    print(filenames)
    train(filenames)
